import os
from typing import Sequence, Mapping, Any
from pathlib import Path
from collections import abc

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

type SomeRecord = Mapping[str, Any] | Sequence[Any]


class StreamingParquetWriter:
    """
    Writes data to Arrow IPC stream format and convert to parquet on close.

    Use `write` to append a single row. Rows are buffered until the buffer
    reaches `batch_size` rows, at which point a RecordBatch is written to the
    a temporary stream file (Arrow Streaming IPC format).

    On `close`, data is read from the stream file (.arrows) and written to the
    final .parquet file. The stream file is optionally deleted afterwards.

    In case of improper termination, valid data that was written to the stream
    file can be recovered with `stream_to_parquet`.

    Args:
        path: Path of the parquet file to be written.
        schema: Arrow schema
        batch_size: Number of records written to the Arrow IPC Streaming file
            at a time. In case of a crash, this is the maximum data loss.
        rowgroup_size: Size of the row groups written in the final parquet
            file. This can be tuned for performance reasons.
        fsync: If `True` (default), call fsync after every write to the Arrow
            IPC file.

    Attributes:
        ipc_path: Path of the temporary Arrow IPC Streaming file.

    """

    path: Path
    schema: pa.schema
    batch_size: int
    rowgroup_size: int
    do_fsync: bool

    _buffer: list[SomeRecord]
    _columns: list
    _ipc_path: Path

    def __init__(
        self,
        path: str | Path,
        schema: pa.Schema,
        batch_size: int = 1000,
        rowgroup_size: int = 256 * 1024,
        fsync: bool = True,
    ):
        self.path = Path(path)
        if self.path.suffix == ".parquet":
            self._ipc_path = self.path.with_suffix(".arrows")
        else:
            self._ipc_path = Path(str(self.path) + ".arrows")

        self.schema = schema
        self.batch_size = batch_size
        self.rowgroup_size = rowgroup_size
        self.do_fsync = fsync

        self._buffer = []
        self._columns = [field.name for field in schema]

        # Open append-only stream
        self._sink = open(self._ipc_path, "ab")
        self._writer = ipc.new_stream(self._sink, schema)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(delete_ipc=True)
        return False

    # ----------------------------
    # Public API
    # ----------------------------

    def write(self, record: SomeRecord):
        """Add a single record (dict or sequence).

        Records are buffered until the buffer reaches `batch_size` buffers, at which
        point a RecordBatch is written to the stream file.
        """
        if len(record) != len(self.schema):
            raise ValueError(
                f"Length of record {len(record)} does not match schema length ({len(self.schema)})"
            )

        self._buffer.append(record)

        if len(self._buffer) >= self.batch_size:
            self._flush()

    def write_batch(self, batch: pa.RecordBatch):
        """Write a pyarrow RecordBatch immediately."""
        self._writer.write_batch(batch)

        # durability
        self._sink.flush()
        if self.do_fsync:
            os.fsync(self._sink.fileno())

    def close(self, delete_ipc=False):
        """
        Flush remaining data, close IPC, convert to Parquet.
        """
        if self._buffer:
            self._flush()

        self._writer.close()
        self._sink.close()

        # Convert IPC → Parquet
        self.stream_to_parquet(
            self._ipc_path, self.path, rowgroup_size=self.rowgroup_size
        )

        if delete_ipc:
            try:
                os.remove(self._ipc_path)
            except FileNotFoundError:
                pass

    @property
    def ipc_path(self):
        return self._ipc_path

    # ----------------------------
    # Internal methods
    # ----------------------------

    def _flush(self):
        """
        Convert buffered rows into Arrow arrays and write a batch.
        """
        # Convert buffer into columnar structure
        if isinstance(self._buffer[0], abc.Mapping):
            # Assume _buffer contains name: value dicts
            batch = pa.RecordBatch.from_pylist(self._buffer)
        else:
            # Assume _buffer contains list of sequences of data
            arrays = [
                pa.array(col, type=self.schema.types[i])
                for i, col in enumerate(zip(*self._buffer))
            ]
            batch = pa.RecordBatch.from_arrays(arrays, schema=self.schema)

        self._buffer.clear()
        self.write_batch(batch)

    @staticmethod
    def stream_to_parquet(
        ipc_path: Path | str, parquet_path: Path | str, rowgroup_size: None | int = None
    ) -> int:
        """
        Recover valid batches from Arrow IPC streaming file and write to parquet.
        (streamed, memory efficient).
        """

        def write_row_group(writer, batches):
            if len(batches) == 1:
                writer.write_batch(batches[0])
            else:
                table = pa.Table.from_batches(batches)
                writer.write_table(table, row_group_size=table.num_rows)

        writer = None
        n_records = 0
        total_records = 0
        batches = []
        at_end = False

        with open(ipc_path, "rb") as f:
            reader = ipc.open_stream(f)

            while not at_end:
                try:
                    batch = reader.read_next_batch()
                except Exception:
                    at_end = True
                else:
                    if writer is None:
                        writer = pq.ParquetWriter(parquet_path, batch.schema)
                    batches.append(batch)
                    n_records += batch.num_rows

                enough_rows = (rowgroup_size is None) or (n_records >= rowgroup_size)
                if writer and batches and (enough_rows or at_end):
                    write_row_group(writer, batches)
                    total_records += n_records
                    n_records = 0
                    batches = []

            return total_records
