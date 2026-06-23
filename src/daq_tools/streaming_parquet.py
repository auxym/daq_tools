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
    Writes streaming data to Arrow IPC format and convert to parquet on close.

    Args:
        path: Path of the parquet file to be written.
        schema: Arrow schema
        batch_size: Number of records written to the Arrow IPC Streaming file
            at a time. In case of a crash, this is the maximum data loss.
        rowgroup_size: Size of the row groups written in the final parquet
            file. This can be tuned for performance reasons.

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
        self._sink = open(self.ipc_path, "ab")
        self._writer = ipc.new_stream(self._sink, schema)

    # ----------------------------
    # Public API
    # ----------------------------

    def write(self, record: SomeRecord):
        """
        Add a single record (dict or sequence).
        """
        self._buffer.append(record)

        if len(self._buffer) >= self.batch_size:
            self._flush()

    def close(self, delete_ipc=False):
        """
        Flush remaining data, close IPC, convert to Parquet.
        """
        if self._buffer:
            self._flush()

        self._writer.close()
        # self._sink.close()

        # Convert IPC → Parquet
        self.write_parquet_from_ipc(
            self.ipc_path, self.path, rowgroup_size=self.rowgroup_size
        )

        if delete_ipc:
            try:
                os.remove(self.path)
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
            column_data = list(zip(*self._buffer))
            batch = pa.RecordBatch.from_pydict(
                {col: column_data[i] for i, col in enumerate(self._columns)}
            )

        self._writer.write_batch(batch)

        # durability
        self._sink.flush()
        if self.do_fsync:
            os.fsync(self._sink.fileno())

        # clear buffer
        self._buffer.clear()

    @staticmethod
    def write_parquet_from_ipc(
        ipc_path: Path | str, parquet_path: Path | str, rowgroup_size: None | int = None
    ) -> int:
        """
        Recover valid batches from Arrow IPC streaming file and write to parquet.
        (streamed, memory efficient).
        """
        with open(ipc_path, "rb") as f:
            reader = ipc.open_stream(f)

            writer = None
            n_records = 0
            total_records = 0
            batches = []
            at_end = False

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
                if writer and batches and enough_rows:
                    if len(batches) == 1:
                        writer.write_batch(batches[0])
                    else:
                        table = pa.Table.from_batches(batches)
                        writer.write_table(table, row_group_size=table.num_rows)
                    total_records += n_records
                    n_records = 0
                    batches = []

            if writer is not None:
                writer.close()

            return total_records
