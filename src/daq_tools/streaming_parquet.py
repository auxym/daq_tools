import os
import threading
from typing import Sequence, Mapping, Any
from pathlib import Path
from collections import abc
from queue import Queue, ShutDown

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

type SomeRecord = Mapping[str, Any] | Sequence[Any]


class StreamingParquetWriter:
    """Writes data to Arrow IPC stream format and converts to parquet on close.

    Use `write` to append a single row. Rows are buffered until the buffer
    reaches `batch_size` rows, at which point a RecordBatch is queued for async
    write to the stream file by a background thread.

    On `close`, data is read from the stream file (.arrows) and written to the
    final .parquet file. The stream file is optionally deleted afterwards.

    In case of improper termination, valid data that was written to the stream
    file can be recovered with `stream_to_parquet`.

    This class uses a background thread for writes to the IPC stream file, but
    the public API is **NOT thread safe**. Do not share instances of this class in
    multiple threads.

    """

    path: Path
    schema: pa.Schema
    batch_size: int
    rowgroup_size: int
    do_fsync: bool
    metadata: Mapping[str, bytes | str]

    _buffer: list[SomeRecord]
    _ipc_path: Path
    _stream_writer: ipc.RecordBatchStreamWriter
    _stream_lock: threading.Lock
    _write_queue: Queue[list[SomeRecord]]
    _writer_thread: threading.Thread
    _closed: bool

    def __init__(
        self,
        path: os.PathLike | str,
        schema: pa.Schema,
        batch_size: int = 1000,
        rowgroup_size: int = 256 * 1024,
        fsync: bool = True,
        metadata: Mapping[str, bytes | str] = {},
    ):
        """
        Args:
            path: Path of the parquet file to be written.
            schema: Arrow schema defining the structure of the data.
            batch_size: Number of records written to the Arrow IPC Streaming file
                at a time. In case of a crash, this is the maximum data loss.
                Defaults to 1000.
            rowgroup_size: Size of the row groups written in the final parquet
                file. This can be tuned for performance reasons. Defaults to 256KB.
            fsync: If True (default), call fsync after every write to the Arrow
                IPC file for durability.
            metadata: Arbitrary key-value metadata that will be written to the
                parquet file metadata. Keys and values should be strings or bytes.
        """

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
        self._stream_lock = threading.Lock()

        self._write_queue = Queue(maxsize=10)

        # Open append-only stream
        self._sink = open(self._ipc_path, mode="ab", buffering=0)
        self._stream_writer = ipc.new_stream(self._sink, schema)

        self.metadata = metadata
        pq.write_metadata(
            self.schema.with_metadata(metadata), self._metadata_path(self.path)
        )

        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(delete_ipc=True)
        return False

    # ----------------------------
    # Public API
    # ----------------------------

    def write(self, record: SomeRecord):
        """Add a single record to the buffer.

        Records are buffered until the buffer reaches `batch_size` records, at which
        point a RecordBatch is queued for async write to the stream file (non-blocking).

        Args:
            record: A single record, either as a dict mapping column names to values,
                or as a sequence of values corresponding to schema fields in order.

        Raises:
            ValueError: If the length of the record does not match the schema length.
        """
        if self._closed:
            raise ValueError("Writer is closed")

        if len(record) != len(self.schema):
            raise ValueError(
                f"Length of record {len(record)} does not match schema length ({len(self.schema)})"
            )

        self._buffer.append(record)

        if len(self._buffer) >= self.batch_size:
            self._write_queue.put(self._buffer)
            self._buffer = []

    def write_batch(self, batch: pa.RecordBatch):
        """Write a pyarrow RecordBatch immediately and block until written.

        This bypasses the buffer and writes the batch directly to the stream file.
        Blocks until the write is complete.

        Args:
            batch: A pyarrow RecordBatch to write to the stream.
        """
        if self._closed:
            raise ValueError("Writer is closed")

        with self._stream_lock:
            self._stream_writer.write_batch(batch)
            self._sink.flush()
            if self.do_fsync:
                os.fsync(self._sink.fileno())

    def flush(self):
        """Flush accumulated rows in the buffer and block until written.

        This method flushes any buffered records to the stream file and
        blocks until the data has been written.
        """
        if self._buffer:
            batch = self._buffer_to_batch(self._buffer, self.schema)
            self._buffer = []
            self.write_batch(batch)

    def close(self, delete_ipc=False):
        """Flush remaining data, close IPC stream, and convert to Parquet.

        This method flushes any buffered records to the stream file, closes the
        stream writer, and then converts the Arrow IPC stream to a Parquet file.
        If delete_ipc is True, the temporary .arrows and metadata files are deleted.

        Args:
            delete_ipc: If True, delete the temporary Arrow IPC stream file after
                conversion. Defaults to False for recovery purposes.
        """

        # First, wait for any in-flight writes from previous async flushes
        # Non-immediate shutdown will allow get() until queue is empty, then
        # raise Shutdown, at which point the thread will exit.
        self._write_queue.shutdown(immediate=False)
        self._writer_thread.join()

        # Then flush any remaining buffer (blocking)
        self.flush()

        # Need to wait until all existing data is written before setting _closed,
        # otherwise we will generate an exception.
        self._closed = True

        # Close the IPC stream file
        with self._stream_lock:
            self._stream_writer.close()
            self._sink.close()

        # Convert IPC → Parquet
        self.stream_to_parquet(
            self._ipc_path,
            self.path,
            rowgroup_size=self.rowgroup_size,
            schema=self.schema,
            metadata=self.metadata,
        )

        if delete_ipc:
            for f in [self._ipc_path, self._metadata_path(self.path)]:
                try:
                    os.remove(f)
                except FileNotFoundError:
                    pass

    @property
    def ipc_path(self):
        """Get the path to the temporary Arrow IPC streaming file.

        Returns:
            Path: Path to the .arrows file where stream data is buffered.
        """

        return self._ipc_path

    @property
    def closed(self) -> bool:
        return self._closed

    # ----------------------------
    # Internal methods
    # ----------------------------

    @staticmethod
    def _buffer_to_batch(buffer: list[SomeRecord], schema: pa.Schema) -> pa.RecordBatch:
        """Convert buffer to a RecordBatch."""
        if isinstance(buffer[0], abc.Mapping):
            # Assume rows are {column_name: value} dicts
            return pa.RecordBatch.from_pylist(buffer)
        else:
            # Assume rows are plain sequences of ordered column values
            # Create a pa.Array for each column
            arrays = [
                pa.array(col, type=schema.types[i])
                for i, col in enumerate(zip(*buffer))
            ]
            return pa.RecordBatch.from_arrays(arrays, schema=schema)

    @staticmethod
    def _metadata_path(pq_path: os.PathLike | str) -> Path:
        return Path(pq_path).with_suffix(".parquet_metadata")

    def _writer_loop(self):
        """Background thread that writes batches from the queue."""
        while True:
            try:
                rows = self._write_queue.get()
            except ShutDown:
                break  # exit loop and stop thread

            batch = self._buffer_to_batch(rows, self.schema)
            with self._stream_lock:
                self._stream_writer.write_batch(batch)

    @staticmethod
    def _try_read_metadata_file(
        file_path: os.PathLike,
    ) -> tuple[pa.Schema | None, Mapping[str, Any]]:
        schema, metadata = None, {}
        try:
            with pq.ParquetFile(file_path) as pf:
                schema = pf.schema_arrow
                metadata = {
                    k: v
                    for k, v in pf.metadata.metadata.items()
                    if not k.startswith(b"ARROW:")
                }
        except FileNotFoundError:
            pass

        return schema, metadata

    @staticmethod
    def _create_parquet_writer(
        filepath: os.PathLike | str,
        schema: pa.Schema,
        metadata: Mapping[str, Any] | None,
    ) -> pq.ParquetWriter:
        writer = pq.ParquetWriter(
            filepath,
            schema=schema,
            data_page_version="2.0",
            write_page_checksum=True,
        )
        if metadata:
            writer.add_key_value_metadata(metadata)
        return writer

    @classmethod
    def stream_to_parquet(
        cls,
        ipc_path: os.PathLike | str,
        parquet_path: os.PathLike | str,
        rowgroup_size: int = 0,
        detect_metadata_file=True,
        schema: pa.Schema = None,
        metadata: Mapping[str, str | bytes] = {},
    ) -> int:
        """Convert Arrow IPC streaming file to Parquet format.

        Reads batches from the Arrow IPC streaming file and writes them to a Parquet
        file in a memory-efficient, streaming manner. Useful for recovering data
        from an .arrows file after improper termination.

        Args:
            ipc_path: Path to the Arrow IPC streaming file (.arrows).
            parquet_path: Path for the output Parquet file.
            rowgroup_size: Minimum number of rows for row groups in the Parquet
                file. If `0`, (default), each batch becomes a separate row group.
            detect_metadata_file: If True, attempt to read schema and metadata from
                a .parquet_metadata file if schema is not provided. Defaults to True.
            schema: Arrow schema to use for the Parquet file. If None and
                detect_metadata_file is True, will attempt to read from metadata file.
            metadata: Key-value metadata to write to the Parquet file. If schema
                is provided and metadata is empty, will attempt to read from metadata
                file if detect_metadata_file is True.

        Returns:
            int: Total number of records written to the Parquet file.
        """

        def write_row_group(writer, batches):
            if len(batches) == 1:
                writer.write_batch(batches[0])
            else:
                table = pa.Table.from_batches(batches)
                writer.write_table(table, row_group_size=table.num_rows)

        writer = None
        batch_rows = 0
        total_rows = 0
        batches = []
        at_end = False

        # Try to read schema and optional metadata from a metadata file written by this class
        if schema is None and detect_metadata_file:
            schema, metadata = cls._try_read_metadata_file(
                cls._metadata_path(parquet_path)
            )

        if schema is not None:
            writer = cls._create_parquet_writer(parquet_path, schema, metadata)

        with open(ipc_path, "rb") as f:
            reader = ipc.open_stream(f)

            while not at_end:
                try:
                    batch = reader.read_next_batch()
                except Exception:
                    at_end = True
                else:
                    if writer is None:
                        writer = cls._create_parquet_writer(
                            parquet_path, batch.schema, metadata
                        )
                    batches.append(batch)
                    batch_rows += batch.num_rows

                if batches and (at_end or batch_rows >= rowgroup_size):
                    write_row_group(writer, batches)
                    total_rows += batch_rows
                    batch_rows = 0
                    batches = []

            return total_rows
