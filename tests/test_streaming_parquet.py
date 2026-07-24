from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.ipc as ipc

from daq_tools import StreamingParquetWriter


def test_write_dict_records(tmp_path):
    """Test writing records as dictionaries."""
    schema = pa.schema([("id", pa.int64()), ("name", pa.string())])
    path = tmp_path / "test.parquet"
    with StreamingParquetWriter(path, schema, batch_size=3) as writer:
        writer.write({"id": 1, "name": "Alice"})
        writer.write({"id": 2, "name": "Bob"})
        writer.write({"id": 3, "name": "Charlie"})

    assert path.exists()
    table = pq.read_table(path)
    assert table.num_rows == 3
    assert table.column("id").to_pylist() == [1, 2, 3]
    assert table.column("name").to_pylist() == ["Alice", "Bob", "Charlie"]


def test_write_sequence_records(tmp_path):
    """Test writing records as sequences (lists/tuples)."""
    schema = pa.schema([("id", pa.int64()), ("value", pa.float64())])
    path = tmp_path / "test.parquet"
    with StreamingParquetWriter(path, schema, batch_size=2) as writer:
        writer.write((1, 1.5))
        writer.write((2, 2.5))
        writer.write((3, 3.5))

    assert path.exists()
    table = pq.read_table(path)
    assert table.num_rows == 3
    assert table.column("id").to_pylist() == [1, 2, 3]
    assert table.column("value").to_pylist() == [1.5, 2.5, 3.5]


def test_auto_flush_on_batch_size(tmp_path):
    """Test that buffer auto-flushes when batch_size is reached."""
    schema = pa.schema([("x", pa.int64())])
    path = tmp_path / "test.parquet"

    with StreamingParquetWriter(path, schema, batch_size=3) as writer:
        # Write exactly batch_size records
        for i in range(3):
            writer.write({"x": i})

    assert path.exists()
    table = pq.read_table(path)
    assert table.num_rows == 3


def test_flushing_partial_buffer_on_close(tmp_path):
    """Test that partial buffer is flushed on close."""
    schema = pa.schema([("val", pa.int64())])
    path = tmp_path / "test.parquet"

    with StreamingParquetWriter(path, schema, batch_size=100) as writer:
        # Write fewer records than batch_size
        for i in range(5):
            writer.write({"val": i})

    assert path.exists()
    table = pq.read_table(path)
    assert table.num_rows == 5


def test_ipc_path_derivation(tmp_path):
    """Test that IPC path is derived correctly from parquet path."""
    schema = pa.schema([("a", pa.int64())])
    path = tmp_path / "output.parquet"
    ipc_path = tmp_path / "output.arrows"

    with StreamingParquetWriter(path, schema) as writer:
        writer.write({"a": 1})
        assert writer.ipc_path == ipc_path

    assert path.exists()
    assert not ipc_path.exists()


def test_ipc_path_non_parquet_suffix(tmp_path):
    """Test IPC path when original path doesn't end in .parquet."""
    schema = pa.schema([("a", pa.int64())])
    path = tmp_path / "datastream"
    ipc_path = tmp_path / "datastream.arrows"

    with StreamingParquetWriter(path, schema) as writer:
        writer.write({"a": 1})
        assert writer.ipc_path == ipc_path

    assert path.exists()
    assert not ipc_path.exists()


def test_write_parquet_from_ipc_static(tmp_path):
    """Test the static method for converting IPC to parquet."""
    schema = pa.schema([("key", pa.string()), ("num", pa.int64())])
    ipc_path = tmp_path / "stream.arrows"
    parquet_path = tmp_path / "output.parquet"

    # Create a valid IPC streaming file
    with open(ipc_path, "wb") as sink:
        with ipc.new_stream(sink, schema) as writer:
            batch = pa.RecordBatch.from_pylist(
                [
                    {"key": "a", "num": 1},
                    {"key": "b", "num": 2},
                ]
            )
            writer.write_batch(batch)

    count = StreamingParquetWriter.stream_to_parquet(str(ipc_path), parquet_path)

    assert count == 2
    assert parquet_path.exists()

    table = pq.read_table(parquet_path)
    assert table.num_rows == 2
    assert table.schema == schema


def test_write_parquet_from_ipc_multiple_batches(tmp_path):
    """Test IPC conversion with multiple batches and rowgroup_size."""
    schema = pa.schema([("x", pa.int64())])
    ipc_path = tmp_path / "stream.arrows"
    parquet_path = tmp_path / "output.parquet"

    # Create IPC with multiple batches
    with open(ipc_path, "wb") as sink:
        with ipc.new_stream(sink, schema) as writer:
            for _ in range(3):
                batch = pa.RecordBatch.from_pylist([{"x": 1}, {"x": 2}])
                writer.write_batch(batch)

    # Use small rowgroup_size to trigger multiple writes
    count = StreamingParquetWriter.stream_to_parquet(
        ipc_path, parquet_path, rowgroup_size=2
    )

    assert count == 6
    table = pq.read_table(parquet_path)
    assert table.num_rows == 6


def test_context_manager(tmp_path):
    """Test using StreamingParquetWriter as a context manager."""
    schema = pa.schema([("data", pa.int64())])
    path = tmp_path / "ctx.parquet"

    with StreamingParquetWriter(path, schema) as writer:
        writer.write({"data": 42})

    assert path.exists()
    table = pq.read_table(path)
    assert table.num_rows == 1


def test_delete_ipc_on_close(tmp_path):
    """Test that IPC file is deleted when delete_ipc=True."""
    schema = pa.schema([("x", pa.int64())])
    path = tmp_path / "test.parquet"
    ipc_path = tmp_path / "test.arrows"

    writer = StreamingParquetWriter(path, schema)
    writer.write({"x": 1})
    writer.close(delete_ipc=True)

    assert path.exists()
    assert not ipc_path.exists()


def test_multiple_column_types(tmp_path):
    """Test writing various Arrow column types."""
    schema = pa.schema(
        [
            ("int_val", pa.int64()),
            ("float_val", pa.float64()),
            ("str_val", pa.string()),
            ("bool_val", pa.bool_()),
        ]
    )
    path = tmp_path / "types.parquet"

    with StreamingParquetWriter(path, schema) as writer:
        writer.write(
            {
                "int_val": 42,
                "float_val": 3.14,
                "str_val": "hello",
                "bool_val": True,
            }
        )

    table = pq.read_table(path)
    assert table.column("int_val").to_pylist() == [42]
    assert table.column("float_val").to_pylist() == [3.14]
    assert table.column("str_val").to_pylist() == ["hello"]
    assert table.column("bool_val").to_pylist() == [True]


def test_write_wrong_sequence_length(tmp_path):
    """Test that exception is raised when sequence data has wrong length."""
    schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
    path = tmp_path / "test.parquet"
    with StreamingParquetWriter(path, schema, batch_size=10) as writer:
        # Write correct length
        writer.write((1, "one"))

        # Write wrong length - raises on write
        with pytest.raises(ValueError, match="Length of record"):
            writer.write((2, "two", "extra"))


def test_write_batch(tmp_path):
    """Test writing pre-built RecordBatch directly."""
    schema = pa.schema([("x", pa.int64()), ("y", pa.float64())])
    path = tmp_path / "test.parquet"

    batch = pa.RecordBatch.from_arrays([[1, 2, 3], [1.0, 2.0, 3.0]], names=["x", "y"])
    with StreamingParquetWriter(path, schema) as writer:
        writer.write_batch(batch)

    assert path.exists()
    table = pq.read_table(path)
    assert table.num_rows == 3
    assert table.column("x").to_pylist() == [1, 2, 3]
    assert table.column("y").to_pylist() == [1.0, 2.0, 3.0]


def test_metadata_on_close(tmp_path):
    """Test that metadata is written to parquet file on close."""
    schema = pa.schema([("id", pa.int64())])
    metadata = {"created_by": "test_suite", "version": "1.0"}
    path = tmp_path / "test.parquet"
    with StreamingParquetWriter(path, schema, metadata=metadata) as writer:
        writer.write({"id": 1})

    assert path.exists()
    with pq.ParquetFile(path) as pf:
        file_metadata = pf.metadata.metadata
        assert file_metadata is not None
        assert file_metadata[b"created_by"] == b"test_suite"
        assert file_metadata[b"version"] == b"1.0"


def test_metadata_recovery_after_crash(tmp_path):
    """Test metadata recovery via stream_to_parquet after simulated crash."""
    schema = pa.schema([("val", pa.int64())])
    metadata = {"source": "crash_recovery", "run_id": "12345"}
    path = tmp_path / "output.parquet"
    ipc_path = tmp_path / "output.arrows"
    metadata_path = tmp_path / "output.parquet_metadata"

    # Create writer and write data (but don't close - simulate crash)
    writer = StreamingParquetWriter(path, schema, metadata=metadata, batch_size=1000)
    writer.write({"val": 42})

    # Wait for data to be written to IPC (non-blocking write with batch_size=1000)
    writer.flush()

    # Signal shutdown to stop the writer thread before closing stream
    writer._write_queue.shutdown(immediate=False)
    writer._writer_thread.join()

    # Verify metadata file was created
    assert metadata_path.exists()

    # Close the stream writer manually (simulating crash recovery process)
    writer._stream_writer.close()
    writer._sink.close()

    # Now use stream_to_parquet to recover
    count = StreamingParquetWriter.stream_to_parquet(
        ipc_path, path, detect_metadata_file=True
    )

    assert count == 1
    assert path.exists()

    # Verify metadata was recovered
    with pq.ParquetFile(path) as pf:
        file_metadata = pf.metadata.metadata
        assert file_metadata is not None
        assert file_metadata[b"source"] == b"crash_recovery"
        assert file_metadata[b"run_id"] == b"12345"


def test_write_batch_blocks(tmp_path):
    """Test that write_batch blocks until data is written."""
    import time

    schema = pa.schema([("x", pa.int64())])
    path = tmp_path / "test.parquet"

    batch = pa.RecordBatch.from_arrays([[1, 2, 3]], names=["x"])
    with StreamingParquetWriter(path, schema) as writer:
        start = time.perf_counter()
        writer.write_batch(batch)
        elapsed = time.perf_counter() - start

        # write_batch should block until completed
        assert elapsed < 0.1, "write_batch took too long - should be synchronous"

    table = pq.read_table(path)
    assert table.num_rows == 3


def test_flush_blocks(tmp_path):
    """Test that flush blocks until data is written."""
    import time

    schema = pa.schema([("x", pa.int64())])
    path = tmp_path / "test.parquet"

    with StreamingParquetWriter(path, schema, batch_size=1000) as writer:
        for i in range(5):
            writer.write({"x": i})

        start = time.perf_counter()
        writer.flush()
        elapsed = time.perf_counter() - start

        # flush should block until completed
        assert elapsed < 0.5, (
            f"flush took too long - should be synchronous ({elapsed}s)"
        )

    table = pq.read_table(path)
    assert table.num_rows == 5


def test_write_non_blocking(tmp_path):
    """Test that write is non-blocking when batch_size is reached."""
    import time

    schema = pa.schema([("x", pa.int64())])
    path = tmp_path / "test.parquet"

    with StreamingParquetWriter(path, schema, batch_size=3) as writer:
        start = time.perf_counter()
        for i in range(3):
            writer.write({"x": i})
        elapsed = time.perf_counter() - start

        # write should not block significantly when batch is flushed to queue
        assert elapsed < 0.1, "write took too long - should be non-blocking"

    table = pq.read_table(path)
    assert table.num_rows == 3


def test_thread_exception_on_invalid_type(tmp_path):
    """Test that exception in writer thread is propagated to caller."""
    schema = pa.schema([("x", pa.int64())])
    path = tmp_path / "test.parquet"

    with StreamingParquetWriter(path, schema, batch_size=3) as writer:
        # Write invalid record - string cannot convert to int64
        writer.write({"x": "not_an_int"})
        writer.write({"x": 1})
        writer.write({"x": 2})

        # Flushing should raise the exception from the writer thread
        with pytest.raises(pa.ArrowTypeError):
            writer.flush()


def test_thread_exception_on_write(tmp_path):
    """Test that thread exception is raised on subsequent write calls."""
    schema = pa.schema([("x", pa.int64())])
    path = tmp_path / "test.parquet"

    with StreamingParquetWriter(path, schema, batch_size=3) as writer:
        # Write records to trigger async flush with invalid data (first in buffer)
        writer.write({"x": "bad_value"})
        writer.write({"x": 1})
        writer.write({"x": 2})

        # Give thread time to process and hit the exception
        import time

        time.sleep(0.1)

        # Next write should raise the cached exception
        with pytest.raises(pa.ArrowTypeError):
            writer.write({"x": 3})


def test_thread_exception_on_close(tmp_path):
    """Test that thread exception is raised when calling close."""
    schema = pa.schema([("x", pa.int64())])
    path = tmp_path / "test.parquet"
    writer = StreamingParquetWriter(path, schema, batch_size=3)

    # Write invalid record at start of buffer
    writer.write({"x": "invalid"})
    writer.write({"x": 1})
    writer.write({"x": 2})

    # close() should raise the exception since batch will be flushed
    with pytest.raises(pa.ArrowTypeError):
        writer.close()

    assert writer.closed


def test_thread_exception_with_sequence_record(tmp_path):
    """Test thread exception when sequence record has wrong type."""
    schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
    path = tmp_path / "test.parquet"
    with StreamingParquetWriter(path, schema, batch_size=3) as writer:
        writer.write((1, "one"))
        writer.write((2, "two"))

        # Third record has wrong type (int where string expected)
        writer.write((3, 123))

        with pytest.raises(pa.ArrowTypeError):
            writer.flush()


def test_stream_compression(tmp_path):
    """Test that stream_compression option is applied to writer_options."""
    schema = pa.schema([("id", pa.int64()), ("name", pa.string())])
    path = tmp_path / "test.parquet"
    codecs = ["lz4", "zstd", pa.Codec("zstd", compression_level=3), None]
    for c in codecs:
        # pyarrow doesn't provide a way to check compression on a IPC stream file, so we just smoke
        # test the writer and check that nothing crashes.
        with StreamingParquetWriter(
            path, schema, batch_size=100, stream_compression=c
        ) as writer:
            for i in range(100):
                writer.write({"id": i, "name": f"name_{i}"})
            writer.flush()
            assert writer.ipc_path.exists()


def test_parquet_compression(tmp_path):
    """Test stream_to_parquet static method with compression options."""
    schema = pa.schema([("key", pa.string()), ("num", pa.int64())])
    ipc_path = tmp_path / "stream.arrows"
    parquet_path = tmp_path / "output.parquet"

    with open(ipc_path, "wb") as sink:
        with ipc.new_stream(sink, schema) as writer:
            batch = pa.RecordBatch.from_pylist(
                [{"key": "a", "num": 1}, {"key": "b", "num": 2}]
            )
            writer.write_batch(batch)

    count = StreamingParquetWriter.stream_to_parquet(
        ipc_path, parquet_path, writer_options={"compression": "lz4"}
    )

    assert count == 2
    with pq.ParquetFile(parquet_path) as pf:
        assert pf.metadata.row_group(0).column(0).compression == "LZ4"
