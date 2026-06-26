from pathlib import Path

import tempfile

import numpy as np

import time

import pyarrow as pa

from daq_tools import StreamingParquetWriter

def create_data(num_rows):
    tt = np.linspace(0, 600, num_rows)
    base_wave = np.sin(2 * np.pi * 10 * tt)
    channel_data = np.random.normal(0, 0.1, size=(num_rows, 8)) + base_wave[:, None]
    return channel_data

def bench_streaming_parquet(total_rows, batch_size):
    n_iter = total_rows // batch_size
    if n_iter * batch_size != total_rows:
        raise ValueError("total_rows must be a multiple of batch_size")
    batch_data = create_data(batch_size)

    schema = pa.schema({
        "v1": pa.float64(),
        "v2": pa.float64(),
        "v3": pa.float64(),
        "v4": pa.float64(),
        "v5": pa.float64(),
        "v6": pa.float64(),
        "v7": pa.float64(),
        "v8": pa.float64(),
    })

    with tempfile.TemporaryDirectory() as parent:
        writer = StreamingParquetWriter(
            Path(parent) / "tmp.parquet", schema=schema, batch_size=batch_size
        )

        tic = time.perf_counter()
        for _ in range(n_iter):
            for row in batch_data:
                writer.write(row)
        toc = time.perf_counter()
        writer.close()

        elapsed = toc - tic
        return elapsed

if __name__ == "__main__":
    total_rows = 10_000_000
    batch_size = 1000
    elapsed = bench_streaming_parquet(total_rows, batch_size)
    print(f"Wrote {total_rows} rows in {elapsed:.3f} seconds (batch size = {batch_size})")
