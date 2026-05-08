"""Tests for the Celeborn shuffle backend.

These tests are organized in two layers, mirroring the convention used by
``tests/dataframe/test_shuffles.py``:

1. **Configuration-layer tests** — exercise the Python config surface
   (`daft.execution_config_ctx`) for the new Celeborn fields. They verify that
   the config validation accepts the new ``shuffle_algorithm="celeborn"`` value,
   accepts/rejects the new Celeborn-specific options, and round-trips them onto
   ``DaftExecutionConfig``. These tests run against any runner.

2. **End-to-end tests** — exercise the full Map → push → fetch → Reduce path
   through a real `df.repartition(...).collect()` invocation. These are gated
   by a ``celeborn_shuffle_ctx`` fixture and are currently **skipped**:
   the production-side Celeborn SDK and the Python binding for
   ``MockCelebornClient`` are not yet available, so there is no way to inject a
   client into the running pipeline from Python. The test bodies are kept
   intact so that, once the SDK and Python binding land, removing the
   ``pytest.mark.skip`` decorator is sufficient to enable the suite.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from functools import partial

import numpy as np
import pyarrow as pa
import pytest

import daft
from daft.io._generator import read_generator
from daft.recordbatch.recordbatch import RecordBatch
from tests.conftest import get_tests_daft_runner_name

###
# Helpers (mirrors the data generators used by `test_shuffles.py`).
###


def _generate(partition_id: int, num_rows: int, bytes_per_row: int):
    data = {
        "ids": np.arange(num_rows) + partition_id * num_rows,
        "ints": np.random.randint(0, num_rows, num_rows, dtype=np.uint64),
        "bytes": pa.array(
            [np.random.bytes(bytes_per_row) for _ in range(num_rows)],
            type=pa.binary(bytes_per_row),
        ),
    }
    yield RecordBatch.from_pydict(data)


def _generator(
    num_partitions: int,
    num_rows_fn: Callable[[], int],
    bytes_per_row_fn: Callable[[], int],
):
    for partition_id in range(num_partitions):
        num_rows = num_rows_fn()
        bytes_per_row = bytes_per_row_fn()
        yield partial(_generate, partition_id, num_rows, bytes_per_row)


###
# Fixtures.
###


@pytest.fixture(scope="function")
def celeborn_shuffle_ctx():
    """Context manager for end-to-end Celeborn shuffle tests.

    Yields a ``daft.execution_config_ctx`` configured for the Celeborn backend.
    A throwaway temporary directory is created so that any local spill paths
    referenced by the Celeborn client are isolated per-test.

    Note: this fixture only configures Daft. It does **not** start a Celeborn
    cluster nor inject a `MockCelebornClient`. End-to-end tests that depend on
    this fixture must therefore be marked ``skip`` until the Python binding for
    ``MockCelebornClient`` (or a real Celeborn cluster) is wired up.
    """

    @contextmanager
    def _ctx(lm_host: str = "localhost", lm_port: int = 9097, app_id: str = "test-app"):
        with tempfile.TemporaryDirectory() as _tmp_dir:
            with daft.execution_config_ctx(
                shuffle_algorithm="celeborn",
                celeborn_lm_host=lm_host,
                celeborn_lm_port=lm_port,
                celeborn_app_id=app_id,
                celeborn_compression="lz4",
                celeborn_push_data_timeout_ms=30_000,
                celeborn_fetch_data_timeout_ms=30_000,
            ) as ctx:
                yield ctx

    return _ctx


###
# Configuration-layer tests.
#
# These run on every runner. They pin the Python config surface so that any
# accidental rename or removal of a Celeborn option is caught by a fast,
# environment-free unit test.
###


def test_celeborn_shuffle_algorithm_is_accepted():
    """`shuffle_algorithm="celeborn"` must be a valid value.

    It was added to the validation whitelist alongside `flight_shuffle`.
    """
    with daft.execution_config_ctx(
        shuffle_algorithm="celeborn",
        celeborn_lm_host="host",
        celeborn_lm_port=9097,
    ):
        # Reaching here without ValueError is the assertion.
        pass


def test_celeborn_lm_host_must_not_be_empty():
    """An empty celeborn_lm_host is a configuration mistake.

    The validator in daft-config/src/python.rs must surface it as ValueError so
    users get a clear failure at config time instead of a runtime crash.
    """
    with pytest.raises(ValueError, match="celeborn_lm_host"):
        with daft.execution_config_ctx(
            shuffle_algorithm="celeborn",
            celeborn_lm_host="   ",
        ):
            pass


def test_celeborn_compression_whitelist():
    """Only the codecs supported by the Celeborn worker are accepted."""
    # Allowed values round-trip cleanly.
    for codec in ("lz4", "zstd", "none"):
        with daft.execution_config_ctx(
            shuffle_algorithm="celeborn",
            celeborn_lm_host="host",
            celeborn_lm_port=9097,
            celeborn_compression=codec,
        ):
            pass

    # Anything else must be rejected.
    with pytest.raises(ValueError, match="celeborn_compression"):
        with daft.execution_config_ctx(
            shuffle_algorithm="celeborn",
            celeborn_lm_host="host",
            celeborn_lm_port=9097,
            celeborn_compression="snappy",
        ):
            pass


def test_celeborn_timeouts_are_accepted():
    """User-provided timeouts must be accepted without error.

    We exercise both push and fetch timeouts in a single ctx because
    they share the same code path in `with_config_values`.

    Note: we only assert that the setter succeeds (no TypeError / ValueError).
    Getter-level round-trip assertions are deferred until `PyDaftExecutionConfig`
    exposes `#[getter]` attributes for these fields on the Rust side.
    """
    with daft.execution_config_ctx(
        shuffle_algorithm="celeborn",
        celeborn_lm_host="host",
        celeborn_lm_port=9097,
        celeborn_push_data_timeout_ms=12_345,
        celeborn_fetch_data_timeout_ms=67_890,
    ):
        # Reaching here without TypeError is the assertion — the values were
        # accepted by the Rust-side `with_config_values` validator.
        pass


def test_invalid_shuffle_algorithm_is_rejected():
    """Sanity: a typo in shuffle_algorithm must still fail loudly.

    This guards against the Celeborn whitelist update accidentally widening the
    accepted set to include arbitrary strings.
    """
    with pytest.raises(ValueError, match="shuffle_algorithm"):
        with daft.execution_config_ctx(shuffle_algorithm="celebornn"):
            pass


###
# End-to-end tests.
#
# Skipped until the Python binding for `MockCelebornClient` is exposed and the
# `BuilderContext::with_celeborn_client(...)` setter is reachable from Python.
# The test bodies follow the same shape as `test_pre_shuffle_merge_*` so the
# diff to enable them is mechanical: drop the `skip` decorator.
###


_E2E_SKIP_REASON = (
    "End-to-end Celeborn shuffle tests require either a live Celeborn cluster or a "
    "Python-exposed MockCelebornClient that can be injected via "
    "`BuilderContext::with_celeborn_client`. Neither is available yet — the binding "
    "will be added once the Celeborn-side SDK is finalized."
)


@pytest.mark.skip(reason=_E2E_SKIP_REASON)
@pytest.mark.skipif(
    get_tests_daft_runner_name() != "ray",
    reason="shuffle tests are meant for the ray runner",
)
@pytest.mark.parametrize(
    "input_partitions, output_partitions",
    [(20, 20), (20, 1), (20, 50)],
)
def test_celeborn_shuffle_repartition_small(celeborn_shuffle_ctx, input_partitions, output_partitions):
    """Repartition a small dataset through Celeborn and verify row count.

    Mirrors `test_pre_shuffle_merge_small_partitions`.
    """

    def num_rows_fn():
        return output_partitions

    def bytes_per_row_fn():
        return 1

    with celeborn_shuffle_ctx():
        df = (
            read_generator(
                _generator(input_partitions, num_rows_fn, bytes_per_row_fn),
                schema=daft.Schema._from_field_name_and_types(
                    [
                        ("ids", daft.DataType.uint64()),
                        ("ints", daft.DataType.uint64()),
                        ("bytes", daft.DataType.binary()),
                    ]
                ),
            )
            .repartition(output_partitions, "ints")
            .collect()
        )
        assert len(df) == input_partitions * output_partitions


@pytest.mark.skip(reason=_E2E_SKIP_REASON)
@pytest.mark.skipif(
    get_tests_daft_runner_name() != "ray",
    reason="shuffle tests are meant for the ray runner",
)
def test_celeborn_shuffle_groupby_aggregate(celeborn_shuffle_ctx):
    """A groupby-aggregate exercises the repartition-reduce path.

    This catches regressions where the read source emits empty/oversized
    partitions that the aggregator cannot consume.
    """
    with celeborn_shuffle_ctx(lm_host="30.150.24.146", lm_port=48790, app_id="my-rust-app-001"):
        df = daft.from_pydict(
            {
                "key": ["a", "b", "a", "b", "c", "a"],
                "value": [1, 2, 3, 4, 5, 6],
            }
        )
        # Explicit repartition to force the shuffle path through Celeborn.
        result = df.repartition(3, "key").groupby("key").agg(daft.col("value").sum()).collect()
        # 3 distinct keys; sums: a=10, b=6, c=5.
        as_dict = {row["key"]: row["value"] for row in result.iter_rows()}
        assert as_dict == {"a": 10, "b": 6, "c": 5}
