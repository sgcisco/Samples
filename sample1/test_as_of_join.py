import io
import math
import sys
from contextlib import contextmanager

import pytest
from kairosfunc.entities.common import kinda_asof_join, std_asof_join
from pyspark.sql.types import IntegerType, StructField, StructType

from kairosfunc.base_entities.enum_mapping import enum_mapping


@pytest.fixture
def args(sql_context):
    df1 = sql_context.createDataFrame(
        [
            ("A", "X", 10, 1),
            ("A", "X", 15, 2),
            ("A", "X", 20, 3),
            ("B", "X", 10, 1),
            ("B", "X", 15, 2),
            ("B", "X", 20, 3),
            ("C", "X", 10, 4),
        ],
        ["key", "id", "ts", "col1"],
    )
    df2 = sql_context.createDataFrame(
        [
            ("A", "X", 8, 1, 2),
            ("A", "X", 9, None, 3),
            ("A", "X", 14, 3, None),
            ("B", "X", 1, 1, 2),
            ("B", "Y", 15, 2, None),
            ("B", "X", 20, 3, 4),
        ],
        ["key", "id", "ts", "col2", "col3"],
    )
    df3 = sql_context.createDataFrame(
        [
            ("A", "X", 9, 1, 3),
            ("A", "X", 9, None, 3),
            ("A", "X", 14, 3, None),
            ("B", "X", 1, 1, 2),
            ("B", "Y", 15, 2, None),
            ("B", "X", 20, 3, 4),
        ],
        ["key", "id", "ts", "col2", "col3"],
    )
    df4 = sql_context.createDataFrame(
        [
            ("A", "X", 11, 2, 3),
            ("A", "X", 12, 2, 3),
            ("A", "X", 14, 3, None),
            ("B", "X", 1, 1, 2),
            ("B", "Y", 15, 2, None),
            ("B", "X", 20, 3, 4),
        ],
        ["key", "id", "ts", "col2", "col3"],
    )

    keys = {"key"}
    left_maps = {"key": "key", "col1": "col1"}
    right_maps = {"key": "key", "col2": "col2", "col3": "col3"}

    return df1, df2, df3, df4, keys, left_maps, right_maps


@contextmanager
def capture_stdout():
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    try:
        yield buffer
    finally:
        buffer.close()
        sys.stdout = old_stdout


def test_base(args):
    df1, df2, df3, df4, keys, left_maps, right_maps = args

    def test_asof(join_func):
        df = join_func(
            df1,
            df2,
            keys,
            left_maps=left_maps,
            right_maps=right_maps,
            lookback_period_ms=10,
        )

        assert set(df.columns) == {"key", "ts", "col1", "col2", "col3"}
        assert df.count() == df1.count()

        df = df.toPandas().set_index(["key", "ts"])
        assert df.loc["A", 10]["col1"] == 1
        assert df.loc["A", 10]["col2"] == 1
        assert df.loc["A", 10]["col3"] == 3
        assert df.loc["A", 15]["col3"] == 3
        assert math.isnan(df.loc["A", 20]["col3"])

        assert df.loc["B", 10]["col1"] == 1
        assert df.loc["B", 10]["col2"] == 1
        assert df.loc["B", 10]["col3"] == 2
        assert df.loc["B", 15]["col2"] == 2
        assert math.isnan(df.loc["B", 15]["col3"])
        assert df.loc["B", 20]["col3"] == 4

        df = join_func(
            df1,
            df3,
            keys,
            left_maps=left_maps,
            right_maps=right_maps,
            lookback_period_ms=10,
        )
        df = df.toPandas().set_index(["key", "ts"])
        assert df.loc["A", 10]["col1"] == 1
        assert df.loc["A", 10]["col2"] == 1
        assert df.loc["A", 10]["col3"] == 3

        df = join_func(
            df1,
            df4,
            keys,
            left_maps=left_maps,
            right_maps=right_maps,
            lookback_period_ms=10,
        )
        df = df.toPandas().set_index(["key", "ts"])
        assert df.loc["A", 10]["col1"] == 1
        assert math.isnan(df.loc["A", 10]["col2"])
        assert math.isnan(df.loc["A", 10]["col3"])

    test_asof(kinda_asof_join)
    test_asof(std_asof_join)


def test_multi_keys(args):
    df1, df2, _, _, keys, left_maps, right_maps = args

    keys.add("wlcId")
    left_maps["id"] = "wlcId"
    right_maps["id"] = "wlcId"

    df = kinda_asof_join(
        df1,
        df2,
        keys,
        left_maps=left_maps,
        right_maps=right_maps,
        lookback_period_ms=10,
    )

    assert set(df.columns) == {"key", "wlcId", "ts", "col1", "col2", "col3"}
    assert df.count() == df1.count()

    df = df.toPandas().set_index(["key", "ts"])
    assert math.isnan(df.loc["B", 15]["col2"])


def test_col_ts(args):
    df1, df2, _, _, keys, left_maps, right_maps = args

    df1 = df1.withColumnRenamed("ts", "timestamp")
    df2 = df2.withColumnRenamed("ts", "timestamp")

    df = kinda_asof_join(
        df1,
        df2,
        keys,
        left_maps=left_maps,
        right_maps=right_maps,
        lookback_period_ms=10 * 1000,
        ts_col="timestamp",
    )

    assert set(df.columns) == {"key", "timestamp", "col1", "col2", "col3"}
    assert df.count() == df1.count()


def test_exec_plan(args):
    df1, df2, _, _, keys, left_maps, right_maps = args
    df = kinda_asof_join(
        df1,
        df2,
        keys,
        left_maps=left_maps,
        right_maps=right_maps,
        lookback_period_ms=10,
    )

    with capture_stdout() as outbuf:
        df.explain()
        plan = outbuf.getvalue()

    assert "CartesianProduct" not in plan, "No cartesian product must be used"
