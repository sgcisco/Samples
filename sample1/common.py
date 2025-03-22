import hashlib
import re
import uuid
from collections import Counter
from itertools import chain
from typing import Dict, List, Optional, Set

import pandas as pd
import pyspark.sql.functions as F

from pyspark.sql import DataFrame
from pyspark.sql.functions import PandasUDFType, pandas_udf
from pyspark.sql.window import Window

AVRO_FORMAT = "com.databricks.spark.avro"


# NOTE some eWLC entities update once every hour + epsilon seconds.
DEFAULT_ASOF_JOIN_LOOKBACK_SEC = 3600 + 20


def _add_missing_cols(df, existing_cols, cols):
    new_columns = []
    for col in cols:
        if col not in existing_cols:
            new_columns.append(col)
    return df.select(*existing_cols, *(F.lit(None).alias(new_column) for new_column in new_columns))


# TODO: this function is included for reference and performance tests.
def std_asof_join(
    df_left: DataFrame,
    df_right: DataFrame,
    keys: Set[str],
    left_maps: Dict[str, str],
    right_maps: Dict[str, str],
    lookback_period_ms: int = DEFAULT_ASOF_JOIN_LOOKBACK_SEC * 1000,
    ts_col: str = "ts",
):

    assert ts_col in df_left.columns, "Left dataframe does not have timestamp"
    assert ts_col in df_right.columns, "Right dataframe does not have timestamp"
    assert keys <= set(left_maps.values()), "Keys {} are not in the left columns".format(
        keys - set(left_maps.values())
    )
    assert keys <= set(right_maps.values()), "Keys {} are not in the right columns".format(
        keys - set(right_maps.values())
    )

    lookback_sec = max(int(lookback_period_ms / 1000), 1)
    # Bucket has to be in seconds
    granularity = f"{lookback_sec} seconds"
    # have to convert timestamp to be in seconds to create buckets with window function
    expr_curr = F.window((F.col(ts_col) / 1000).cast("timestamp"), granularity).getField("start")

    # Hereafter we use the bucketing approach described here:
    # http://zachmoshe.com/2016/09/26/efficient-range-joins-with-spark.html
    # This prevents Spark from using cartesian products by bucketizing records and
    # joining on the exact timestamps associated to these buckets.
    df_left = df_left.select(
        *[F.col(col).alias(name) for col, name in left_maps.items()],
        ts_col,
        expr_curr.alias("bucketTs"),
    )
    df_right = df_right.select(
        *[F.col(col).alias(name) for col, name in right_maps.items()],
        ts_col,
        expr_curr.alias("bucketTs"),
    )

    # We can (and should) still impose the records of the right-hand dataframe
    # to precede those of the left-hand dataframe.
    conds = [df_left[col] == df_right[col] for col in keys] + [
        df_left[ts_col] >= df_right[ts_col],
        (df_left[ts_col] - df_right[ts_col]) <= lookback_period_ms,  # NOTE use millis
    ]

    left_cols = set([col for col in left_maps.values() if col not in keys])
    right_cols = set([col for col in right_maps.values() if col not in keys])
    # select non-null entries at identical timestamps
    df_right = df_right.groupBy(ts_col, "bucketTs", *keys).agg(
        *[F.last(col, True).alias(col) for col in right_cols]
    )

    assert len(left_cols & right_cols) == 0, "{} are both in left and right mappings".format(
        left_cols & right_cols
    )

    sel_expr = (
        [F.expr("left.*")]
        + [F.col("right.{}".format(col)).alias(col) for col in right_cols]
        + [F.col(f"right.{ts_col}").alias("ts_")]
    )

    # We now join every entry to those in its matching bucket and the one before that.
    # Note that we still must perform a filtering step after the join, but this is
    # much more efficient since data are joined and properly partitioned by bucket.
    # NOTE: converting bucketTs to long will be in whole seconds, not millis.
    df_right = df_right.unionByName(
        df_right.withColumn(
            "bucketTs",
            (F.col("bucketTs").cast("long") + lookback_sec).cast("timestamp"),
        )
    )

    df = (
        df_left.alias("left")
        .join(
            df_right.alias("right"),
            conds + [df_left["bucketTs"] == df_right["bucketTs"]],
            how="left_outer",
        )
        .select(sel_expr)
    )

    # Enforce data from right data frame is earlier than left
    for col in right_cols:
        df = df.withColumn(
            col,
            F.when(
                (
                    (F.col(ts_col) - F.col("ts_") <= lookback_period_ms)
                    & (F.col(ts_col) >= F.col("ts_"))
                ),
                F.col(col),
            ).otherwise(F.lit(None)),
        )

    # We want null values not to be taken into account upon selecting the
    # latest value.
    win = Window.partitionBy(ts_col, *keys).orderBy("ts_").rowsBetween(Window.unboundedPreceding, 0)
    for col in right_cols:
        df = df.withColumn(col, F.last(col, True).over(win))

    # Finally, we keep only the record with the most recent update from the
    # left-hand dataframe.
    win_rank = Window.partitionBy(ts_col, *keys).orderBy(F.desc("ts_"))
    return (
        df.withColumn("rn", F.row_number().over(win_rank))
        .where(F.col("rn") == 1)
        .select(ts_col, *(keys | left_cols | right_cols))
    )


def kinda_asof_join(
    df_left: DataFrame,
    df_right: DataFrame,
    keys: List[str],
    left_maps: Dict[str, str] = None,
    right_maps: Dict[str, str] = None,
    lookback_period_ms: Optional[int] = DEFAULT_ASOF_JOIN_LOOKBACK_SEC * 1000,
    ts_col: str = "ts",
    target_cols: Optional[List[str]] = [],
    win_direction: str = WIN_DIR_FORWARD,
):
    """Join the `target_cols` cols of dataframe `df_right` to `df_left` using the rows as close
    in time to the keys in `df_left` as possible. For example, `df_left` might be client related
    data, and `df_right` maps a client Mac to an AP and radio slot. This function will assign
    the AP and radio slot closest in time to each client Mac. This function uses an optimal
    bucketing and Union approach. The bucket size is determined by `lookback_period_ms`, and this
    limits how far back (or forwards_in time the join can occur).
    Ideally, the bucket size is small relative to the total time of the dataframes.
    IMPORTANT: the timestamp has to be in milliseconds.

    Parameters:
    df_left (DataFrame): main dataframe to which we want to join data
    df_right (DataFrame): Additional dataframe, we want to join columns
    that are closest in time to `df_left`
    keys (List[str]): primary keys of both left and right dataframes
    left_maps (Dict[str, str]):   Optional renaming of column names
    right_maps (Dict[str, str] ):  Optional renaming of column names
    lookback_period_ms (int): The maximum duration of the lookback. For each key in `df_left`,
    the nearest key in df_right is joined unless it is longer than `lookback_period_ms`
    away (past or future)
    ts_col (str): name of the timestamp column in both dataframes. The timestamp HAS
    to be in milliseconds
    target_cols (List[str]):  Optionally, select a subset of columns from `df_right` to join.
    if no values are specified, all columns are joined
    win_direction (str): direction of the asof join. `WIN_DIR_FORWARD` or `WIN_DIR_BACKWARD`.
    `WIN_DIR_FORWARD` means that the data from `df_right` is joined to `df_left` while maintianing
    the timestamps of `df_right` occur at or before `df_left`. WIN_DIR_BACKWARD` is the reverse

    Returns:
    DataFrame: Returns a DataFrame of df_right joined to df_left by time
    """

    assert ts_col in df_left.columns, "Left dataframe does not have timestamp"
    assert ts_col in df_right.columns, "Right dataframe does not have timestamp"
    if left_maps:
        assert keys <= set(left_maps.values()), "Keys {} are not in the left columns".format(
            keys - set(left_maps.values())
        )

        df_left = df_left.select(
            *[F.col(col).alias(name) for col, name in left_maps.items()],
            ts_col,
        )

    if right_maps:
        assert keys <= set(right_maps.values()), "Keys {} are not in the right columns".format(
            keys - set(right_maps.values())
        )

        df_right = df_right.select(
            *[F.col(col).alias(name) for col, name in right_maps.items()],
            ts_col,
        )

    win = Window.partitionBy(*keys)
    if win_direction == WIN_DIR_BACKWARD:
        win = win.orderBy(F.desc(ts_col))
    else:
        # We need F.isnull() in this case, otherwise nulls come first and will
        # always be selected by
        #
        #    .withColumn("rowNumber", F.row_number().over(dedup_win))
        #    .where(F.col("rowNumber") == 1)
        #
        win = win.orderBy(F.isnull(ts_col), F.asc(ts_col))

    if not target_cols:
        target_cols = list((set(df_right.columns) - set(keys)) - set([ts_col]))

    # Any columns that only contain null should be ignored to prevent OOM
    data_counts = df_right.select(
        [F.count(F.when(F.col(c).isNotNull(), c)).alias(c) for c in target_cols]
    ).collect()[0]

    null_columns = set()
    for target_col in target_cols:
        if target_col == ts_col:
            continue
        elif data_counts[target_col] == 0:
            null_columns.add(target_col)
    target_cols = list(set(target_cols) - null_columns)

    df_left_tmp = df_left.withColumn(f"left_{ts_col}", F.col(ts_col))
    df_right_tmp = df_right.withColumn(f"right_{ts_col}", F.col(ts_col))

    cols_df_left = set(df_left_tmp.columns)
    cols_df_right = set(df_right_tmp.columns)
    union_cols = cols_df_left.union(cols_df_right)

    tmp_time_cols = set([f"left_{ts_col}", f"right_{ts_col}"])

    df_left_tmp = _add_missing_cols(df_left_tmp, cols_df_left, union_cols)
    df_right_tmp = _add_missing_cols(df_right_tmp, cols_df_right, union_cols)

    df_union = df_left_tmp.unionByName(df_right_tmp)
    df_union = df_union.repartition(200)
    df_union.persist()
    df_union.count()

    win_diff_ts = win.orderBy(F.asc_nulls_first(ts_col))

    df_union = df_union.select(
        "*",
        *(
            F.when(
                F.col(target_col).isNotNull(),
                F.col(f"right_{ts_col}"),
            )
            .otherwise(F.lit(None))
            .alias(f"ts_{target_col}")
            for target_col in target_cols
        ),
    )

    df_union = df_union.select(
        "*",
        *(
            (F.col(f"left_{ts_col}") - F.last(f"ts_{target_col}", True).over(win_diff_ts)).alias(
                f"diff_ts_{target_col}"
            )
            for target_col in target_cols
        ),
        *(
            F.last(target_col, True).over(win).alias(f"new_{target_col}")
            for target_col in target_cols
        ),
    )

    df_union = df_union.drop(*[target_col for target_col in target_cols])

    df_union = df_union.select(
        "*",
        *(
            F.when(
                (
                    F.col(f"diff_ts_{target_col}").isNull()
                    | (
                        (F.col(f"diff_ts_{target_col}") <= lookback_period_ms)
                        & (F.col(f"diff_ts_{target_col}") >= 0)
                    )
                ),
                F.col(f"new_{target_col}"),
            )
            .otherwise(F.lit(None))
            .alias(f"{target_col}")
            for target_col in target_cols
        ),
    )

    df_union = (
        df_union.filter(F.col(f"left_{ts_col}").isNotNull())
        .drop(f"{ts_col}")
        .withColumnRenamed(f"left_{ts_col}", f"{ts_col}")
        .select(list(union_cols.difference(tmp_time_cols)))
    )

    return df_union

