# Databricks notebook source
from pyspark.sql.functions import from_utc_timestamp, col 
from pyspark import sql
import pandas as pd
import itertools as itl
import re
from pyspark.sql import functions as f
from pyspark.sql import Window, SparkSession
from pyspark.sql.types import IntegerType

# COMMAND ----------

def to_nzt(df, timestamp_cols):
    nzt_timezone = "Pacific/Auckland"
    for timestamp_col in timestamp_cols:
        df = df.withColumn(timestamp_col, 
                           from_utc_timestamp(from_unixtime(col(timestamp_col)), nzt_timezone))
    return df


# COMMAND ----------

def lower_col_names(
    df: sql.DataFrame
) -> sql.DataFrame:

    df_out = (
        df
        .toDF(*[c.lower() for c in df.columns])
    )

    return df_out

# COMMAND ----------

def check_similarity(dataframe, dict_pairs, value_adj=1, threshold=0.9, excl_zero = True):
    """
    Check if two numeric columns in a DataFrame are within a certain threshold of similarity.
 
    :param dataframe: Input DataFrame
    :param dict_pairs: pairs of columns to compare, exclude columns containing special characters. e.g. |, &
    :param threshold: Similarity threshold (default is 0.9)
    :return: DataFrame with an additional column indicating similarity
    """
 
    ls_result = []
 
    for column1, column2 in dict_pairs.items():
        # Convert the column dictionary into a DataFrame
        if re.search(r'cnt', column1):
            value = 1
        else:
            value = value_adj
       
        if excl_zero:
            str_filter_non_zero = f'({column1} > 0 or {column2} > 0)'
       
        result = (
            dataframe
            .withColumn("fs_col", f.lit(column1))
            .withColumn("benchmark_col", f.lit(column2))
            .withColumn("fs_col_val", f.col(column1))
            .withColumn("benchmark_col_val", value*f.col(column2))
            # Calculate the absolute difference between the two columns
            .withColumn("abs_diff", f.abs(f.col(column1) - value*f.col(column2)))
            # Calculate the maximum value between the two columns
            .withColumn("max_value", f.greatest(f.col(column1), value*f.col(column2)))
            .withColumn(
                "similarity_ratio",
                f.when(f.col("abs_diff") == 0, 1.0)
                .otherwise(1 - (f.col("abs_diff") / f.col("max_value"))))
            .withColumn(
                "similar_flag",
                f.when(f.col("similarity_ratio") >= threshold,1).otherwise(0)
                )
            .select(
                "reporting_date",
                "ACCOUNT REF NO",
                "fs_col",
                "benchmark_col",
                "fs_col_val",
                "benchmark_col_val",
                "similarity_ratio",
                "similar_flag"
                )
        )
        ls_result.append(result)
 
    combined_df = None
    for df in ls_result:
        if combined_df is None:
            combined_df = df
        else:
            combined_df = combined_df.union(df)
 
    return combined_df

# COMMAND ----------

def export_data(
    df: sql.DataFrame
    , export_path: str
    , export_format: str
    , export_mode: str
    , flag_overwrite_schema: bool
    , flag_dynamic_partition: bool
    , ls_dynamic_partition: list = None
):
    export_obj = (
        df
        .write
        .format(export_format)
        .mode(export_mode)
    )

    if flag_overwrite_schema:
        export_obj = (
            export_obj
            .option("overwriteSchema", "true")
        )

    if flag_dynamic_partition:
        export_obj = (
            export_obj
            .partitionBy(ls_dynamic_partition)
            .option("partitionOverwriteMode", "dynamic")
        )

    (
        export_obj
        .save(export_path)
    )

# COMMAND ----------


def repeat_seq(
    x: list
    , each: int
) -> list:

    ls_output = (
        list(
            itl.chain.from_iterable(
                itl.repeat(i, each)
                for i in x
            )
        )
    )

    return ls_output


def get_lookback_cycle_meta(
    df_calendar: sql.DataFrame
    , vt_param_date: str
    , vt_param_lookback_cycles: int
    , vt_param_lookback_cycle_unit_type: str
    , vt_param_lookback_units_per_cycle: int = None
) -> sql.DataFrame:

    """
    inputs:
        df_calendar: feature store global ssc calendar
        vt_param_date: date used to calculate the lookback cycles
        vt_param_lookback_cycles: # cycles to lookback
        vt_param_lookback_cycle_unit_types: base units of cycle in days/weeks/cycles/months
        vt_param_lookback_units_per_cycle: parameter to overwrite the default units/cycle
            months: 1 month/cycle
            cycles: 1 cycle/cycle
            weeks:  4 weeks/cycle
            days:  28 days/cycle

    output: cycle meta with partition index for p0/p1/.../px

    """

    spark = SparkSession.builder.getOrCreate()

    # input preparation
    if vt_param_lookback_cycle_unit_type == "months":
        if vt_param_lookback_units_per_cycle is None:
            vt_param_multiplier = 1
        else:
            vt_param_multiplier = vt_param_lookback_units_per_cycle

        df_input = (
            df_calendar
            .filter(f.col("cycle_type") == "calendar cycle")
            .withColumnRenamed("base_month", "cycle_date")
        )

    if vt_param_lookback_cycle_unit_type == "cycles":
        if vt_param_lookback_units_per_cycle is None:
            vt_param_multiplier = 1
        else:
            vt_param_multiplier = vt_param_lookback_units_per_cycle

        df_input = (
            df_calendar
            .filter(f.col("cycle_type") == "rolling cycle")
            .filter(f.col("proc_freq_cycle_flag") == 'Y')
            .withColumnRenamed("base_cycle", "cycle_date")
        )

    if vt_param_lookback_cycle_unit_type == "weeks":
        if vt_param_lookback_units_per_cycle is None:
            vt_param_multiplier = 4
        else:
            vt_param_multiplier = vt_param_lookback_units_per_cycle

        df_input = (
            df_calendar
            .filter(f.col("cycle_type") == "rolling cycle")
            .filter(f.col("proc_freq_weekly_flag") == 'Y')
            .withColumnRenamed("base_week", "cycle_date")
        )

    if vt_param_lookback_cycle_unit_type == "days":
        if vt_param_lookback_units_per_cycle is None:
            vt_param_multiplier = 28
        else:
            vt_param_multiplier = vt_param_lookback_units_per_cycle

        df_input = (
            df_calendar
            .filter(f.col("cycle_type") == "rolling cycle")
            .filter(f.col("proc_freq_daily_flag") == 'Y')
            .withColumnRenamed("base_date", "cycle_date")
        )

    # cycle creation
    vt_param_lookback_cycle_units = vt_param_lookback_cycles * vt_param_multiplier

    # cycle meta
    df_tmp_cycle_meta = (
        df_input
        .filter(f.col("cycle_date") <= vt_param_date)
        .withColumn(
            "index"
            , f.row_number().over(
                Window
                .partitionBy(f.lit(1))
                .orderBy(f.desc("cycle_date"))
            )
        )
        .filter(f.col("index") <= vt_param_lookback_cycle_units)
    )

    # cycle index
    df_tmp_cycle_index = (
        spark
        .createDataFrame(
            repeat_seq(
                list(range(vt_param_lookback_cycles + 1)[1:(vt_param_lookback_cycles + 1)])
                , vt_param_multiplier
            )
            , IntegerType()
            # , "week_seq"
        )
        .withColumnRenamed("value", 'cycle_index')
        .withColumn(
            "index"
            , f.row_number()
            .over(
                Window
                .orderBy(f.monotonically_increasing_id())
            )
        )
    )

    df_output = (
        df_tmp_cycle_meta
        .orderBy(f.desc("cycle_date"))
        .join(df_tmp_cycle_index, ["index"], "left")
        .withColumnRenamed("cycle_date", "partition_date")
        .withColumn(
            "cycle_date"
            , f.max("partition_date").over(
                Window
                .partitionBy("cycle_index")
            )
        )
        .select(
            "cycle_date"
            , "cycle_type"
            , f.col("base_snapshot_date_start").alias('reporting_start_date')
            , f.col("base_snapshot_date_end").alias('reporting_end_date')
            , "cycle_index"
            , "partition_date"
            , f.col("index").alias("partition_index")
        )
    )

    return df_output
