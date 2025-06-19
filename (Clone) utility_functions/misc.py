# Databricks notebook source
import pyspark
from pyspark.sql import functions as f
from functools import reduce
from typing import Dict, List, Union

from pyspark import sql
from pyspark.sql import SparkSession
# from pyspark.sql.types import StringType
import pyspark.sql.types as sqltypes
from pyspark.sql.types import (
    StructType
    , StructField
    , StringType
    , IntegerType
    , DoubleType
    , BooleanType
    , TimestampType
    , DateType
    , LongType
    # , sqltypes
)

#from fuzzywuzzy import fuzz


# COMMAND ----------


def reduce_union(
    list_of_dataframes: List[pyspark.sql.DataFrame]
    , union_by_name: bool = False
) -> pyspark.sql.DataFrame:
    """Given a list of dataframes, union all of them into a single dataframe.
    Args:
        list_of_dataframes: A list of dataframes with equivalent schemas.
        union_by_name: Whether to union dataframe by column names.
    Returns:
        A spark dataframe.
    """
    if union_by_name:
        return reduce(lambda x, y: x.unionByName(y), list_of_dataframes)

    return reduce(lambda x, y: x.union(y), list_of_dataframes)


def create_spark_df(
    ls_values: list
    , vt_col_nm: str
    , col_type: sqltypes
) -> sql.DataFrame:

    spark = SparkSession.builder.getOrCreate()

    df_out = (
        spark
        .createDataFrame(
            ls_values
            , col_type
        )
        .withColumnRenamed("value", vt_col_nm)
    )

    return df_out


def reformat_pivot_col_names(
    ls_col_nm: list
    # , vt_pattern_sep: str = "_\\|"
    , vt_pattern_org: str = "(.*)|(.*)|(.*)"
    , vt_pattern_present: str = "$2_$1_$3"
    , vt_pattern_suppress: str = None
) -> dict:

    spark = SparkSession.builder.getOrCreate()

    df_func_workspace = (
        spark
        .createDataFrame(
            ls_col_nm
            , StringType()
        )
        .withColumnRenamed("value", "col_nm_org")
        .withColumn(
            "col_nm_proc"
            , f.regexp_replace(f.col("col_nm_org"), vt_pattern_org, vt_pattern_present)
        )
    )

    if vt_pattern_suppress is not None:
        df_func_workspace = (
            df_func_workspace
            .withColumn(
                "col_nm_proc"
                , f.trim(
                    f.regexp_replace(f.col("col_nm_proc"), vt_pattern_suppress, "")
                )
            )
        )

    ls_col_nm_proc = (
        df_func_workspace
        .select("col_nm_proc")
        .rdd.map(lambda x: x[0])
        .collect()
    )

    dict_out = {
        "col_nm_org": ls_col_nm
        , "col_nm_proc": ls_col_nm_proc
    }

    return dict_out


def add_missing_cols(
    df: sql.DataFrame
    , ls_fields: list
    , vt_assign_value=None
) -> sql.DataFrame:

    df_out = df
    for column in [column for column in ls_fields if column not in df.columns]:
        df_out = df_out.withColumn(column, f.lit(vt_assign_value))

    return df_out


def add_missing_cols_v2(
    df: sql.DataFrame
    , ls_cols: list
    , vt_value=None
    , vt_datatype='double'
) -> sql.DataFrame:
    spark = sql.SparkSession.builder.getOrCreate()
    ls_missing_cols = []

    for column in [column for column in ls_cols if column not in df.columns]:
        ls_missing_cols.append(column)

    df_out = df
    if len(ls_missing_cols) > 0:
        missing_cols = [sql.Row(**{col: vt_value for col in ls_missing_cols})]
        schema = ",".join([f"{col} {vt_datatype}" for col in ls_missing_cols])
        df_missing_cols = spark.createDataFrame(missing_cols, schema)
        df_out = (
            df_out
            .crossJoin(df_missing_cols)
        )

    return df_out


def regex_map_values_from_dict(
    column: str,
    mapping: Dict[str, Union[str, List[str]]],
    other: str = None,
    coalesce: bool = False,
    ignore_case: str = None,
) -> sql.Column:
    """
    Maps values using a dictionary with regular expression rules.

    The key represents the final value and its value represent a list of rules defined
    in regular expressions.

    Args:
        column: The string name of the column.
        mapping: Dictionary of mapping values.
        other: default values when the column doesn't match any of the expressions.
        coalesce: Coalesce to original value if ``True``.
        ignore_case: this will ignore the case for column values

    Returns:
        A spark column object.
    """
    coalesce_replacements = list()
    for replacement, pattern in mapping.items():
        # Converting list of regex patterns into a string separated by an or condition.
        if isinstance(pattern, list):
            pattern = "|".join(pattern)
        if ignore_case:
            replacement = f.when(f.lower(f.col(column)).rlike(pattern), replacement)
        else:
            replacement = f.when(f.col(column).rlike(pattern), replacement)
        coalesce_replacements.append(replacement)

    if coalesce:
        other_replacement = f.col(column)
    else:
        other_replacement = f.lit(other)

    coalesce_replacements.append(other_replacement)

    return f.coalesce(*coalesce_replacements)


def match_map_values_from_dict(
    column: str
    , mapping: Dict[str, Union[str, List[str]]]
    , other: str = None
    , coalesce: bool = False
    , ignore_case: str = None
) -> sql.Column:
    """
    Maps values using a dictionary with list.

    The key represents the final value and its value in a list.

    Args:
        column: The string name of the column.
        mapping: Dictionary of mapping values.
        other: default values when the column doesn't match any of the expressions.
        coalesce: Coalesce to original value if ``True``.
        ignore_case: this will ignore the case for column values

    Returns:
        A spark column object.
    """
    coalesce_replacements = list()

    for replacement, values in mapping.items():

        if ignore_case:
            replacement = f.when(f.lower(f.col(column)).isin(values), replacement)

        else:
            replacement = f.when(f.col(column).isin(values), replacement)

        coalesce_replacements.append(replacement)

    if coalesce:
        other_replacement = f.col(column)
    else:
        other_replacement = f.lit(other)

    coalesce_replacements.append(other_replacement)

    return f.coalesce(*coalesce_replacements)


# Apply Fuzzy Function
# def fuzzy_match_string(s1, s2):
#     return fuzz.token_sort_ratio(s1, s2)


# udf_fuzzy_match_string = f.udf(fuzzy_match_string, StringType())

def lower_col_names(
    df: sql.DataFrame
) -> sql.DataFrame:

    df_out = (
        df
        .toDF(*[c.lower() for c in df.columns])
    )

    return df_out


def drop_system_fields(
    df: sql.DataFrame
) -> sql.DataFrame:
    df_out = (
        df
        .drop("reporting_cycle_type", "reporting_date", "data_update_date", "data_update_dttm")
    )

    return df_out

