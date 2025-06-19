# Databricks notebook source
from pyspark import sql
from pyspark.sql import SparkSession

import pandas as pd

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

def add_missing_cols(
    df: sql.DataFrame
    , ls_fields: list
    , vt_assign_value=None
) -> sql.DataFrame:
    
    df_out = df
    for column in [column for column in ls_fields if column not in df.columns]:
        df_out = df_out.withColumn(column, f.lit(vt_assign_value))
        
    return df_out

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

def pull_col(
    df: sql.DataFrame
    , vt_col: str
) -> list:
    '''pull column of a spark df into a list
    Args:
        df: spark dataframe
        vt_col: column name
    '''

    ls_out = (
        df
        .select(vt_col)
        .rdd.map(lambda x: x[0])
        .collect()
    )

    return ls_out
