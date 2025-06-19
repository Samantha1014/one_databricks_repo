# Databricks notebook source
from pyspark import sql
from pyspark.sql import SparkSession

import pandas as pd

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
    return df.select(vt_col).rdd.flatMap(lambda x: x).collect()

# COMMAND ----------

def lower_col_names(
    df: sql.DataFrame
) -> sql.DataFrame:
    for col in df.columns:
        df = df.withColumnRenamed(col, col.lower())
    return df

# COMMAND ----------

def get_schema(
    df: sql.DataFrame
) -> sql.DataFrame:
    '''extract the schema of a spark df into a spark df
    Args:
        df: spark dataframe
    '''
    spark = SparkSession.builder.getOrCreate()
    schema = df.schema
    df_schema = spark.createDataFrame(
        [(field.name, str(field.dataType)) for field in schema],
        ["field", "type"]
    )
    return df_schema

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
