# Databricks notebook source
# import pyspark
# from pyspark.sql import functions as f
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

    ls_out = (
        df
        .select(vt_col)
        .rdd.map(lambda x: x[0])
        .collect()
    )

    return ls_out


def get_schema(
    df: sql.DataFrame
) -> sql.DataFrame:
    '''extract the schema of a spark df into a spark df
    Args:
        df: spark dataframe
    '''
    spark = SparkSession.builder.getOrCreate()

    ls_df_field = []
    ls_df_field_type = []

    for i in df.dtypes:
        ls_df_field.append(i[0])
        ls_df_field_type.append(i[1])

    df_output = pd.DataFrame(
        data={
            "field": ls_df_field
            , "type": ls_df_field_type
        }
    )

    df_output = spark.createDataFrame(df_output)

    return df_output

