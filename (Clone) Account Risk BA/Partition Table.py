# Databricks notebook source
import pyspark
from pyspark import sql 
from pyspark.sql import Window
from pyspark.sql import functions as f
from pyspark.sql.functions import from_unixtime,from_utc_timestamp
from pyspark.sql.functions import date_format

# COMMAND ----------

df_item_t_load = spark.read.format("delta").load('dbfs:/mnt/prod_brm/raw/cdc/RAW_PINPAP_ITEM_T')
df_bill_t_load = spark.read.format("delta").load('dbfs:/mnt/prod_brm/raw/cdc/RAW_PINPAP_BILL_T')

# COMMAND ----------

display(df_item_t_load.select('poid_type').distinct())

# COMMAND ----------

#  dbutils.fs.mkdirs("/mnt/ml-lab/dev_users/dev_sc")
# dbutils.fs.rm("dbfs:/mnt/ml-lab/dev_users/dev_sc",True)
# dbutils.fs.mkdirs("/mnt/ml-lab/dev_users/dev_sc/RAW_PINPAP_BILL_T")

# COMMAND ----------

# MAGIC %run "./Function"

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

df_item_t_filtered =( 
        df_item_t_load
        .filter(f.col("_IS_LATEST") ==1)
        .filter(f.col("_IS_DELETED") ==0)
        .filter(f.col("CREATED_T")>= '1609412400' ) # 2021-01-01
        .withColumn('CREATED_TIME', f.from_utc_timestamp(from_unixtime("CREATED_T"),"Pacific/Auckland"))
        .withColumn("CREATED_MONTH", date_format("CREATED_TIME","yyyy-MM") )
        .filter(f.col("poid_type").isin(
             "/item/payment",
             "/item/payment/reversal",
             "/item/writeoff",
             "/item/writeoff_reversal",
             "/item/adjustment")
        )
    )

#display(df_item_t_filtered.count()) # 38,794,942


# COMMAND ----------

display(df_item_t_filtered.select('poid_type').distinct())

# COMMAND ----------

dbutils.fs.ls('/mnt/ml-lab/dev_users/dev_sc/')

# COMMAND ----------

# dbfs:/mnt/ml-lab/dev_users/dev_sc/RAW_PINPAP_ITEM_T

dir_export = 'dbfs:/mnt/ml-lab/dev_users/dev_sc/RAW_PINPAP_ITEM_T'
export_data(
    df = df_item_t_filtered
    , export_path = dir_export
    , export_format = "delta"
    , export_mode = "overwrite"
    , flag_overwrite_schema = True
    , flag_dynamic_partition = True
    , ls_dynamic_partition=["CREATED_MONTH"]
)


# COMMAND ----------

# display(df_item_t_filtered.groupBy("POID_TYPE").agg(f.count("poid_id0")))

# COMMAND ----------

df_brm_t_filtered =( 
        df_bill_t_load
        .filter(f.col("_IS_LATEST") ==1)
        .filter(f.col("_IS_DELETED") ==0)
        .withColumn('due_time', f.from_utc_timestamp(from_unixtime("due_t"),"Pacific/Auckland"))
        .withColumn("due_month", date_format("due_time","yyyy-MM") )
        .filter(f.col("due_month")>='2021-01')
        )
        # 134581935
    

# COMMAND ----------

dir_export = '/mnt/ml-lab/dev_users/dev_sc/RAW_PINPAP_BILL_T'
export_data(
    df = df_brm_t_filtered
    , export_path = dir_export
    , export_format = "delta"
    , export_mode = "overwrite"
    , flag_overwrite_schema = True
    , flag_dynamic_partition = True
    , ls_dynamic_partition=["due_month"]
)

# COMMAND ----------

display(dbutils.fs.ls('/mnt/ml-lab/dev_users/dev_sc/RAW_PINPAP_BILL_T'))

# COMMAND ----------

dbutils.fs.ls('mnt/feature-store-dev/dev_users/dev_sc/d100_raw')

# COMMAND ----------

df_test = spark.read.format('delta').load('dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d100_raw/WRITEOFF_BASE/')

# COMMAND ----------

display(df_test
        .withColumn('create_month', date_format(f.col('rec_created_dttm'), 'yyyy-MM'))
        )
