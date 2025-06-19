# Databricks notebook source
# MAGIC %md ## s000 environment setup

# COMMAND ----------

# DBTITLE 1,libraries
import pyspark
import os
from pyspark import sql 
from pyspark.sql import Window
from pyspark.sql import functions as f
from pyspark.sql.types import IntegerType, StructType
from pyspark.sql.functions import monotonically_increasing_id, row_number

import numpy as np

import pandas as pd
import datetime as dt
from jinja2 import Template

from datetime import date
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta

# ------------- Use snowflake utility
sfUtils = sc._jvm.net.snowflake.spark.snowflake.Utils

# COMMAND ----------

#dbutils.widgets.text(name="v00_param_date", defaultValue="")
#dbutils.widgets.text(name="v00_param_lookback_months", defaultValue="")
#dbutils.widgets.text(name="v01_master_dir", defaultValue="")
#dbutils.widgets.text(name="v02_export_file", defaultValue="")
#dbutils.widgets.text(name="v03_sf_db", defaultValue="")
#dbutils.widgets.text(name="v04_sf_db_schema", defaultValue="")

# COMMAND ----------

# DBTITLE 1,parameters 01
vt_param_date = dbutils.widgets.get('v00_param_date')
vt_param_lookback_months = float(dbutils.widgets.get("v00_param_lookback_months"))
dir_data_master = dbutils.widgets.get("v01_master_dir")
vt_param_export_file = dbutils.widgets.get("v02_export_file")
vt_param_sf_db = dbutils.widgets.get("v03_sf_db")
vt_param_sf_db_schema= dbutils.widgets.get("v04_sf_db_schema")

# COMMAND ----------

# DBTITLE 1,parameters 02
if vt_param_date == '':
    dbutils.notebook.exit("no date supplied")

# COMMAND ----------

# DBTITLE 1,parameters 03
vt_param_date = datetime.strptime(vt_param_date, '%Y-%m-%d').date()
vt_param_date_end = vt_param_date.replace(day = 1) - timedelta(days = 1)
vt_param_date_start = vt_param_date_end - relativedelta(months = vt_param_lookback_months - 1)
vt_param_date_start = vt_param_date_start.replace(day = 1)

vt_param_date_start = vt_param_date_start.strftime("%Y-%m-%d")
vt_param_date_end = vt_param_date_end.strftime("%Y-%m-%d")

# COMMAND ----------

# DBTITLE 1,parameters 04
vt_param_sf_db_table = f"{vt_param_sf_db}.{vt_param_sf_db_schema}.{vt_param_export_file}"

# COMMAND ----------

# DBTITLE 1,DB connectivity
# ------------ login to snowflake
password = dbutils.secrets.get(scope = "auea-kv-sbx-dxdtlprdct01", key = "sfdbrsdskey")

options = {
  "sfUrl": "vodafonenz_prod.australia-east.azure.snowflakecomputing.com/", 
  "sfUser": "SVC_LAB_DS_DATABRICKS",
  "pem_private_key": password.replace('\\n', '\n'),
  "sfDatabase": vt_param_sf_db,
  "sfSchema": vt_param_sf_db_schema,
  "sfWarehouse": "LAB_DS_WH_SCALE"
}


# COMMAND ----------

# DBTITLE 1,utility functions
# MAGIC %run "./utility_functions"

# COMMAND ----------

# DBTITLE 1,directories
dir_data_dl_edw = 'dbfs:/mnt/prod_edw/raw/cdc'
dir_data_dl_sbl = 'dbfs:/mnt/prod_siebel/raw/cdc'

# COMMAND ----------

# MAGIC %md
# MAGIC ## s100 data import

# COMMAND ----------

# DBTITLE 1,account
vt_param_table_acct = os.path.join(dir_data_dl_sbl, 'RAW_SBL_ORG_EXT')

df_db_acct = spark.sql(
    f"""
        select
            distinct
            ou_num as account_number
            , par_ou_id as customer_id
        from delta.`{vt_param_table_acct}`
        where 1 = 1
            and _is_latest = 1
            and accnt_type_cd = 'Billing'
    """
)

# COMMAND ----------

# DBTITLE 1,aod
vt_param_aod_table = os.path.join(dir_data_dl_edw, "RAW_EDW2PRD_STAGEPERM_S_INF_AOD")

df_db_aod = spark.sql(
    f"""
        select 
            account_ref_no as account_number
            , payment_status as old_payment_status

            , case 
                when aod_181plus  > 0 then 'x'
                when aod_151to180 > 0 then '6'
                when aod_121to150 > 0 then '5'
                when aod_91to120  > 0 then '4'
                when aod_61to90   > 0 then '3'
                when aod_31to60   > 0 then '2'
                when aod_01to30   > 0 then '1'
                else '0'
            end as payment_status
            
            , aod_current + aod_01to30 + aod_31to60 + aod_61to90 + aod_91to120 + aod_121to150 + aod_151to180 + aod_181plus as sum_aod
            , hist_start_dt
            , hist_end_dt
        from delta.`{vt_param_aod_table}`
        where 1 = 1
            and _is_latest = 1
            and _is_deleted = 0
    """
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## s200 data processing

# COMMAND ----------

# DBTITLE 1,lookback months
df_param_lookback_months = (
    spark
    .range(-vt_param_lookback_months, 0)
    .withColumn(
        'run_date'
        , f.lit(vt_param_date).cast("date")
    )
    .withColumn(
        "reporting_date",
        f.last_day(f.add_months('run_date', 'id'))
    )
)

display(df_param_lookback_months)

# COMMAND ----------

# DBTITLE 1,output
df_output = (
    df_db_aod
    .filter(
        (f.col("payment_status") != '0')
        & (f.col("sum_aod") >= 20)
    )
    .crossJoin(f.broadcast(df_param_lookback_months))
    .filter(
        (f.col("hist_start_dt") <= f.col("reporting_date"))
        & (f.col("hist_end_dt") >= f.col("reporting_date"))
    )
    .join(
        df_db_acct
        , ["account_number"]
        , "left"
    )
    .filter(
        (f.col("customer_id").isNotNull())
        & (f.col("account_number").isNotNull())
    )
    .withColumn(
        'account_number'
        , f.lpad(f.col("account_number").cast("string"), 9, '0')
    )
    .select(
        f.date_format('reporting_date', 'yyyy-MM-01').alias('created_month')
        , "customer_id"
        , "account_number"
        , "payment_status"
    )
    .distinct()
    .withColumn(
        "data_update_date"
        , f.current_date()
    )
    .withColumn(
        "data_update_dttm"
        , f.current_timestamp()
    )
)

display(df_output.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## s300 data checking

# COMMAND ----------

# DBTITLE 1,data size
display(df_output.count())

# COMMAND ----------

# DBTITLE 1,data size by filter months
display(
    df_output
    .groupBy("created_month")
    .agg(
        f.count("*")
        , f.countDistinct("customer_id")
        , f.countDistinct("account_number")
    )
    .orderBy(f.desc("created_month"))
)

# COMMAND ----------

# DBTITLE 1,duplicate check
display(
    df_output
    .withColumn(
        "cnt"
        , f.count("*").over(
            Window
            .partitionBy("created_month", "customer_id", "account_number", "payment_status")
        )
    )
    .filter(f.col("cnt") > 1)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## s400 data export

# COMMAND ----------

# DBTITLE 1,delta
dir_data_output= os.path.join(dir_data_master, vt_param_export_file)

export_data(
    df = df_output
    , export_path = dir_data_output
    , export_format = "delta"
    , export_mode = "overwrite"
    , flag_overwrite_schema = False
    , flag_dynamic_partition = True
    , ls_dynamic_partition = ["created_month"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### snowflake

# COMMAND ----------

# DBTITLE 1,remove sql 01
vt_param_sql_remove = """
    DELETE FROM {{param_table}}
    WHERE CREATED_MONTH = '{{param_date}}'
"""

vt_param_sql_remove = Template(vt_param_sql_remove)

# COMMAND ----------

# DBTITLE 1,remove sql 02
ls_param_upload_months = pull_col(df_param_lookback_months.withColumn("reporting_date", f.trunc(f.col("reporting_date"), "MM").cast("string")), "reporting_date")

# COMMAND ----------

# DBTITLE 1,remove sql 03
for i in ls_param_upload_months:
    print(i)

    vt_param_sql_remove_curr = vt_param_sql_remove.render(
        param_table = vt_param_sf_db_table
        , param_date = i
    )

    sfUtils.runQuery(
        options
        , vt_param_sql_remove_curr
    )

# COMMAND ----------

# DBTITLE 1,upload SF
(
    df_output
    .write
    .format("snowflake")
    .options(**options)
    .option("dbtable", vt_param_sf_db_table)
    .mode("append")
    .save()
)
