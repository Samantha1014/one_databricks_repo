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

# DBTITLE 1,parameters 00
#dbutils.widgets.text(name="v00_param_date", defaultValue="")
#dbutils.widgets.text(name="v00_param_lookback_days", defaultValue="")
#dbutils.widgets.text(name="v01_master_dir", defaultValue="")
#dbutils.widgets.text(name="v02_export_file", defaultValue="")
#dbutils.widgets.text(name="v03_sf_db", defaultValue="")
#dbutils.widgets.text(name="v04_sf_db_schema", defaultValue="")

# COMMAND ----------

vt_param_date = dbutils.widgets.get('v00_param_date')
vt_param_lookback_days = float(dbutils.widgets.get("v00_param_lookback_days"))
dir_data_master = dbutils.widgets.get("v01_master_dir")
vt_param_export_file = dbutils.widgets.get("v02_export_file")
vt_param_sf_db = dbutils.widgets.get("v03_sf_db")
vt_param_sf_db_schema= dbutils.widgets.get("v04_sf_db_schema")

# COMMAND ----------

vt_param_date

# COMMAND ----------

# DBTITLE 1,parameters 02
if vt_param_date == '':
    dbutils.notebook.exit("no date supplied")

# COMMAND ----------

# DBTITLE 1,parameters 03
vt_param_date = datetime.strptime(vt_param_date, '%Y-%m-%d').date()
vt_param_date_end = vt_param_date - timedelta(days = 1)
vt_param_date_start = vt_param_date_end - timedelta(days = vt_param_lookback_days)

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
dir_data_dl_brm = 'dbfs:/mnt/prod_brm/raw/cdc'

# COMMAND ----------

# MAGIC %md
# MAGIC ## s100 data import

# COMMAND ----------

# DBTITLE 1,account base
df_db_acct = spark.sql(
    f"""
        select
            distinct
            ou_num as account_number
            , par_ou_id as customer_id
        from delta.`{dir_data_dl_sbl}/RAW_SBL_ORG_EXT`
        where 1 = 1
            and _is_latest = 1
            and accnt_type_cd = 'Billing'
    """
)

# COMMAND ----------

# DBTITLE 1,payment arrangement
df_db_payment_arrange = spark.sql(
    f"""
        with base as (
            select 
                acct.account_no as account_number,
                ca.created_t as arrangement_t,
                to_date(from_utc_timestamp(from_unixtime(ca.created_t), 'Pacific/Auckland')) as arrangement_date,
                bi.pending_recv as old_balance,
                cpm.amount as installment_amount,
                cpm.descr as arrangement_type
            from delta.`{dir_data_dl_brm}/RAW_PINPAP_COLLECTIONS_ACTION_T` as ca
            inner join delta.`{dir_data_dl_brm}/RAW_PINPAP_COLLECTIONS_SCENARIO_T` as cs 
                on ca.scenario_obj_id0 = cs.poid_id0
                and cs._is_latest = 1 
                and cs._is_deleted = 0
            inner join delta.`{dir_data_dl_brm}/RAW_PINPAP_ACCOUNT_T` as acct 
                on ca.account_obj_id0 = acct.poid_id0 
                and acct._is_latest = 1 
                and acct._is_deleted = 0
            inner join delta.`{dir_data_dl_brm}/RAW_PINPAP_VF_COLL_P2P_MILESTONES_T` as cpm
                on cpm.action_obj_id0 = ca.poid_id0 
                and cpm._is_latest = 1 
                and cpm._is_deleted = 0
            inner join delta.`{dir_data_dl_brm}/RAW_PINPAP_BILLINFO_T` as bi 
                on bi.account_obj_id0 = acct.poid_id0
                and bi._is_latest = 1 
                and bi._is_deleted = 0 
            where 
                ca.poid_type = '/collections_action/promise_to_pay'
                and ca._is_latest = 1 
                and ca._is_deleted = 0 
        )
        select * from base
        where 1 = 1
            and arrangement_t >= unix_timestamp('{vt_param_date_start}', 'yyyy-MM-dd')
            and arrangement_t <= unix_timestamp('{vt_param_date_end}', 'yyyy-MM-dd')
    """
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## s200 data processing

# COMMAND ----------

# DBTITLE 1,main
df_output = (
    df_db_payment_arrange
    .groupBy(
        "account_number"
        , "arrangement_date"
        , "arrangement_type"
    )
    .agg(
        f.round(f.sum("installment_amount")).alias("balance")
        , f.count("*").alias("number_of_installment")
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
    .select(
        "customer_id"
        , "account_number"
        , "arrangement_date"
        , "arrangement_type"
        , "balance"
        , "number_of_installment"
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

# DBTITLE 1,data size by filter dates
display(
    df_output
    .groupBy("arrangement_date")
    .agg(
        f.count("*")
        , f.countDistinct("customer_id")
        , f.countDistinct("account_number")
    )
    .orderBy(f.desc("arrangement_date"))
)

# COMMAND ----------

# DBTITLE 1,duplicate check
display(
    df_output
    .withColumn(
        "cnt"
        , f.count("*").over(
            Window
            .partitionBy("customer_id", "account_number", "arrangement_date", "arrangement_type", "balance", "number_of_installment")
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
    , ls_dynamic_partition = ["arrangement_date"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### snowflake

# COMMAND ----------

# DBTITLE 1,remove sql 01
vt_param_sql_remove = """
    DELETE FROM {{param_table}}
    WHERE ARRANGEMENT_DATE = '{{param_date}}'
"""

vt_param_sql_remove = Template(vt_param_sql_remove)

# COMMAND ----------

# DBTITLE 1,remove sql 02
ls_param_arrangement_date = pull_col(df_output.withColumn("arrangement_date", f.col("arrangement_date").cast("string")).select("arrangement_date").distinct(), "arrangement_date")
ls_param_arrangement_date

# COMMAND ----------

# DBTITLE 1,remove sql 03
for i in ls_param_arrangement_date:
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
