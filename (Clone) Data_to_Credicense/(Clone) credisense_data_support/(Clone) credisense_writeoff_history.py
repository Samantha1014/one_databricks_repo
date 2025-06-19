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

# DBTITLE 1,brm
df_db_brm_item_t = spark.read.format('delta').load(os.path.join(dir_data_dl_brm, 'RAW_PINPAP_ITEM_T'))
df_db_brm_item_t = lower_col_names(df_db_brm_item_t)

df_db_brm_account_t = spark.read.format('delta').load(os.path.join(dir_data_dl_brm,'RAW_PINPAP_ACCOUNT_T'))
df_db_brm_account_t = lower_col_names(df_db_brm_account_t)


# COMMAND ----------

df_db_brm_wo = spark.sql(
    f"""
        with acct as (
            select 
                distinct account_no, poid_id0
            from delta.`{dir_data_dl_brm}/RAW_PINPAP_ACCOUNT_T`
        )
        
        , base_wo_hist as (
            select *
            from
                delta.`{dir_data_dl_brm}/RAW_PINPAP_ITEM_T`
            where
                _is_latest = 1
                and _is_deleted = 0
                and poid_type = '/item/writeoff'
        )
        , ranked_wo_hist as (
            select
                *
                , row_number() over (
                    partition by account_obj_id0
                    order by created_t asc, item_total asc
                ) as rnk
            from
                base_wo_hist
        )
        , filtered_wo_hist as (
            select
                *
                , date_format(
                    from_utc_timestamp(from_unixtime(created_t), 'Pacific/Auckland'),
                    'yyyy-MM-dd'
                ) as created_date
            from
                ranked_wo_hist 
            where
                rnk = 1
                and item_total <= -500
        )
        , joined_wo_hist as (
            select
                a.account_no
                , b.created_date
                , b.item_total
            from
                filtered_wo_hist as b
            inner join
                acct as a
            on
                b.account_obj_id0 = a.poid_id0
        )
        select
            account_no
            , created_date
            , item_total
        from
            joined_wo_hist
        where
            created_date >= '{vt_param_date_start}'
    """
)

display(df_db_brm_wo.limit(10))

# COMMAND ----------

# DBTITLE 1,siebel
df_db_sbl_cust = spark.sql(
    f"""
        with cust as (
            select 
                a.row_id as customer_id,
                b.birth_dt,
                b.fst_name,
                b.last_name,
                b.mid_name
            from delta.`{dir_data_dl_sbl}/RAW_SBL_ORG_EXT` as a
            left join delta.`{dir_data_dl_sbl}/RAW_SBL_CONTACT` as b
                on a.pr_con_id = b.row_id 
            where a.accnt_type_cd = 'Customer'
                and a._is_latest = 1 
                and a._is_deleted = 0 
                and b._is_latest = 1 
                and b._is_deleted = 0
        )
        
        , bill as (
            select 
                ou_num as account_no,
                row_id as billing_acct_id,
                par_ou_id as customer_id
            from delta.`{dir_data_dl_sbl}/RAW_SBL_ORG_EXT` as b
            where accnt_type_cd = 'Billing'
                and b._is_latest = 1 
                and b._is_deleted = 0
        )
        
        , output as (
            select 
                bill.*,
                cust.birth_dt,
                cust.fst_name,
                cust.last_name,
                cust.mid_name 
            from bill 
            left join cust 
                on bill.customer_id = cust.customer_id
        )
        select *
        from output;

    """
)

display(df_db_sbl_cust.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## s200 data processing

# COMMAND ----------

# DBTITLE 1,main
df_output = (
    df_db_brm_wo
    .join(
        df_db_sbl_cust
        , ["account_no"]
        , "left"
    )
    .withColumn(
        "rnk"
        , f.row_number().over(
            Window
            .partitionBy("customer_id")
            .orderBy(f.desc("created_date"))
        )
    )
    .filter(f.col("rnk") == 1)
    .select(
        f.col('customer_id').alias("customerid")
        , f.date_format('created_date','yyyy-MM-dd').alias('writeoffdate')
        , f.date_format('birth_dt', 'yyyy-MM-dd').alias('dateofbirth')
        , f.col('fst_name').alias('firstname')
        , f.col('last_name').alias("lastname")
        , f.col('mid_name').alias("middlename")
        , f.date_format(f.add_months('created_date', 60), 'yyyy-MM-dd').alias('expirydate')
        , f.lit('WRITEOFF').alias('casetype')
    )
    .filter(f.col("customerid").isNotNull())
    .withColumn(
        "searchterm"
        , f.concat(f.lower(f.trim("firstname")), f.lit(" "), f.lower(f.trim("lastname")), f.lit(" "), f.col("dateofbirth"))
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

display(df_output.limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC ## s300 data checking

# COMMAND ----------

display(
     df_output
     .agg(
          f.count('*')
          , f.countDistinct('customerid')
     )
)

# COMMAND ----------

# DBTITLE 1,data size by filter dates
display(
    df_output
    .withColumn(
        "mnth"
        , f.substring("writeoffdate", 1, 7)
    )
    .groupBy("mnth")
    .agg(
        f.count("*")
        , f.countDistinct("customerid")
    )
    .orderBy(f.desc("mnth"))
)

# COMMAND ----------

# DBTITLE 1,duplicate check
display(
    df_output
    .withColumn(
        "cnt"
        , f.count("*").over(
            Window
            .partitionBy("customerid")
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
    , flag_overwrite_schema = True
    , flag_dynamic_partition = True
    , ls_dynamic_partition = ["writeoffdate"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### snowflake

# COMMAND ----------

# DBTITLE 1,remove sql 01
vt_param_sql_remove = """
    DELETE FROM {{param_table}}
    WHERE WRITEOFFDATE = '{{param_date}}'
"""

vt_param_sql_remove = Template(vt_param_sql_remove)

# COMMAND ----------

# DBTITLE 1,remove sql 02
ls_param_writeoff_date = pull_col(df_output.withColumn("writeoffdate", f.col("writeoffdate").cast("string")).select("writeoffdate").distinct(), "writeoffdate")
ls_param_writeoff_date

# COMMAND ----------

# DBTITLE 1,remove sql 03
for i in ls_param_writeoff_date:
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
