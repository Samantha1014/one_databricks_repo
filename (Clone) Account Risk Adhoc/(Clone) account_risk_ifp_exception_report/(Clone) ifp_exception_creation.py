# Databricks notebook source
# MAGIC %md ## s000 environment setup

# COMMAND ----------

# MAGIC %md ### s001 libraries

# COMMAND ----------

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
from datetime import timedelta

# ------------- Use snowflake utility
sfUtils = sc._jvm.net.snowflake.spark.snowflake.Utils

# COMMAND ----------

df_check_01 = spark.read.format("delta").load("/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_ifp_device_on_bill")
df_check_02 = spark.read.format("delta").load("/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_ifp_device_on_service")

# COMMAND ----------

display(df_check_01.filter(f.col("ifp_bill_dvc_flag") == 'Y').limit(100))

# COMMAND ----------

display(
    df_check_02
    .filter(f.col("ifp_srvc_dvc_flag") == 'Y')
    .limit(100)
)

# COMMAND ----------

# MAGIC %md ### s002 parameters

# COMMAND ----------

vt_param_extract_date = date.today()
vt_param_extract_date = vt_param_extract_date - timedelta(days = 1)
vt_param_extract_date = vt_param_extract_date.strftime("%Y-%m-%d")

# COMMAND ----------

# MAGIC %md ### s003 DB connectivity

# COMMAND ----------

# ------------ login to snowflake
password = dbutils.secrets.get(scope = "auea-kv-sbx-dxdtlprdct01", key = "sfdbrsdskey")

options = {
  "sfUrl": "vodafonenz_prod.australia-east.azure.snowflakecomputing.com/", 
  "sfUser": "SVC_LAB_DS_DATABRICKS",
  "pem_private_key": password.replace('\\n', '\n'),
  "sfDatabase": "LAB_ML_STORE",
  "sfSchema": "RAW",
  "sfWarehouse": "LAB_DS_WH_SCALE"
}


# COMMAND ----------

# MAGIC %md ### s004 utility functions + external sql

# COMMAND ----------

# MAGIC %run "./utility_functions"

# COMMAND ----------

# MAGIC %run "./sql"

# COMMAND ----------

# MAGIC %md ### s005 directories

# COMMAND ----------

dir_data_master = "/mnt/ml-lab/dev_shared/tactical_solutions/account_risk_daily_exception"
dir_data_wip = os.path.join(dir_data_master, "wip")
dir_data_output = os.path.join(dir_data_master, "output")

# COMMAND ----------

# MAGIC %md ## s100 data import

# COMMAND ----------

vt_param_sql_active_base_core_curr = vt_param_sql_active_base_core.render(
    param_start_date = vt_param_extract_date
    , param_end_date = vt_param_extract_date
)

# COMMAND ----------

# MAGIC %md ### s101 active base

# COMMAND ----------

dir_data_wip_active_base = os.path.join(dir_data_wip, "raw_active_base")

vt_param_sql_active_base_curr = vt_param_sql_active_base.render(
    param_sql_active_base_core = vt_param_sql_active_base_core_curr
)

df_raw_base = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , vt_param_sql_active_base_curr
    )
    .load()
)

df_raw_base = lower_col_names(df_raw_base)

export_data(
    df = df_raw_base
    , export_path = dir_data_wip_active_base
    , export_format = "delta"
    , export_mode = "overwrite"
    , flag_overwrite_schema = True
    , flag_dynamic_partition = False    
)

df_raw_base = spark.read.format("delta").load(dir_data_wip_active_base)

# COMMAND ----------

display(df_raw_base.limit(100))

# COMMAND ----------

display(
    df_raw_base
    .groupBy('curr_mkt_seg_name')
    .agg(
        f.count('*').alias("cnt")
        , f.countDistinct("prim_accs_num").alias("conn")
        , f.countDistinct("acct_num").alias("acct")
    )
    .orderBy(f.desc("conn"))
)

# COMMAND ----------

# MAGIC %md ### s102 ifp on service

# COMMAND ----------

dir_data_wip_ifp_srvc = os.path.join(dir_data_wip, "raw_ifp_srvc")

vt_param_sql_ifp_srvc_curr = vt_param_sql_ifp_srvc.render(
    param_sql_active_base_core = vt_param_sql_active_base_core_curr
    , param_start_date = vt_param_extract_date
    , param_end_date = vt_param_extract_date
)

df_raw_ifp_srvc = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , vt_param_sql_ifp_srvc_curr
    )
    .load()
)

df_raw_ifp_srvc = lower_col_names(df_raw_ifp_srvc)

export_data(
    df = df_raw_ifp_srvc
    , export_path = dir_data_wip_ifp_srvc
    , export_format = "delta"
    , export_mode = "overwrite"
    , flag_overwrite_schema = True
    , flag_dynamic_partition = False    
)

df_raw_ifp_srvc = spark.read.format("delta").load(dir_data_wip_ifp_srvc)

# COMMAND ----------

display(
    df_raw_ifp_srvc
    .limit(100)
)

# COMMAND ----------

display(
    df_raw_ifp_srvc
    .agg(
        f.count("*").alias("cnt")
        , f.countDistinct("conn_id").alias("conn")
    )
)

# COMMAND ----------

# MAGIC %md ### s103 ifp on bill

# COMMAND ----------

dir_data_wip_ifp_bill = os.path.join(dir_data_wip, "raw_ifp_bill")

vt_param_sql_ifp_bill_curr = vt_param_sql_ifp_bill.render(
  param_sql_active_base_core = vt_param_sql_active_base_core_curr
  , param_start_date = vt_param_extract_date
  , param_end_date = vt_param_extract_date
)

df_raw_ifp_bill = (
  spark
  .read
  .format("snowflake")
  .options(**options)
  .option(
    "query"
    , vt_param_sql_ifp_bill_curr
    #, "select * from lab_ml_store.sandbox.account_risk_daily_exception_active_base"
  )
  .load()
)

df_raw_ifp_bill = lower_col_names(df_raw_ifp_bill)

export_data(
  df = df_raw_ifp_bill
  , export_path = dir_data_wip_ifp_bill
  , export_format = "delta"
  , export_mode = "overwrite"
  , flag_overwrite_schema = True
  , flag_dynamic_partition = False    
)

df_raw_ifp_bill = spark.read.format("delta").load(dir_data_wip_ifp_bill)

# COMMAND ----------

display(
    df_raw_ifp_bill
    .limit(100)
)

# COMMAND ----------

display(
    df_raw_ifp_bill
    .groupBy("ifp_type")
    #.distinct()
    .agg(
        f.count("*").alias("cnt")
        , f.countDistinct("acct_num").alias("acct")
    )
)

# COMMAND ----------

# MAGIC %md ## s200 data processing

# COMMAND ----------

# MAGIC %md ### s201 active base

# COMMAND ----------

display(
    df_raw_base
    .limit(100)
)

# COMMAND ----------

df_proc_base = (
    df_raw_base
    .withColumn(
        "segment"
        , f.col("curr_mkt_seg_name")
    )
    .withColumn(
        "acct_tenure"
        , f.round(
            f.months_between(
                f.lit(vt_param_extract_date)
                , f.col("acct_actv_dt")
            )
        )
    )
    .withColumn(
        "msisdn"
        , f.col("prim_accs_num")
    )
    .withColumn(
        "first_activation_date"
        , f.col("first_actvn_dt")
    )
    .withColumn(
        "srvc_tenure"
        , f.round(
            f.months_between(
                f.lit(vt_param_extract_date)
                , f.col("first_actvn_dt")
            )
        )
    )
    .withColumn(
        "srvc_array"
        , f.collect_list(
            f.struct(
                "msisdn"
                , "first_activation_date"
                , "srvc_tenure"
                #, "dctv_dt"
            )
        ).over(
            Window
            .partitionBy(["cust_src_id", "acct_num", "acct_src_id"])
        )
    )
    .withColumn(
        "srvc_cnt"
        , f.size(f.col("srvc_array"))
    )
    .groupBy(
        "cust_src_id"
        , "cust_full_name"
        , "acct_num"
        , "acct_src_id"
        , "acct_actv_dt"
        , "acct_tenure"
        , "segment"
        #, "dss_update_dttm"
    )
    .agg(
        f.countDistinct("prim_accs_num").alias("cnt")
        , f.first("srvc_cnt").alias("srvc_cnt")
        , f.first("srvc_array").alias("srvc_array")
    )
    #.withColumn("data_update_date", f.lit(vt_param_extract_date))
    .withColumn(
        "index"
        , f.row_number().over(
            Window
            .partitionBy("cust_src_id", "acct_num", "acct_src_id")
            .orderBy(f.desc("cnt"))
        )
    )
    .filter(f.col("index") == 1)
    .filter(f.col("acct_num").isNotNull())
    .drop("cnt")
)

display(
    df_proc_base
    .limit(100)
)

# COMMAND ----------

display(
    df_proc_base
    .withColumn(
        "cnt"
        , f.count("*").over(
            Window
            .partitionBy("acct_num")
        )
    )
    .filter(f.col("cnt") > 1)
)

# COMMAND ----------

display(
    df_proc_base
    .groupBy("segment")
    .agg(
        f.count("*")
        , f.countDistinct("acct_num")
        , f.sum("srvc_cnt")
    )
)

# COMMAND ----------

# MAGIC %md ### s202 ifp on service

# COMMAND ----------

vt_param_ssc_start_date = vt_param_extract_date
vt_param_ssc_end_date = vt_param_extract_date

# COMMAND ----------

display(
    df_raw_ifp_srvc
    .limit(100)
)

# COMMAND ----------

dir_data_wip_int_ifp_srvc = os.path.join(dir_data_wip, "int_ifp_srvc")

df_int_ifp_srvc = transform_ifp_srvc(
    df_raw_ifp_srvc
    .withColumn(
        "fs_cust_id"
        , f.col("cust_src_id")
    )
    .withColumn(
        "fs_acct_id"
        , f.col("acct_num")
    )
    .withColumn(
        "fs_srvc_id"
        , f.col("prim_accs_num")
    )
    .withColumn(
        "fs_ifp_order_id"
        , f.col("siebel_order_num")
    )
    , vt_param_ssc_start_date
    , vt_param_ssc_end_date
)


export_data(
  df = df_int_ifp_srvc
  , export_path = dir_data_wip_int_ifp_srvc
  , export_format = "delta"
  , export_mode = "overwrite"
  , flag_overwrite_schema = True
  , flag_dynamic_partition = False    
)

df_int_ifp_srvc = spark.read.format("delta").load(dir_data_wip_int_ifp_srvc)

# COMMAND ----------

display(
    df_int_ifp_srvc
    .groupBy("ifp_event_type")
    .agg(
        f.count("*").alias("cnt")
        , f.countDistinct("conn_id").alias("conn")
    )
)

display(
    df_int_ifp_srvc
    .limit(100)
)

display(
    df_int_ifp_srvc
    .filter(f.col("ifp_event_type") == 'finish')
    .limit(100)
)

# COMMAND ----------

df_proc_ifp_srvc = (
    df_int_ifp_srvc
    .filter(f.col("ifp_end_date") >= vt_param_extract_date)
    .withColumn(
        "cust_src_id"
        , f.col("fs_cust_id")
    )
    .withColumn(
        "acct_num"
        , f.col("fs_acct_id")
    )
    .join(
        df_proc_base
        .select(
            "cust_src_id"
            , "acct_num"
            , "acct_src_id"
        )
        , ["cust_src_id", "acct_num"]
    )
    .select(
        "cust_src_id"
        , "acct_num"
        , "acct_src_id"
        , f.col("fs_ifp_id").alias("ifp_id")
        , "ifp_level"
        , "ifp_type"
        , "ifp_order_num"
        , f.col("ifp_event_date").alias("ifp_order_date")
        , "ifp_model"
        , "ifp_value"
        , "ifp_term"
        , "ifp_term_start_date"
        , "ifp_term_end_date"
        , "ifp_sales_channel_group"
        , 'ifp_sales_channel'
        , "ifp_sales_channel_branch"
        , "ifp_sales_agent"
    )
)

display(
    df_proc_ifp_srvc
    .limit(100)
)

display(
    df_proc_ifp_srvc
    .agg(
        f.count("*")
        , f.countDistinct("ifp_id")
        , f.countDistinct("acct_num")
    )
)

# COMMAND ----------

# MAGIC %md ### s203 ifp on bill

# COMMAND ----------

dir_data_wip_int_ifp_bill = os.path.join(dir_data_wip, "int_ifp_bill")

df_int_ifp_bill = transform_ifp_bill(
    df_raw_ifp_bill
    .withColumn(
        "fs_acct_src_id"
        , f.col("acct_src_id")
    )
    .withColumn(
        "fs_acct_id"
        , f.col("acct_num")
    )
    .withColumn(
        "fs_ifp_id"
        , f.col("ifp_txn_id")
    )
    .withColumn(
        "fs_ifp_order_id"
        , f.col("siebel_order_num")
    )
    , vt_param_ssc_start_date
    , vt_param_ssc_end_date
)

export_data(
  df = df_int_ifp_bill
  , export_path = dir_data_wip_int_ifp_bill
  , export_format = "delta"
  , export_mode = "overwrite"
  , flag_overwrite_schema = True
  , flag_dynamic_partition = False    
)

df_int_ifp_bill = spark.read.format("delta").load(dir_data_wip_int_ifp_bill)


# COMMAND ----------

display(
    df_int_ifp_bill
    .limit(100)
)

# COMMAND ----------

display(
    df_int_ifp_bill
    .groupBy("ifp_event_type", "ifp_type")
    .agg(
        f.count("*").alias("cnt")
        , f.countDistinct("fs_acct_src_id").alias("acct")
        , f.countDistinct("fs_ifp_id").alias("ifp")
    )
)

display(
    df_int_ifp_bill
    .limit(100)
)

# COMMAND ----------

df_proc_ifp_bill = (
    df_int_ifp_bill
    .filter(f.col("ifp_end_date") >= vt_param_extract_date)
    #.filter(f.col("ifp_type").isin(["device"]))
    .withColumn(
        "acct_num"
        , f.col("fs_acct_id")
    )
    .withColumn(
        "acct_src_id"
        , f.col("fs_acct_src_id")
    )
    .join(
        df_proc_base
        .select(
            "cust_src_id"
            , "acct_num"
            , "acct_src_id"
        )
        , ["acct_num", "acct_src_id"]
    )
    .select(
        "cust_src_id"
        , "acct_num"
        , "acct_src_id"
        , f.col("fs_ifp_id").alias("ifp_id")
        , "ifp_level"
        , "ifp_type"
        , "ifp_order_num"
        , f.col("ifp_event_date").alias("ifp_order_date")
        , "ifp_model"
        , "ifp_value"
        , "ifp_term"
        , "ifp_term_start_date"
        , "ifp_term_end_date"
        , "ifp_sales_channel_group"
        , 'ifp_sales_channel'
        , "ifp_sales_channel_branch"
        , "ifp_sales_agent"
    )
)

display(
    df_proc_ifp_bill
    .limit(100)
)

display(
    df_proc_ifp_bill
    .agg(
        f.count("*")
        , f.countDistinct("ifp_id")
        , f.countDistinct("acct_num")
    )
)

# COMMAND ----------

df_proc_ifp = (
    df_proc_ifp_srvc
    .unionByName(
        df_proc_ifp_bill
    )
)


display(
    df_proc_ifp
    .groupBy("ifp_level", "ifp_type")
    .agg(
        f.count("*")
        , f.countDistinct("ifp_id")
        , f.countDistinct("acct_num")
    )
)

# COMMAND ----------

# MAGIC %md ### s204 ifp on account

# COMMAND ----------

display(
    df_proc_ifp
    .limit(100)
)

# COMMAND ----------

df_proc_ifp_acct_dvc = (
    df_proc_ifp
    .filter(f.col("ifp_type") == 'device')
    .withColumn(
        "index"
        , f.row_number().over(
            Window
            .partitionBy(["cust_src_id", "acct_num", "acct_src_id"])
            .orderBy(f.desc("ifp_term_start_date"), f.desc("ifp_order_date"), f.desc("ifp_term_end_date"))
        )
    )
    .withColumn(
        "ifp_dvc_array"
        , f.collect_list(
            f.struct(
                "ifp_id"
                , "ifp_level"
                , "ifp_order_num"
                , "ifp_order_date"
                , "ifp_model"
                , "ifp_value"
                , "ifp_term"
                , "ifp_term_start_date"
                , "ifp_term_end_date"
                , "ifp_sales_channel_group"
                , "ifp_sales_channel"
                , "ifp_sales_channel_branch"
                , "ifp_sales_agent"
            )
        ).over(
            Window
            .partitionBy(["cust_src_id", "acct_num", "acct_src_id"])
            #.orderBy(f.desc("ifp_term_end_date"), f.desc("ifp_term_start_date"), f.desc("ifp_order_date"))
        )
    )
    .withColumn("ifp_dvc_cnt", f.size("ifp_dvc_array"))
    .filter(f.col("index") == 1)
    .select(
        "cust_src_id"
        , "acct_num"
        , "acct_src_id"
        , "ifp_dvc_cnt"
        , "ifp_dvc_array"
        , f.col("ifp_id").alias("last_ifp_id")
        , f.col("ifp_order_num").alias("last_ifp_order_num")
        , f.col("ifp_order_date").alias("last_ifp_order_date")
        , f.col("ifp_model").alias("last_ifp_model")
        , f.col("ifp_value").alias("last_ifp_value")
        , f.col("ifp_term").alias("last_ifp_term")
        , f.col("ifp_term_start_date").alias("last_ifp_term_start_date")
        , f.col("ifp_term_end_date").alias("last_ifp_term_end_date")
        , f.col("ifp_sales_channel_group").alias("last_ifp_channel_group")
        , f.col("ifp_sales_channel").alias("last_ifp_channel")
        , f.col("ifp_sales_channel_branch").alias("last_ifp_sales_channel_branch")
        , f.col("ifp_sales_agent").alias("last_ifp_sales_agent")
        , f.col("index")
    )
)

display(
    df_proc_ifp_acct_dvc
    .filter(f.col("acct_num") == '337786300')
    #.limit(100)
)

# COMMAND ----------

df_proc_ifp_acct_accs = (
    df_proc_ifp
    .filter(f.col("ifp_type") == 'accessory')
    .withColumn(
        "index"
        , f.row_number().over(
            Window
            .partitionBy(["cust_src_id", "acct_num", "acct_src_id"])
            .orderBy(f.desc("ifp_term_start_date"), f.desc("ifp_order_date"), f.desc("ifp_term_end_date"))
        )
    )
    .withColumn(
        "ifp_accs_array"
        , f.collect_list(
            f.struct(
                "ifp_id"
                , "ifp_level"
                , "ifp_order_num"
                , "ifp_order_date"
                , "ifp_model"
                , "ifp_value"
                , "ifp_term"
                , "ifp_term_start_date"
                , "ifp_term_end_date"
                , "ifp_sales_channel_group"
                , "ifp_sales_channel"
                , "ifp_sales_channel_branch"
                , "ifp_sales_agent"
            )
        ).over(
            Window
            .partitionBy(["cust_src_id", "acct_num", "acct_src_id"])
        )
    )
    .withColumn("ifp_accs_cnt", f.size("ifp_accs_array"))
    .filter(f.col("index") == 1)
    .select(
        "cust_src_id"
        , "acct_num"
        , "acct_src_id"
        , "ifp_accs_cnt"
        , "ifp_accs_array"
    )
)

display(
    df_proc_ifp_acct_accs
    .limit(100)
)

# COMMAND ----------

# MAGIC %md ### s205 combine

# COMMAND ----------

display(df_proc_base.limit(100))
display(df_proc_ifp_acct_dvc.limit(100))
display(df_proc_ifp_acct_accs.limit(100))

# COMMAND ----------

df_model = (
    df_proc_base
    .join(
        df_proc_ifp_acct_dvc
        .withColumn("ifp_dvc_flag", f.lit('Y'))
        , ["cust_src_id", "acct_num", "acct_src_id"]
        , "left"
    )
    .join(
        df_proc_ifp_acct_accs
        .withColumn("ifp_accs_flag", f.lit('Y'))
        , ["cust_src_id", "acct_num", "acct_src_id"]
        , "left"
    )
    .fillna(value = 'N', subset = ["ifp_dvc_flag", "ifp_accs_flag"])
    .fillna(value = 0, subset = ["ifp_dvc_cnt", "ifp_accs_cnt"])
    .withColumn(
        "breach_flag_01"
        , f.when(
            (f.col("acct_tenure") <= 4)
            & (f.col("ifp_dvc_cnt") > 1)
            , f.lit(1)
        ).otherwise(f.lit(0))
    )
    .withColumn(
        "breach_score_01"
        , f.when(
            f.col("breach_flag_01") == 1
            , f.col("ifp_dvc_cnt") - 1
        ).otherwise(f.lit(0))
    )
    .withColumn(
        "breach_flag_02"
        , f.when(
            (f.col("ifp_dvc_cnt") > 4)
            , f.lit(1)
        ).otherwise(f.lit(0))
    )
    .withColumn(
        "breach_score_02"
        , f.when(
            f.col("breach_flag_02") == 1
            , f.col("ifp_dvc_cnt") - 4
        ).otherwise(f.lit(0))
    )
    .withColumn(
        "breach_score"
        #, f.col("breach_flag_01") + f.col("breach_flag_02")
        , f.col("breach_score_01") + f.col("breach_score_02")
    )
    .withColumn(
        "breach_flag_01"
        , f.when(f.col("breach_flag_01") == 1, f.lit('Y')).otherwise(f.lit('N'))
    )
    .withColumn(
        "breach_flag_02"
        , f.when(f.col("breach_flag_02") == 1, f.lit('Y')).otherwise(f.lit('N'))
    )
    .withColumn(
        "data_update_dttm"
        , f.current_timestamp()
    )
    .select(
        f.lit(vt_param_extract_date).alias("reporting_date")
        , f.col("cust_src_id").alias("customer_source_id")
        , f.col("acct_src_id").alias("account_source_id")
        , f.col("acct_num").alias("account_number")
        , f.col("cust_full_name").alias("customer_name")
        , "segment"
        , f.col("acct_actv_dt").alias("account_open_date")
        , f.col("acct_tenure").alias("account_tenure")
        , f.col("srvc_cnt").alias("mobile_service")
        , f.col("ifp_dvc_flag").alias("ifp_device_flag")
        , f.col("ifp_dvc_cnt").alias("ifp_device")
        , f.col("ifp_accs_cnt").alias("ifp_accessory")
        , "breach_flag_01"
        , "breach_flag_02"
        , "breach_score"
        , "last_ifp_id"
        , "last_ifp_order_num"
        , "last_ifp_order_date"
        , "last_ifp_model"
        , "last_ifp_value"
        , "last_ifp_term"
        , "last_ifp_term_start_date"
        , "last_ifp_term_end_date"
        , "last_ifp_channel_group"
        , "last_ifp_channel"
        , "last_ifp_sales_channel_branch"
        , "last_ifp_sales_agent"
        , "srvc_array"
        , "ifp_dvc_array"
        , "ifp_accs_array"
        , f.current_date().alias("data_update_date")
        , f.current_timestamp().alias("data_update_dttm")
    )
)

display(
    df_model
    .filter(f.col("breach_score") > 0)
    .groupBy("segment", "breach_flag_01", "breach_flag_02")
    .agg(
        f.count("*")
        , f.countDistinct("account_number")
    )
)

display(
    df_model
    .filter(f.col("breach_score") > 0)
)

# COMMAND ----------

display(
    df_model
    .groupBy("segment")
    .agg(
        f.count("*")
        , f.countDistinct("account_number")
        , f.sum("ifp_device")
        , f.sum("ifp_accessory")
    )
)

# COMMAND ----------

# MAGIC %md ## s300 data export

# COMMAND ----------

# MAGIC %md ### s301 delta

# COMMAND ----------

dir_data_output_ifp_exception_full = os.path.join(dir_data_output, "ifp_exception_full")

export_data(
    df = df_model
    , export_path = dir_data_output_ifp_exception_full
    , export_format = "delta"
    , export_mode = "overwrite"
    , flag_overwrite_schema = True
    , flag_dynamic_partition = True
    , ls_dynamic_partition=["reporting_date"]
)

# COMMAND ----------

# MAGIC %md ### s302 snowflake

# COMMAND ----------

vt_param_table_curr = "lab_ml_store.ml_marketing.account_risk_ifp_exception_report_hist"

df_upload = (
    df_model
    .filter(f.col("breach_score") > 0)
    .withColumn("active_flag", f.lit('N'))
)

vt_param_sql_upload_remove_curr = vt_param_sql_upload_remove.render(
    param_table = vt_param_table_curr
    , param_date = vt_param_extract_date
)

print(vt_param_sql_upload_remove_curr)

sfUtils.runQuery(options, vt_param_sql_upload_remove_curr)

(
    df_upload
    .write
    .format("snowflake")
    .options(**options)
    .option("dbtable", vt_param_table_curr)
    .mode("append")
    .save()
)

# COMMAND ----------

vt_param_sql_upload_update_01_curr = vt_param_sql_upload_update_01.render(
    param_table = vt_param_table_curr
)

print(vt_param_sql_upload_update_01_curr)

sfUtils.runQuery(options, vt_param_sql_upload_update_01_curr)

vt_param_sql_upload_update_02_curr = vt_param_sql_upload_update_02.render(
    param_table = vt_param_table_curr
)

print(vt_param_sql_upload_update_02_curr)

sfUtils.runQuery(options, vt_param_sql_upload_update_02_curr)
