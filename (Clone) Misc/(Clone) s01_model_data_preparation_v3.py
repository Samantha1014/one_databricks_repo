# Databricks notebook source
# MAGIC %md
# MAGIC ### s1 environment setup

# COMMAND ----------

# DBTITLE 1,environment configs
# MAGIC %run "./s98_environment_setup"

# COMMAND ----------

# DBTITLE 1,libraries
import os
import pyspark

from pyspark import sql
from pyspark.sql import Window
from pyspark.sql import functions as f

import pandas as pd

# COMMAND ----------

# DBTITLE 1,utils - mlf
# MAGIC %run "./s99_utility_functions"

# COMMAND ----------

# DBTITLE 1,utils - spark df
# MAGIC %run "/Workspace/Users/ethan.li@one.nz/s04_resources/s01_utility_functions/spark_df_utils"

# COMMAND ----------

# DBTITLE 1,utils - qa
# MAGIC %run "/Workspace/Users/ethan.li@one.nz/s04_resources/s01_utility_functions/qa_utils"

# COMMAND ----------

# DBTITLE 1,utils - sampling
# MAGIC %run "Workspace/Users/ethan.li@one.nz/s04_resources/s01_utility_functions/utils_stratified_sampling"

# COMMAND ----------

get_db_notebook_dir(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())

# COMMAND ----------

# DBTITLE 1,sf  connection
# snowflake connector
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

# DBTITLE 1,parameters - fs
# feature store parameters
dir_data_fs_parent = "/mnt/feature-store-prod-lab"

vt_param_fs_reporting_cycle_type = "calendar cycle"
vt_param_fs_reporting_freq_type = "proc_freq_monthly_flag"
vt_param_fs_reporting_freq_label = "monthly"
ls_param_fs_reporting_keys = ["reporting_cycle_type", "reporting_date"] 
ls_param_fs_primary_keys = ["fs_cust_id", "fs_acct_id", "fs_srvc_id"]

# COMMAND ----------

# DBTITLE 1,parameters - mlf
# ml factory parameters
# generic
vt_param_mlf_model_pattern = "classification"

vt_param_mlf_user_id = "dev_el"
dir_data_mlf_parent = "/mnt/ml-factory-gen-x-dev"

# model
vt_param_mlf_model_id = "mobile_oa_consumer_srvc_churn_pred30d"

# experiment
vt_param_mlf_exp_id = "mobile_oa_consumer_srvc_churn_pred30d_202412_exp1"

# model data
vt_param_mlf_model_feature_meta = "LAB_ML_STORE.SANDBOX.MLF_FEA_META_MOBILE_OA_CONSUMER_SRVC_CHURN_PRED30D_202501_EXP2" 

vt_param_mlf_model_data_start_date = '2023-06-30'
vt_param_mlf_model_data_end_date = "2024-11-30"

vt_param_mlf_model_data_version_id = "v3"
vt_param_mlf_model_data_predict_days = 30
vt_param_mlf_model_data_predict_date_from_base_date = 1
vt_param_mlf_model_data_valid_pct = 0.1
vt_param_mlf_model_data_calibrate_pct = 0.1

ls_param_mlf_model_data_training_date = (
    pd
    .date_range(start = '2023-06-30', end = '2024-06-30', freq = 'M')
    .tolist()
)

ls_param_mlf_model_data_blend_date = (
    pd
    .date_range(start = '2024-07-01', end = '2024-07-31', freq = 'M')
    .tolist()
)

ls_param_mlf_model_data_holdout_date = (
    pd
    .date_range(start = '2024-08-31', end = '2024-11-30', freq = 'M')
    .tolist()
)

# COMMAND ----------

# DBTITLE 1,directories
# data directories
# feature store
dir_data_fs_meta = os.path.join(dir_data_fs_parent, "d000_meta")
dir_data_fs_fea = os.path.join(dir_data_fs_parent, "d400_feature")
dir_data_fs_mvmt = os.path.join(dir_data_fs_parent, "d500_movement")
dir_data_fs_serv = os.path.join(dir_data_fs_parent, "d600_serving")

# ml factory
dir_data_mlf_exp = os.path.join(dir_data_mlf_parent, "dev_users", vt_param_mlf_user_id, vt_param_mlf_model_pattern, vt_param_mlf_model_id, vt_param_mlf_exp_id)

dir_data_mlf = dir_data_mlf_exp

# COMMAND ----------

# MAGIC %md
# MAGIC ### s2 data import

# COMMAND ----------

# DBTITLE 1,import - master
df_fs_global_cycle_calendar = spark.read.format("delta").load(os.path.join(dir_data_fs_meta, "d001_global_cycle_calendar"))
df_fsr_dict_meta = spark.read.format("delta").load(os.path.join(dir_data_fs_meta, "d004_fsr_meta/fsr_dict_meta"))
df_fs_serv = spark.read.format("delta").load(os.path.join(dir_data_fs_serv, "serv_mobile_oa_consumer"))
df_fs_mvmt_srvc_deact = spark.read.format("delta").load(os.path.join(dir_data_fs_mvmt, "d501_mobile_oa_consumer/mvmt_service_deactivation"))

# COMMAND ----------

# DBTITLE 1,import - meta
df_mlf_feature_meta = lower_col_names(
    spark.read.format("snowflake")
    .options(**options)
    .option(
        "query"
        , f"select * from {vt_param_mlf_model_feature_meta}"
    )
    .load()
)

print("feature meta")
display(df_mlf_feature_meta)

# COMMAND ----------

# DBTITLE 1,sample output
print("global cycle calendar")
display(df_fs_global_cycle_calendar.limit(10))

print("fs serv")
display(df_fs_serv.limit(10))

print("deact")
display(df_fs_mvmt_srvc_deact.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### s3 data preparation

# COMMAND ----------

# MAGIC %md
# MAGIC #### s401 model meta

# COMMAND ----------

# DBTITLE 1,model parameter check
print(vt_param_fs_reporting_cycle_type)
print(vt_param_fs_reporting_freq_type)
print(vt_param_mlf_model_data_start_date)
print(vt_param_mlf_model_data_end_date)

# COMMAND ----------

# DBTITLE 1,model prediction meta
# model ssc meta for extraction
df_model_ssc_meta = (
    df_fs_global_cycle_calendar
    .filter(f.col("cycle_type") == vt_param_fs_reporting_cycle_type)
    .filter(f.col(vt_param_fs_reporting_freq_type) == 'Y')
    .filter(f.col("base_date") <= vt_param_mlf_model_data_end_date)
    .filter(f.col("base_date") >= vt_param_mlf_model_data_start_date)
    .withColumn(
        "target_ssc_start_date"
        #, f.date_add(f.col("base_date"), 1)
        , f.date_add(f.col("base_date"), vt_param_mlf_model_data_predict_date_from_base_date)
    )
    .withColumn(
        "target_ssc_end_date"
        #, f.date_add(f.col("base_date"), vt_param_mlf_model_data_predict_days)
        , f.date_add(f.col("base_date"), (vt_param_mlf_model_data_predict_date_from_base_date + vt_param_mlf_model_data_predict_days) - 1)
        
    )
    .withColumn(
        "model_period_type"
        , f.when(
            f.col("base_date").isin(ls_param_mlf_model_data_holdout_date)
            , "holdout"
        ).when(
            f.col("base_date").isin(ls_param_mlf_model_data_blend_date)
            , "blending"
        ).otherwise("others")
    )
    .select(
        f.col("cycle_type").alias("reporting_cycle_type")
        , f.col("base_date").alias("reporting_date")
        , f.col("base_snapshot_date_start").alias("feature_ssc_start_date")
        , f.col("base_snapshot_date_end").alias("feature_ssc_end_date")
        , "target_ssc_start_date"
        , "target_ssc_end_date"
        , "model_period_type"
    )
    .orderBy(f.desc("reporting_date"))
)

display(df_model_ssc_meta.limit(100))

# COMMAND ----------

# DBTITLE 1,interpolation for target dates
# target extractor
df_model_target_ssc_meta = (
    df_model_ssc_meta
    .withColumn(
        "target_date"
        , interpolation_linear(
            f.col("target_ssc_start_date")
            , f.col("target_ssc_end_date")
        )
    )
    .select(
        "reporting_cycle_type"
        , "reporting_date"
        , "target_date"
    )
    .orderBy(f.desc("reporting_date"), f.desc("target_date"))
)

display(df_model_target_ssc_meta)

# COMMAND ----------

# MAGIC %md
# MAGIC #### s402 model unit base creation

# COMMAND ----------

# DBTITLE 1,unit base
# create model unit base from the feature store
df_model_unit_base = (
    df_fs_serv
    .filter(f.col("reporting_date").between(vt_param_mlf_model_data_start_date, vt_param_mlf_model_data_end_date))
    .join(
        df_model_ssc_meta
        .select(ls_param_fs_reporting_keys)
        , ls_param_fs_reporting_keys
        , "inner"
    )
    .select(*ls_param_fs_reporting_keys, *ls_param_fs_primary_keys)
)

# COMMAND ----------

# DBTITLE 1,data check
print("sample check")
display(df_model_unit_base.limit(10))

print("summary check")
display(
    df_model_unit_base
    .groupBy("reporting_cycle_type", "reporting_date")
    .agg(
        f.count("*")
        , f.countDistinct("fs_srvc_id")
    )
    .orderBy(f.desc("reporting_date"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### s402 model target creation
# MAGIC Target label table creation based on:
# MAGIC - global cycle calendar
# MAGIC - unit base from feature store
# MAGIC - deactivation movement

# COMMAND ----------

# DBTITLE 1,movement data check
display(df_fs_mvmt_srvc_deact.limit(10))

# COMMAND ----------

# DBTITLE 1,movement creation
# extract voluntary deactivation
df_base_mvmt_srvc_deact = (
    df_fs_mvmt_srvc_deact
    .filter(f.col("deactivate_type") == 'Voluntary')
    .drop("reporting_date", "reporting_cycle_type")
    .withColumnRenamed("movement_date", "target_date")
    .join(
        df_model_target_ssc_meta
        , ["target_date"]
        , "inner"
    )
    .join(
        df_model_unit_base
        , ls_param_fs_reporting_keys + ls_param_fs_primary_keys 
        , "inner"
    )
    .select(
        *ls_param_fs_reporting_keys
        , *ls_param_fs_primary_keys
        , "target_date"
        , "deactivate_reason_std"
    )
    .withColumn(
        "index"
        , f.row_number().over(
            Window
            .partitionBy(ls_param_fs_reporting_keys + ls_param_fs_primary_keys)
            .orderBy(f.desc("target_date"))
        )
    )
    .filter(f.col("index") == 1)
    .drop("index")
)

display(
    df_base_mvmt_srvc_deact 
    .limit(100)
)

# COMMAND ----------

# DBTITLE 1,data check
print("sample check")
display(df_base_mvmt_srvc_deact.limit(10))

print("summary check")
display(
    df_base_mvmt_srvc_deact
    .groupBy("reporting_date")
    .agg(
        f.count("*")
        , f.countDistinct("fs_srvc_id")
        , f.min("target_date")
        , f.max("target_date")
    )
    .orderBy(f.desc("reporting_date"))
)

# COMMAND ----------

# DBTITLE 1,target creation
df_model_target = (
    df_model_unit_base
    .join(
        df_base_mvmt_srvc_deact
        .select(
            *ls_param_fs_reporting_keys
            , *ls_param_fs_primary_keys
            , f.lit("Y").alias("target_label")
        )
        , ls_param_fs_reporting_keys + ls_param_fs_primary_keys
        , "left"
    )
    .fillna(value='N', subset=["target_label"])
)

# COMMAND ----------

# DBTITLE 1,data check
print("sample check")
display(df_model_target.limit(10))

print("summary check")
display(
    df_model_target
    .groupBy("reporting_date", "target_label")
    .agg(f.countDistinct("fs_srvc_id").alias("conn"))
    .withColumn(
        "conn_tot"
        , f.sum("conn").over(
            Window
            .partitionBy("reporting_date")
        )
    )
    .withColumn(
        "pct"
        , f.round(f.col("conn")/f.col("conn_tot") * 100, 2)
    )
    .groupBy("reporting_date")
    .pivot("target_label")
    .agg(
        f.sum("conn")
        , f.sum("pct")
    )
    .orderBy(f.desc("reporting_date"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### s404 model feature creation
# MAGIC
# MAGIC Model Data Creation
# MAGIC - create model feature table with selected features
# MAGIC - left join model feature table to model traget table based on reporting keys & primary keys
# MAGIC - model specific data transformation

# COMMAND ----------

# DBTITLE 1,model raw 00
# model features creation
ls_param_model_features = pull_col(df_mlf_feature_meta, "feature")

df_model_features = (
     df_fs_serv
    .select(
        ls_param_fs_reporting_keys 
        + ls_param_fs_primary_keys 
        + ls_param_model_features
    )
)

display(df_model_features.limit(10))

# COMMAND ----------

# DBTITLE 1,model raw 01
# combine model features & target
df_model_raw = (
    df_model_unit_base
    .join(
        df_model_target
        , ls_param_fs_reporting_keys + ls_param_fs_primary_keys
        , "left"
    )
    .join(
        df_model_features
        , ls_param_fs_reporting_keys + ls_param_fs_primary_keys
        , "left"
    )
)

display(df_model_raw.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### pre-run check

# COMMAND ----------

# DBTITLE 1,schema check
df_model_raw_schema = get_spkdf_schema(df_model_raw)

display(df_model_raw_schema)

# COMMAND ----------

# DBTITLE 1,data check - missing
# checking missing values for the selected features
check_missing_features(
    df_model_raw
    , "fs_srvc_id"
    , ls_param_model_features
    , ls_param_fs_reporting_keys
    , "pct"
)

# COMMAND ----------

# DBTITLE 1,data check - distinct
ls_param_model_features_char = pull_col(
    df_model_raw_schema
    .filter(f.col("field").isin(ls_param_model_features))
    .filter(f.col("type") == "string")
    , "field"
)

check_distinct_values(
    df_model_raw
    , ls_param_model_features_char
    , ls_param_fs_reporting_keys
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### feature engineering

# COMMAND ----------

# DBTITLE 1,initial
df_model_prep = df_model_raw

# COMMAND ----------

# MAGIC %md
# MAGIC ###### rate plan

# COMMAND ----------

# DBTITLE 1,rate plan - input check 01
display(
    df_model_raw
    .groupBy("plan_share_name")
    .agg(f.count("*").alias("cnt"))
    .orderBy(f.desc("cnt"))
)

# COMMAND ----------

# DBTITLE 1,rate plan - input check 02
display(
    df_model_raw
    .groupBy("plan_name_std")
    .agg(f.count("*").alias("cnt"))
    .orderBy(f.desc("cnt"))
)

# COMMAND ----------

# DBTITLE 1,rate plan - transform
df_model_prep = (
    df_model_prep
    .withColumn("plan_name_std_lump", f.col("plan_name_std"))
    .withColumn(
        "plan_family_derived"
        , f.when(
            (f.col("plan_share_flag") == 'Y')
            & (f.col("plan_share_relation_type") == "child")
            , f.col("plan_share_type")
        )
        .otherwise(f.col("plan_family"))
    )
)

df_model_prep = spkdf_lump_factor(
    df = df_model_prep
    , vt_target_col = "plan_name_std_lump"
    , ls_group_cols = ["plan_family_derived"]
    , vt_n_levels = 5
    , vt_other_level = "others"
)

df_model_prep = (
    df_model_prep
    .withColumn(
        "plan_name_std"
        , f.when(
            f.col("plan_family") == "endless data"
            , f.col("plan_name_std")
        ).otherwise(f.col("plan_name_std_lump"))
    )
    .drop("plan_name_std_lump", "pseudo_group")
)

# COMMAND ----------

# DBTITLE 1,rate plan - output check
display(
    df_model_prep
    .groupBy("plan_family", "plan_name_std")
    .agg(f.countDistinct("fs_srvc_id").alias("cnt"))
    .orderBy("plan_family", f.desc("cnt"))
)

# COMMAND ----------

# DBTITLE 1,rate plan - output check 02
check_distinct_values(
    df_model_prep
    , ["plan_family", "plan_name_std"]
    , ls_param_fs_reporting_keys
)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### network device brand

# COMMAND ----------

# DBTITLE 1,input check 01
display(
    df_model_raw
    .groupBy("network_dvc_brand")
    .agg(f.count("*").alias("cnt"))
    .orderBy(f.desc("cnt"))
)

# COMMAND ----------

# DBTITLE 1,transform
df_model_prep = (
    df_model_prep
    .withColumn("network_dvc_brand_lump", f.col("network_dvc_brand"))
)

df_model_prep = spkdf_lump_factor(
    df = df_model_prep
    , vt_target_col = "network_dvc_brand_lump"
    , ls_group_cols = None
    , vt_n_levels = 5
    , vt_other_level = "others"
)

df_model_prep = (
    df_model_prep
    .withColumn(
        "network_dvc_brand"
        , f.lower(f.col("network_dvc_brand_lump"))
    )
    .drop("network_dvc_brand_lump", "pseudo_group")
)

# COMMAND ----------

# DBTITLE 1,output check 01
display(
    df_model_prep
    .groupBy("network_dvc_brand")
    .agg(f.count("*").alias("cnt"))
    .orderBy(f.desc("cnt"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### network device model

# COMMAND ----------

# DBTITLE 1,transform 02
df_model_prep = (
    df_model_prep
    .withColumn("network_dvc_model_marketing_lump", f.col("network_dvc_model_marketing"))
)

df_model_prep = spkdf_lump_factor(
    df = df_model_prep
    , vt_target_col = "network_dvc_model_marketing_lump"
    , ls_group_cols = ["network_dvc_brand"]
    , vt_n_levels = 10
    , vt_other_level = "others"
)

df_model_prep = (
    df_model_prep
    #.withColumn(
    #    "network_dvc_model_marketing"
    #    , f.when(
    #        f.lower(f.col("network_dvc_brand")).isin(["apple"])
    #        , f.lower(f.col("network_dvc_model_marketing"))
    #    ).otherwise(f.lower(f.col("network_dvc_model_marketing_lump")))
    #)
    .withColumn(
        "network_dvc_model_marketing"
        , f.when(
            f.lower(f.col("network_dvc_brand")).isin(["others"])
            , f.lit("others")
        ).otherwise(f.lower(f.col("network_dvc_model_marketing_lump")))
    )
    .drop("network_dvc_model_marketing_lump", "pseudo_group")
)

# COMMAND ----------

# DBTITLE 1,output check 01
display(
    df_model_prep
    .groupBy("network_dvc_brand", "network_dvc_model_marketing")
    .agg(f.countDistinct("fs_srvc_id").alias("cnt"))
    .withColumn(
        "rank"
        , f.row_number().over(
            Window
            .partitionBy("network_dvc_brand")
            .orderBy(f.desc("cnt"))
        )
    )
    .withColumn(
        "rank"
        , f.when(f.col("network_dvc_model_marketing") == 'others', f.lit(99)).otherwise(f.col("rank"))
    )
    .orderBy("network_dvc_brand", "rank", f.desc("cnt"))
)

# COMMAND ----------

# DBTITLE 1,output check 02
check_distinct_values(
    df_model_prep
    , ["network_dvc_brand", "network_dvc_model_marketing"]
    , ls_param_fs_reporting_keys
)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### network device model version

# COMMAND ----------

# DBTITLE 1,transform
df_model_prep = (
    df_model_prep
    .withColumn("network_dvc_model_version_lump", f.col("network_dvc_model_version"))
)

df_model_prep = spkdf_lump_factor(
    df = df_model_prep
    , vt_target_col = "network_dvc_model_version_lump"
    , ls_group_cols = ["network_dvc_model_marketing"]
    , vt_n_levels = 5
    , vt_other_level = "others"
)

df_model_prep = (
    df_model_prep
    .withColumn(
        "network_dvc_model_version"
        , f.when(
            f.lower(f.col("network_dvc_model_marketing")).isin(["others"])
            , f.lit("others")
        ).otherwise(f.lower(f.col("network_dvc_model_version_lump")))
    )
    .drop("network_dvc_model_version_lump", "pseudo_group")
)

# COMMAND ----------

# DBTITLE 1,output check
display(
    df_model_prep
    .groupBy("network_dvc_brand", "network_dvc_model_marketing", "network_dvc_model_version")
    .agg(f.countDistinct("fs_srvc_id").alias("cnt"))
    .withColumn(
        "rank"
        , f.row_number().over(
            Window
            .partitionBy("network_dvc_brand")
            .orderBy(f.desc("cnt"))
        )
    )
    .withColumn(
        "rank"
        , f.when(f.col("network_dvc_model_marketing") == 'others', f.lit(99)).otherwise(f.col("rank"))
    )
    .orderBy("network_dvc_brand", "rank", f.desc("cnt"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### service sales channel

# COMMAND ----------

# DBTITLE 1,transform 01
df_model_prep = (
    df_model_prep
    .withColumn("service_sales_channel_group_lump", f.col("service_sales_channel_group"))
)

df_model_prep = spkdf_lump_factor(
    df = df_model_prep
    , vt_target_col = "service_sales_channel_group_lump"
    , ls_group_cols = None
    , vt_n_levels = 5
    , vt_other_level = "others"
)

df_model_prep = (
    df_model_prep
    .withColumn(
        "service_sales_channel_group"
        , f.lower(f.col("service_sales_channel_group_lump"))
    )
    .drop("service_sales_channel_group_lump", "pseudo_group")
)

# COMMAND ----------

# DBTITLE 1,transform 02
df_model_prep = (
    df_model_prep
    .withColumn(
        "service_sales_channel_nl_flag"
        , f.when(
            f.lower(f.col("service_sales_channel")).rlike("noel leeming")
            , f.lit('Y')
        ).otherwise(f.lit('N'))
    )
    .withColumn(
        "service_sales_channel_hn_flag"
        , f.when(
            f.lower(f.col("service_sales_channel")).rlike("harvey norman")
            , f.lit('Y')
        ).otherwise(f.lit('N'))
    )
    .withColumn("service_sales_channel_lump", f.col("service_sales_channel"))
)

df_model_prep = spkdf_lump_factor(
    df = df_model_prep
    , vt_target_col = "service_sales_channel_lump"
    , ls_group_cols = ["service_sales_channel_group"]
    , vt_n_levels = 5
    , vt_other_level = "others"
)

df_model_prep = (
    df_model_prep
    #.withColumn(
    #    "service_sales_channel"
    #    , f.when(
    #        f.lower(f.col("service_sales_channel_group")).isin(["others"])
    #        , f.lit("others")
    #    ).otherwise(f.lower(f.col("service_sales_channel_lump")))
    #)
    .withColumn(
        "service_sales_channel"
        , f.lower(f.col("service_sales_channel_lump"))
    )
    .drop("service_sales_channel_lump", "pseudo_group")
)

# COMMAND ----------

# DBTITLE 1,output check
display(
    df_model_prep
    .groupBy("service_sales_channel_group", "service_sales_channel")
    .agg(f.countDistinct("fs_srvc_id").alias("cnt"))
    .withColumn(
        "rank"
        , f.row_number().over(
            Window
            .partitionBy("service_sales_channel_group")
            .orderBy(f.desc("cnt"))
        )
    )
    .withColumn(
        "rank"
        , f.when(f.col("service_sales_channel") == 'others', f.lit(99)).otherwise(f.col("rank"))
    )
    .orderBy("service_sales_channel_group", "rank", f.desc("cnt"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### ifp primary sales channel

# COMMAND ----------

# DBTITLE 1,transform 01
df_model_prep = (
    df_model_prep
    .withColumn("ifp_prm_dvc_sales_channel_group_lump", f.col("ifp_prm_dvc_sales_channel_group"))
)

df_model_prep = spkdf_lump_factor(
    df = df_model_prep
    , vt_target_col = "ifp_prm_dvc_sales_channel_group_lump"
    , ls_group_cols = None
    , vt_n_levels = 5
    , vt_other_level = "others"
)

df_model_prep = (
    df_model_prep
    .withColumn(
        "ifp_prm_dvc_sales_channel_group"
        , f.lower(f.col("ifp_prm_dvc_sales_channel_group_lump"))
    )
    .drop("ifp_prm_dvc_sales_channel_group_lump", "pseudo_group")
)

# COMMAND ----------

# DBTITLE 1,transform 02
df_model_prep = (
    df_model_prep
    .withColumn(
        "ifp_prm_dvc_sales_channel_nl_flag"
        , f.when(
            f.lower(f.col("ifp_prm_dvc_sales_channel")).rlike("noel leeming")
            , f.lit('Y')
        ).otherwise(f.lit('N'))
    )
    .withColumn(
        "ifp_prm_dvc_sales_channel_hn_flag"
        , f.when(
            f.lower(f.col("ifp_prm_dvc_sales_channel")).rlike("harvey norman")
            , f.lit('Y')
        ).otherwise(f.lit('N'))
    )
    .withColumn("ifp_prm_dvc_sales_channel_lump", f.col("ifp_prm_dvc_sales_channel"))
)

df_model_prep = spkdf_lump_factor(
    df = df_model_prep
    , vt_target_col = "ifp_prm_dvc_sales_channel_lump"
    , ls_group_cols = ["ifp_prm_dvc_sales_channel_group"]
    , vt_n_levels = 5
    , vt_other_level = "others"
)

df_model_prep = (
    df_model_prep

    .withColumn(
        "ifp_prm_dvc_sales_channel"
        , f.lower(f.col("ifp_prm_dvc_sales_channel_lump"))
    )
    .drop("ifp_prm_dvc_sales_channel_lump", "pseudo_group")
)

# COMMAND ----------

# DBTITLE 1,output check
display(
    df_model_prep
    .groupBy("ifp_prm_dvc_sales_channel_group", "ifp_prm_dvc_sales_channel")
    .agg(f.countDistinct("fs_srvc_id").alias("cnt"))
    .withColumn(
        "rank"
        , f.row_number().over(
            Window
            .partitionBy("ifp_prm_dvc_sales_channel_group")
            .orderBy(f.desc("cnt"))
        )
    )
    .withColumn(
        "rank"
        , f.when(f.col("ifp_prm_dvc_sales_channel") == 'others', f.lit(99)).otherwise(f.col("rank"))
    )
    .orderBy("ifp_prm_dvc_sales_channel_group", "rank", f.desc("cnt"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### seasonality

# COMMAND ----------

# DBTITLE 1,transform
# seasonality flags
df_model_prep = (
    df_model_prep
    .withColumn("reporting_mnth", f.month("reporting_date").cast("string"))
    .withColumn("reporting_year", f.year('reporting_date'))
    .withColumn(
        "xmas_flag"
        , f.when(
            f.col("reporting_mnth").isin("11", "12")
            , f.lit("Y")
        ).otherwise(f.lit("N"))
    )
    .withColumn(
        "lockdown_flag"
        , f.when(
            f.col("reporting_date").isin("2021-08-31")
            , f.lit("Y")
        ).otherwise(f.lit("N"))
    )
    .withColumn(
        "lockdown_flag_a1"
        , f.when(
            f.col("reporting_date").isin("2021-09-30")
            , f.lit("Y")
        ).otherwise(f.lit("N"))
    )
    .withColumn(
        "lockdown_flag_a2"
        , f.when(
            f.col("reporting_date").isin("2021-10-31")
            , f.lit("Y")
        ).otherwise(f.lit("N"))
    )
    
    .withColumn(
        "iphone_release_flag"
        , f.when(
            f.col("reporting_date").isin("2021-09-30", "2022-09-30", '2023-09-30', '2024-09-30')
            , f.lit("Y")
        ).otherwise(f.lit("N"))
    )
    .withColumn(
        "iphone_release_flag_p1"
        , f.when(
            f.col("reporting_date").isin("2021-08-31", "2022-08-31", '2023-08-31', '2024-08-31')
            , f.lit("Y")
        ).otherwise(f.lit("N"))
    )
    .withColumn(
        "iphone_release_flag_p2"
        , f.when(
            f.col("reporting_date").isin("2021-07-31", "2022-07-31", '2023-07-31', '2024-07-31')
            , f.lit("Y")
        ).otherwise(f.lit("N"))
    )
    .withColumn(
        "iphone_release_flag_a1"
        , f.when(
            f.col("reporting_date").isin("2021-10-31", "2022-10-31", '2023-10-31', '2024-10-31')
            , f.lit("Y")
        ).otherwise(f.lit("N"))
    )
    .withColumn(
        "iphone_release_flag_a2"
        , f.when(
            f.col("reporting_date").isin("2021-11-30", "2022-11-30", "2023-11-30", '2024-11-30')
            , f.lit("Y")
        ).otherwise(f.lit("N"))
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### discount removal 

# COMMAND ----------

# DBTITLE 1,discount removal comms query
# via email 
query_sfmc_email = """ 
    select * from PROD_MAR_TECH.SERVING.SFMC_EMAIL_PERFORMANCE
    where campaignname in (
       '240827-RM-FIX-Converged-Discount-Removal-Email-Scale 2210'
        ,'240827-RM-FIX-Converged-Discount-Removal-Email-Scale 2210-Queued'
        ,'240827-RM-FIX-Converged-Discount-Removal-Email-Cohort B-0210'
        ,'240827-RM-FIX-Converged-Discount-Removal-Email-Cohort B'
        ,'240827-RM-FIX-Converged-Discount-Removal-Email-Oreo Customers'
        ,'240827-RM-FIX-Converged-Discount-Removal-Email-Scale Ineligible DOM 8-11'
        ,'240827-RM-FIX-Converged-Discount-Removal-Email-Scale 3110'
        ,'240827-RM-FIX-Converged-Discount-Removal-Email-Scale Ineligible DOM 13-16-18'
        ,'240827-RM-FIX-Converged-Discount-Removal-Email-Scale Ineligible DOM 26-28-29'
        ,'240827-RM-FIX-Converged-Discount-Removal-Email-Cohort A-0210'
        ,'240827-RM-FIX-Converged-Discount-Removal-Email-Cohort A'
        ,'240827-RM-FIX-Converged-Discount-Removal-Email-Scale Ineligible DOM 20-22-24 (v2)'
        ,'240827-RM-FIX-Converged-Discount-Removal-Email-Scale 1710'
        , '240501-DH-MOBPM-Project_Oreo_Batch_Three Day Two'
        , '240501-DH-MOBPM-Project_Oreo_Batch_Four'
        , '240501-DH-MOBPM-Project_Oreo_Batch_Three_Email_Washup'
        , '240501-DH-MOBPM-Project_Oreo_Batch_Five V2'
        ,  '240501-DH-MOBPM-Project_Oreo_Batch_Six'
        ,'240501-DH-MOBPM-Project_Oreo-WashUp'
        ,'240501-DH-MOBPM-Project_Oreo_Batch_Six' 
        ,'240501-DH-MOBPM-Project_Oreo_Batch_Five'
        ,'240501-DH-MOBPM-Project_Oreo_Batch_Three_Email_Washup'
        , '240501-DH-MOBPM-Project_Oreo_Batch_Four'
        ,'240501-DH-MOBPM-Project_Oreo_Batch_Three Day Two'
        ,'240501-DH-MOBPM-Project_Oreo_Batch_Three'
        ,'240501-DH-MOBPM-Project_Oreo_Send_Version_Three'
        ,'240501-DH-MOBPM-Project_Oreo_Send'
    )   
""" 

# via sms 
query_sfmc_sms = """
    select * from PROD_MAR_TECH.SERVING.SFMC_ON_NET_SMS_MESSAGE 
    where sms_name ilike '%converged%'
"""

# read in spark 
df_campaign_list_e = (
    spark.read.format("snowflake")
    .options(**options)
    .option(
        "query"
        , query_sfmc_email
    )
    .load()
)

# read in spark 
df_campaign_list_s = (
    spark.read.format("snowflake")
    .options(**options)
    .option(
        "query"
        , query_sfmc_sms
    )
    .load()
)


# union to get per customers min comms date and max comms date and count of comms 
df_comm_base_agg = (
    df_campaign_list_e
    .select('customer_id', 'EVENTDATE', 'EMAILNAME')
    .distinct()
    .union(
        df_campaign_list_s
        .select('customer_id', 'SEND_DATE', 'SMS_NAME')
    )
    .distinct()
    .withColumnRenamed('customer_id', 'fs_cust_id')
    .groupBy('fs_cust_id')
    .agg(f.countDistinct('EMAILNAME').alias('comm_cnt')
         , f.min('eventdate').alias('min_event_date')
         , f.max('eventdate').alias('max_event_date')
    )
)

# COMMAND ----------

# DBTITLE 1,check comms base
display(
    df_comm_base_agg
    .limit(100)
)

display(
    df_comm_base_agg
    .agg(
        f.countDistinct('fs_cust_id')
        , f.count('*')
    )
)

# COMMAND ----------

# DBTITLE 1,bb base

# ls all dates that used for training and holdout and blend 
ls_all_dates = (
    ls_param_mlf_model_data_training_date + 
    ls_param_mlf_model_data_blend_date + 
    ls_param_mlf_model_data_holdout_date
)

# Calculate training and holdout min and max
vt_reporting_date_min = min(ls_all_dates)
vt_reporting_date_max = max(ls_all_dates)

# get calendar dates  in a list 
list_of_month_ends = pd.date_range(
    start=vt_reporting_date_min, 
    end=vt_reporting_date_max, 
    freq='M'
).strftime('%Y%m%d').tolist()

# Convert the list into a string formatted for SQL
formatted_dates = ', '.join([f"'{date}'" for date in list_of_month_ends])

# query corresponding bb base 

query_bb_base = f"""
select 
    d_snapshot_date_key 
    , TO_DATE(TO_VARCHAR(d_snapshot_date_key), 'YYYYMMDD') as bb_reporting_date
    ,s.service_id as bb_fs_serv_id
    , billing_account_number as bb_fs_acct_id
    , c.customer_source_id as bb_fs_cust_id
    , service_access_type_name
    , s.proposition_product_name
    , s.plan_name
    , s.broadband_discount_oa_msisdn as converged_oa_msisdn
from prod_pdb_masked.modelled.f_service_activity_daily_snapshot f 
inner join prod_pdb_masked.modelled.d_service_curr s 
on f.service_source_id = s.service_source_id 
and s.current_record_ind = 1 
inner join prod_pdb_masked.modelled.d_billing_account_curr b 
on b.billing_account_source_id = s.billing_account_source_id 
and b.current_record_ind = 1
inner join prod_pdb_masked.modelled.d_customer_curr c 
on c.customer_source_id = b.customer_source_id
and c.current_record_ind = 1
where  
    s.service_type_name in ('Broadband')
    and 
    ( 
    (service_access_type_name is not null)
    
    or 
    ( s.service_type_name in ('Broadband')
    and service_access_type_name is null   
    and s.proposition_product_name in ('home phone wireless broadband discount proposition'
    , 'home phone plus broadband discount proposition')
    )
    )
    and c.market_segment_name in ('Consumer')
    and f.d_snapshot_date_key in ({formatted_dates})
"""

# load bb base into spark 

df_bb_base = (
    spark.read.format("snowflake")
    .options(**options)
    .option(
        "query"
        , query_bb_base
    )
    .load()
)

# COMMAND ----------

# DBTITLE 1,check bb base
display(
    df_bb_base 
    .groupBy('bb_reporting_date')
    .agg(
        f.countDistinct('BB_FS_ACCT_ID')
        , f.countDistinct('BB_FS_SERV_ID')
    )
)

# COMMAND ----------

# DBTITLE 1,transform
# plan discount removal and converged discount flags

df_model_prep_test =  (
    df_model_prep
    .join(
        df_bb_base
        .select('bb_reporting_date', 'BB_FS_ACCT_ID')
        .distinct()
        , (f.col('reporting_date') == f.col('bb_reporting_date'))
        & (f.col('fs_acct_id') == f.col('BB_FS_ACCT_ID'))
        , 'left' 
        )
    .withColumn(
        'converged_acct_flag'
        , f.when(
            f.col('bb_fs_acct_id').isNotNull()
            , f.lit('Y')
        )
        .otherwise('N')
    )
    .join(
        df_comm_base_agg.alias('comm')
        , ['fs_cust_id']
        , 'left'
    )
    .withColumn(
        'disc_rm_comms_flag'
        , f.when(
            f.col('comm_cnt').isNotNull()
            , f.lit('Y')
        )
        .otherwise(f.lit('N'))    
    )
    .drop('bb_reporting_date', 'bb_fs_acct_id')
   #.limit(10)
)

# COMMAND ----------

# DBTITLE 1,check data count
display(
    df_model_prep_test
    .groupby('reporting_date', 'converged_acct_flag', 'disc_rm_comms_flag')
    .agg(
        f.countDistinct('fs_acct_id')
        , f.countDistinct('fs_srvc_id')
        , f.count('*')
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### finalisation

# COMMAND ----------

# DBTITLE 1,transform
df_model = df_model_prep

display(df_model.limit(10))

# COMMAND ----------

# DBTITLE 1,output check - distinct
check_distinct_values(
    df_model
    , ls_param_model_features_char
    , ls_param_fs_reporting_keys
)

# COMMAND ----------

export_model_data(
    df_model
    , os.path.join(dir_data_mlf, f"model_full_{vt_param_mlf_model_data_version_id}")
    , ls_param_fs_reporting_keys
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### s404 model sets creation (sampling)
# MAGIC Stratified Sampling based on df_model_ssc_meta & target flag to generate:
# MAGIC * training set - model creation
# MAGIC * blending set - model optimisation for stack ensembling
# MAGIC * validation set - model selection
# MAGIC * calibration set - probability calibration
# MAGIC * holdout set - model evaluation

# COMMAND ----------

df_model = spark.read.format("delta").load(os.path.join(dir_data_mlf, f"model_full_{vt_param_mlf_model_data_version_id}"))

# COMMAND ----------

# DBTITLE 1,holdout & blending sets
# holdout set
df_model_holdout = (
    df_model
    .filter(f.col("reporting_date").isin(ls_param_mlf_model_data_holdout_date))
)

# blending set
df_model_blend = (
    df_model
    .filter(f.col("reporting_date").isin(ls_param_mlf_model_data_blend_date))
)

# COMMAND ----------

# DBTITLE 1,train/valid/calibrate sets
# train/validation/calibrate set
df_model_split = spkdf_initial_split(
    df_model
    .filter(~f.col("reporting_date").isin(ls_param_mlf_model_data_holdout_date + ls_param_mlf_model_data_blend_date))
    , vt_valid_pct = vt_param_mlf_model_data_valid_pct
    , ls_strata = ["reporting_date", "target_label"]
    , seed = 42
)

df_model_valid = (
    df_model_split
    .filter(f.col("split_label") == "valid")
    .drop("split_label")
)

df_model_train_calibrate = (
    df_model_split
    .filter(f.col("split_label") == "train")
    .drop("split_label")
)

# train/calibrate set
df_model_train_calibrate_split = spkdf_initial_split(
    df_model_train_calibrate
    , vt_valid_pct = vt_param_mlf_model_data_calibrate_pct
    , ls_strata = ["reporting_date", "target_label"]
    , seed = 43
)

df_model_calibrate = (
    df_model_train_calibrate_split
    .filter(f.col("split_label") == "valid")
    .drop("split_label")
)

df_model_train = (
    df_model_train_calibrate_split
    .filter(f.col("split_label") == "train")
    .drop("split_label")
)

# COMMAND ----------

# DBTITLE 1,output check
df_model_summary_valid = (
    df_model_valid
    .groupBy("reporting_date", "target_label")
    .agg(f.countDistinct("fs_srvc_id").alias("cnt"))
    .orderBy(f.desc("reporting_date"), f.desc("target_label"))
    .withColumn("type", f.lit("valid"))
)

df_model_summary_calibrate = (
    df_model_calibrate
    .groupBy("reporting_date", "target_label")
    .agg(f.countDistinct("fs_srvc_id").alias("cnt"))
    .orderBy(f.desc("reporting_date"), f.desc("target_label"))
    .withColumn("type", f.lit("calibrate"))
)

df_model_summary_train = (
    df_model_train
    .groupBy("reporting_date", "target_label")
    .agg(f.countDistinct("fs_srvc_id").alias("cnt"))
    .orderBy(f.desc("reporting_date"), f.desc("target_label"))
    .withColumn("type", f.lit("train"))
)

df_model_summary_blend = (
    df_model_blend
    .groupBy("reporting_date", "target_label")
    .agg(f.countDistinct("fs_srvc_id").alias("cnt"))
    .orderBy(f.desc("reporting_date"), f.desc("target_label"))
    .withColumn("type", f.lit("blend"))
)


df_model_summary_holdout = (
    df_model_holdout
    .groupBy("reporting_date", "target_label")
    .agg(f.countDistinct("fs_srvc_id").alias("cnt"))
    .orderBy(f.desc("reporting_date"), f.desc("target_label"))
    .withColumn("type", f.lit("holdout"))
)

display(
    df_model_summary_valid
    .union(df_model_summary_calibrate)
    .union(df_model_summary_train)
    .union(df_model_summary_blend)
    .union(df_model_summary_holdout)
    .withColumn("cnt_total", f.sum("cnt").over(Window.partitionBy("reporting_date", "target_label")))
    .withColumn("pct", f.col("cnt")/f.col("cnt_total"))
    .orderBy(f.desc("reporting_date"), f.desc("target_label"))
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### s4 data export

# COMMAND ----------

# DBTITLE 1,parameters registry 01
# create model data parameters meta
df_model_train_param = (
    df_model_train
    .agg(
        f.max("reporting_date").alias("train_date_max")
        , f.min("reporting_date").alias("train_date_min")
    )
)

vt_param_train_date_min = df_model_train_param.collect()[0].train_date_min
vt_param_train_date_max = df_model_train_param.collect()[0].train_date_max

df_model_calibrate_param = (
    df_model_calibrate
    .agg(
        f.max("reporting_date").alias("calibrate_date_max")
        , f.min("reporting_date").alias("calibrate_date_min")
    )
)

vt_param_calibrate_date_min = df_model_calibrate_param.collect()[0].calibrate_date_min
vt_param_calibrate_date_max = df_model_calibrate_param.collect()[0].calibrate_date_max


df_model_valid_param = (
    df_model_valid
    .agg(
        f.max("reporting_date").alias("valid_date_max")
        , f.min("reporting_date").alias("valid_date_min")
    )
)

vt_param_valid_date_min = df_model_valid_param.collect()[0].valid_date_min
vt_param_valid_date_max = df_model_valid_param.collect()[0].valid_date_max

df_model_blend_param = (
    df_model_blend
    .agg(
        f.max("reporting_date").alias("blend_date_max")
        , f.min("reporting_date").alias("blend_date_min")
    )
)

vt_param_blend_date_min = df_model_blend_param.collect()[0].blend_date_min
vt_param_blend_date_max = df_model_blend_param.collect()[0].blend_date_max

df_model_holdout_param = (
    df_model_holdout
    .agg(
        f.max("reporting_date").alias("holdout_date_max")
        , f.min("reporting_date").alias("holdout_date_min")
    )
)

vt_param_holdout_date_min = df_model_holdout_param.collect()[0].holdout_date_min
vt_param_holdout_date_max = df_model_holdout_param.collect()[0].holdout_date_max

# COMMAND ----------

# DBTITLE 1,parameters registry 02
df_model_data_params = pd.DataFrame(
    data = {
        "model_cycle_type": [vt_param_fs_reporting_cycle_type]
        , "model_freq_type": [vt_param_fs_reporting_freq_label]
        , "predict_days": [vt_param_mlf_model_data_predict_days]
        , "data_version": [vt_param_mlf_model_data_version_id]
        , "train_set_date_min": [vt_param_train_date_min]
        , "train_set_date_max": [vt_param_train_date_max]
        , "valid_set_sample_pct": [vt_param_mlf_model_data_valid_pct]
        , "valid_set_date_min": [vt_param_valid_date_min]
        , "valid_set_date_max": [vt_param_valid_date_max]
        , "calibrate_set_sample_pct": [vt_param_mlf_model_data_calibrate_pct]
        , "calibrate_set_date_min": [vt_param_calibrate_date_min]
        , "calibrate_set_date_max": [vt_param_calibrate_date_max]
        
        , "blend_set_date_min": [vt_param_blend_date_min]
        , "blend_set_date_max": [vt_param_blend_date_max]
        , "holdout_set_date_min": [vt_param_holdout_date_min]
        , "holdout_set_date_max": [vt_param_holdout_date_max]
    }
)

df_model_data_params = spark.createDataFrame(df_model_data_params)
display(df_model_data_params)

# COMMAND ----------

# DBTITLE 1,data export 01
# export data for model training in the next step
(
    df_model_data_params
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "True")
    .save(os.path.join(dir_data_mlf, f"model_data_params_{vt_param_mlf_model_data_version_id}"))
)

# COMMAND ----------

# DBTITLE 1,data export 02
export_model_data(
    df_model_train
    , os.path.join(dir_data_mlf, f"model_train_{vt_param_mlf_model_data_version_id}")
    , ls_param_fs_reporting_keys
)

export_model_data(
    df_model_valid
    , os.path.join(dir_data_mlf, f"model_valid_{vt_param_mlf_model_data_version_id}")
    , ls_param_fs_reporting_keys
)

export_model_data(
    df_model_calibrate
    , os.path.join(dir_data_mlf, f"model_calibrate_{vt_param_mlf_model_data_version_id}")
    , ls_param_fs_reporting_keys
)

export_model_data(
    df_model_blend
    , os.path.join(dir_data_mlf, f"model_blend_{vt_param_mlf_model_data_version_id}")
    , ls_param_fs_reporting_keys
)

export_model_data(
    df_model_holdout
    , os.path.join(dir_data_mlf, f"model_holdout_{vt_param_mlf_model_data_version_id}")
    , ls_param_fs_reporting_keys
)
