# Databricks notebook source
import os
import pyspark

from pyspark import sql
from pyspark.sql import Window
from pyspark.sql import functions as f

import pandas as pd

# COMMAND ----------

# MAGIC %run "./s99_utility_functions"

# COMMAND ----------

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

# feature store parameters
dir_data_fs_parent = "/mnt/feature-store-prod-lab"

vt_param_fs_reporting_cycle_type = "rolling cycle"
vt_param_fs_reporting_freq_type = "proc_freq_monthly_flag"
#vt_param_fs_reporting_freq_label = "monthly"
ls_param_fs_reporting_keys = ["reporting_cycle_type", "reporting_date"] 
ls_param_fs_primary_keys = ["fs_cust_id", "fs_acct_id", "fs_srvc_id"]

# ml factory parameters
# generic
vt_param_mlf_model_pattern = "classification"

vt_param_mlf_user_id = "dev_sc"
dir_data_mlf_parent = "/mnt/ml-factory-gen-x-dev"

# model
vt_param_mlf_model_id = "mobile_oa_consumer_srvc_aod30d_pred90d"

# experiment
vt_param_mlf_exp_id = "mobile_oa_consumer_srvc_aod30d_pred90d_202409_exp2"

# model data
vt_param_mlf_model_feature_meta = "LAB_ML_STORE.SANDBOX.MLF_FEA_META_MOBILE_OA_CONSUMER_SRVC_WRITEOFF_PRED120D_202404_EXP1" 

vt_param_mlf_model_data_version_id = "v1"
vt_param_mlf_model_data_predict_days = 90
vt_param_mlf_model_data_predict_date_from_base_date = 1
vt_param_mlf_model_data_valid_pct = 0.1
vt_param_mlf_model_data_calibrate_pct = 0.1

#vt_param_ssc_reporting_date = '2024-11-03'


# COMMAND ----------

# data directories
# feature store
dir_data_fs_meta = os.path.join(dir_data_fs_parent, "d000_meta")
dir_data_fs_fea = os.path.join(dir_data_fs_parent, "d400_feature")
# dir_data_fs_mvmt = os.path.join(dir_data_fs_parent, "d500_movement")
dir_data_fs_serv = os.path.join(dir_data_fs_parent, "d600_serving")

# customize addon 
dir_data_fs_mvnt = '/mnt/ml-lab/dev_users/dev_sc/aod30_mvnt_acct_agg_curr'
dir_data_fs_aod30 = '/mnt/ml-lab/dev_users/dev_sc/df_aod_stauts_curr'

# ml factory
dir_data_mlf_exp = os.path.join(dir_data_mlf_parent, "dev_users", vt_param_mlf_user_id, vt_param_mlf_model_pattern, vt_param_mlf_model_id, vt_param_mlf_exp_id)

dir_data_mlf = dir_data_mlf_exp

# COMMAND ----------

df_fs_global_cycle_calendar = spark.read.format("delta").load(os.path.join(dir_data_fs_meta, "d001_global_cycle_calendar"))
df_fsr_dict_meta = spark.read.format("delta").load(os.path.join(dir_data_fs_meta, "d004_fsr_meta/fsr_dict_meta"))
df_fs_serv = spark.read.format("delta").load(os.path.join(dir_data_fs_serv, "serv_mobile_oa_consumer"))
df_fs_mvmt_acct_aod30 = spark.read.format("delta").load(dir_data_fs_mvnt)
df_fs_aod30_status = spark.read.format('delta').load(dir_data_fs_aod30)

# COMMAND ----------



# COMMAND ----------

display(
        df_fs_serv
        .agg(f.max('reporting_date'))
)

# COMMAND ----------

vt_param_ssc_reporting_date = (df_fs_mvmt_acct_aod30
        .agg(f.max('reporting_date'))
        .collect()[0][0]
        )
        

# COMMAND ----------

vt_param_ssc_reporting_date

# COMMAND ----------

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

print("global cycle calendar")
display(df_fs_global_cycle_calendar.limit(10))

print("fs serv")
display(df_fs_serv.limit(10))

print("acct aod")
display(df_fs_mvmt_acct_aod30.limit(10))

# COMMAND ----------

ls_param_model_features = pull_col(df_mlf_feature_meta, "feature")

df_model_features = (
     df_fs_serv
    .select(
        ls_param_fs_reporting_keys 
        + ls_param_fs_primary_keys 
        + ls_param_model_features
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### s402 model unit base creation

# COMMAND ----------

# create model unit base from the feature store 
df_model_unit_base = (df_fs_serv
        .join(df_fs_aod30_status, ls_param_fs_primary_keys+ls_param_fs_reporting_keys, 
              'left'
            )
        .filter(f.col('aod30_status').isNull())
        .filter(f.col('reporting_date') == vt_param_ssc_reporting_date )
        .filter(f.col('reporting_cycle_type') == vt_param_fs_reporting_cycle_type)
        .select(*ls_param_fs_reporting_keys, *ls_param_fs_primary_keys)
    )


# COMMAND ----------

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

# combine model features & target
df_model_raw = (
    df_model_unit_base
    .join(
        df_model_features
        , ls_param_fs_reporting_keys + ls_param_fs_primary_keys
        , "left"
    )
)

# COMMAND ----------

display(df_model_raw
          .filter(f.col('reporting_date') == vt_param_ssc_reporting_date)
          .filter(f.col('reporting_cycle_type') == vt_param_fs_reporting_cycle_type)
          .groupBy('reporting_date')
          .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_srvc_id')
             )
        )

# COMMAND ----------

df_model_prep = df_model_raw

# COMMAND ----------

df_model_prep = (
    df_model_prep
    .withColumn("plan_name_std_lump", f.col("plan_name_std"))
)

df_model_prep = spkdf_lump_factor(
    df = df_model_prep
    , vt_target_col = "plan_name_std_lump"
    , ls_group_cols = ["plan_family"]
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
    .drop("plan_name_std_lump")
)

# COMMAND ----------

df_model_prep = (
    df_model_prep
    .withColumn("plan_name_std_lump", f.col("plan_name_std"))
)

df_model_prep = spkdf_lump_factor(
    df = df_model_prep
    , vt_target_col = "plan_name_std_lump"
    , ls_group_cols = ["plan_family"]
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
    .drop("plan_name_std_lump")
)

# COMMAND ----------

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

df_model_prep = (
    df_model_prep
    .withColumn("network_dvc_manufacturer_lump", f.col("network_dvc_manufacturer"))
)

df_model_prep = spkdf_lump_factor(
    df = df_model_prep
    , vt_target_col = "network_dvc_manufacturer_lump"
    , ls_group_cols = None
    , vt_n_levels = 10
    , vt_other_level = "others"
)

df_model_prep = (
    df_model_prep
    .withColumn(
        "network_dvc_manufacturer"
        , f.lower(f.col("network_dvc_manufacturer_lump"))
    )
    .drop("network_dvc_manufacturer_lump", "pseudo_group")
)

# COMMAND ----------

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
    .drop("network_dvc_model_marketing_lump")
)

# COMMAND ----------

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
    #.withColumn(
    #    "network_dvc_model_marketing"
    #    , f.when(
    #        f.lower(f.col("network_dvc_brand")).isin(["apple"])
    #        , f.lower(f.col("network_dvc_model_marketing"))
    #    ).otherwise(f.lower(f.col("network_dvc_model_marketing_lump")))
    #)
    .withColumn(
        "network_dvc_model_version"
        , f.when(
            f.lower(f.col("network_dvc_model_marketing")).isin(["others"])
            , f.lit("others")
        ).otherwise(f.lower(f.col("network_dvc_model_version_lump")))
    )
    .drop("network_dvc_model_version_lump")
)

# COMMAND ----------

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
            f.col("reporting_date").isin("2021-09-30", "2022-09-30")
            , f.lit("Y")
        ).otherwise(f.lit("N"))
    )
    .withColumn(
        "iphone_release_flag_p1"
        , f.when(
            f.col("reporting_date").isin("2021-08-31", "2022-08-31")
            , f.lit("Y")
        ).otherwise(f.lit("N"))
    )
    .withColumn(
        "iphone_release_flag_p2"
        , f.when(
            f.col("reporting_date").isin("2021-07-31", "2022-07-31")
            , f.lit("Y")
        ).otherwise(f.lit("N"))
    )
    .withColumn(
        "iphone_release_flag_a1"
        , f.when(
            f.col("reporting_date").isin("2021-10-31", "2022-10-31")
            , f.lit("Y")
        ).otherwise(f.lit("N"))
    )
    .withColumn(
        "iphone_release_flag_a2"
        , f.when(
            f.col("reporting_date").isin("2021-11-30", "2022-11-30")
            , f.lit("Y")
        ).otherwise(f.lit("N"))
    )
)

# COMMAND ----------

df_model = df_model_prep

# COMMAND ----------

df_model_holdout = (
    df_model
    .filter(f.col("reporting_date") == vt_param_ssc_reporting_date)
    )

# COMMAND ----------

display(df_model_holdout
        .groupBy('reporting_date')
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_srvc_id')
             )
        )

# COMMAND ----------

export_model_data(
    df_model_holdout
    , '/mnt/ml-lab/dev_users/dev_sc/99_misc/df_model_test_v4'
    , ls_param_fs_reporting_keys
)

# COMMAND ----------

df_test = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/99_misc/df_model_test_v4')

# COMMAND ----------

display(df_test.limit(10))

display(df_test
        .groupBy('reporting_date')
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_srvc_id')
             )
        )
