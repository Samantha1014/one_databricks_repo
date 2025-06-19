# Databricks notebook source
# MAGIC %run "./s98_environment_setup"

# COMMAND ----------

import os
import pyspark

from pyspark import sql
from pyspark.sql import Window
from pyspark.sql import functions as f

import pandas as pd

# COMMAND ----------

# MAGIC %run "./s99_utility_functions"

# COMMAND ----------

# MAGIC  %run "./qa_utils"

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


vt_param_fs_reporting_cycle_type = "calendar cycle"
vt_param_fs_reporting_freq_type = "proc_freq_monthly_flag"
vt_param_fs_reporting_freq_label = "monthly"
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
#vt_param_mlf_model_data_start_date = '2021-09-01'
#vt_param_mlf_model_data_end_date = "2023-11-30"

vt_param_mlf_model_data_start_date = '2024-10-31'
vt_param_mlf_model_data_end_date = "2024-10-31"

vt_param_mlf_model_data_version_id = "v1"
vt_param_mlf_model_data_predict_days = 90
vt_param_mlf_model_data_predict_date_from_base_date = 1
vt_param_mlf_model_data_valid_pct = 0.1
vt_param_mlf_model_data_calibrate_pct = 0.1
#ls_param_mlf_model_data_holdout_date = ["2023-09-30", "2023-10-31", "2023-11-30"]
#ls_param_mlf_model_data_blend_date = ["2023-08-31", "2023-07-31"]

# ls_param_mlf_model_data_training_date = (
#     pd
#     .date_range(start = '2023-01-31', end = '2023-07-31', freq = 'M')
#     .tolist()
# )

# ls_param_mlf_model_data_blend_date = (
#     pd
#     .date_range(start = '2023-08-31', end = '2023-9-30', freq = 'M')
#     .tolist()
# )

ls_param_mlf_model_data_holdout_date = (
     pd
     .date_range(start = '2024-10-31', end = '2024-10-31', freq = 'M')
     .tolist()
)

# COMMAND ----------

ls_param_mlf_model_data_holdout_date

# COMMAND ----------

# data directories
# feature store
dir_data_fs_meta = os.path.join(dir_data_fs_parent, "d000_meta")
dir_data_fs_fea = os.path.join(dir_data_fs_parent, "d400_feature")
# dir_data_fs_mvmt = os.path.join(dir_data_fs_parent, "d500_movement")
dir_data_fs_serv = os.path.join(dir_data_fs_parent, "d600_serving")

# customize addon 
dir_data_fs_mvnt = '/mnt/ml-lab/dev_users/dev_sc/aod30_mvnt_acct_agg'
dir_data_fs_aod30 = '/mnt/ml-lab/dev_users/dev_sc/99_misc/df_aod_stauts_v3'

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

# MAGIC %md
# MAGIC ## S3.5 data integration

# COMMAND ----------

# MAGIC %md
# MAGIC ### S4 data preparation

# COMMAND ----------

# MAGIC %md
# MAGIC ### s401 model meta data creation

# COMMAND ----------

print(vt_param_fs_reporting_cycle_type)
print(vt_param_fs_reporting_freq_type)
print(vt_param_mlf_model_data_start_date)
print(vt_param_mlf_model_data_end_date)

# COMMAND ----------

# DBTITLE 1,model meta
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
        )
        # ).when(
        #     f.col("base_date").isin(ls_param_mlf_model_data_blend_date)
        #     , "blending"
        .otherwise("others")
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

# DBTITLE 1,interpolation
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
# MAGIC model data creation 

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
# MAGIC ### s402 model unit base creation

# COMMAND ----------

# DBTITLE 1,model unit base
# create model unit base from the feature store 
df_model_unit_base = (df_fs_serv
        .join(df_fs_aod30_status, ls_param_fs_primary_keys+ls_param_fs_reporting_keys, 
              'left'
            )
        .filter(f.col('aod30_status').isNull())
        .filter(f.col('reporting_date').between(vt_param_mlf_model_data_start_date, vt_param_mlf_model_data_end_date))
        .filter(f.col('reporting_cycle_type') == 'calendar cycle')
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
# MAGIC ### s403 model target creation

# COMMAND ----------

display(df_fs_mvmt_acct_aod30
        .groupBy('reporting_date', 'reporting_cycle_type')
        .agg(f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_cust_id')
             , f.count('*')
             )
)


# COMMAND ----------

ls_param_fs_reporting_keys


# COMMAND ----------

# aod 30d movement
df_base_mvmt_srvc_aod30 = (
    df_fs_mvmt_acct_aod30
    .withColumnRenamed("movement_date", "target_date")  
    .select('fs_acct_id', 'target_date')
    .join(
        df_model_target_ssc_meta
        , ["target_date"]
        , "inner"
    )
    .join(
        df_model_unit_base
        #, ls_param_fs_reporting_keys + ls_param_fs_primary_keys 
        , ls_param_fs_reporting_keys + ["fs_acct_id"]
        , "inner"
    )
    .select(
        *ls_param_fs_reporting_keys
        , *ls_param_fs_primary_keys
        , "target_date"
    )
    .withColumn('rank', 
                f.row_number().over(
                  Window
                  .partitionBy(*ls_param_fs_primary_keys, *ls_param_fs_reporting_keys)
                  .orderBy(f.asc('target_date'))
                  )
                )
    .filter(f.col('rank') == 1)
    .drop('rank')
    .select(
        *ls_param_fs_reporting_keys
        , *ls_param_fs_primary_keys
        , "target_date"
    )
)

# COMMAND ----------

# DBTITLE 1,data check
print("sample check")
display(df_base_mvmt_srvc_aod30.limit(10))

print("summary check")
display(
    df_base_mvmt_srvc_aod30
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

# DBTITLE 1,model target creation
df_model_target = (
    df_model_unit_base
    .join(
        df_base_mvmt_srvc_aod30
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

# MAGIC %md ### s404 model feature creation
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Model Data Creation
# MAGIC - create model feature table with selected features
# MAGIC - left join model feature table to model traget table based on reporting keys & primary keys
# MAGIC - model specific data transformation

# COMMAND ----------

# MAGIC %md
# MAGIC #### data check

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



# COMMAND ----------

# DBTITLE 1,data check
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

# COMMAND ----------

# DBTITLE 1,data check
display(df_model_raw.limit(10))

# COMMAND ----------

# DBTITLE 1,check schema
df_model_raw_schema = get_spkdf_schema(df_model_raw)

display(df_model_raw_schema)

# COMMAND ----------

display(df_model_raw
          .filter(f.col('reporting_date').between(vt_param_mlf_model_data_start_date, vt_param_mlf_model_data_end_date))
          .filter(f.col('reporting_cycle_type') == 'calendar cycle')
          .groupBy('reporting_date')
          .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_srvc_id')
             )
        )

# COMMAND ----------

# MAGIC %md
# MAGIC Model Specific Transform Steps:
# MAGIC
# MAGIC 1. lump factors for plan_nm_std
# MAGIC 2. lump device model
# MAGIC 3. create seasonality flags:
# MAGIC     - xmas flag
# MAGIC     - lockdown flags
# MAGIC     - iphone launch flags

# COMMAND ----------

# MAGIC %md #### initial

# COMMAND ----------

df_model_prep = df_model_raw

# COMMAND ----------

# DBTITLE 1,lump plan
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

# DBTITLE 1,lump network device
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

# MAGIC %md
# MAGIC ### network device manufacturer

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

# MAGIC %md
# MAGIC ### network device model

# COMMAND ----------

# DBTITLE 1,process
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

# MAGIC
# MAGIC %md #### network device model version

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

# DBTITLE 1,seasonality
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

# MAGIC %md
# MAGIC ### finalization

# COMMAND ----------

df_model = df_model_prep

#display(df_model.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### export data

# COMMAND ----------

df_model_holdout = (
    df_model
    .filter(f.col("reporting_date").isin(ls_param_mlf_model_data_holdout_date))
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

display(df_test
        .groupBy('reporting_date')
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_srvc_id')
             )
        )
