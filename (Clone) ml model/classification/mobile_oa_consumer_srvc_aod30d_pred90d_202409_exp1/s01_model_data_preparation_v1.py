# Databricks notebook source
# MAGIC %md ## s1 environment setup

# COMMAND ----------

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

get_db_notebook_dir(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())

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

# MAGIC %md ## s2 parameters

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
vt_param_mlf_exp_id = "mobile_oa_consumer_srvc_aod30d_pred90d_202409_exp1"

# model data
vt_param_mlf_model_feature_meta = "LAB_ML_STORE.SANDBOX.MLF_FEA_META_MOBILE_OA_CONSUMER_SRVC_WRITEOFF_PRED120D_202404_EXP1" 
#vt_param_mlf_model_data_start_date = '2021-09-01'
#vt_param_mlf_model_data_end_date = "2023-11-30"

vt_param_mlf_model_data_start_date = '2023-01-31'
vt_param_mlf_model_data_end_date = "2023-12-31"

vt_param_mlf_model_data_version_id = "v1"
vt_param_mlf_model_data_predict_days = 90
vt_param_mlf_model_data_predict_date_from_base_date = 1
vt_param_mlf_model_data_valid_pct = 0.1
vt_param_mlf_model_data_calibrate_pct = 0.1
#ls_param_mlf_model_data_holdout_date = ["2023-09-30", "2023-10-31", "2023-11-30"]
#ls_param_mlf_model_data_blend_date = ["2023-08-31", "2023-07-31"]

ls_param_mlf_model_data_training_date = (
    pd
    .date_range(start = '2023-01-31', end = '2023-07-31', freq = 'M')
    .tolist()
)

ls_param_mlf_model_data_blend_date = (
    pd
    .date_range(start = '2023-08-31', end = '2023-9-30', freq = 'M')
    .tolist()
)

ls_param_mlf_model_data_holdout_date = (
    pd
    .date_range(start = '2023-10-31', end = '2023-12-31', freq = 'M')
    .tolist()
)

# COMMAND ----------

# data directories
# feature store
dir_data_fs_meta = os.path.join(dir_data_fs_parent, "d000_meta")
dir_data_fs_fea = os.path.join(dir_data_fs_parent, "d400_feature")
# dir_data_fs_mvmt = os.path.join(dir_data_fs_parent, "d500_movement")
dir_data_fs_serv = os.path.join(dir_data_fs_parent, "d600_serving")

# customize addon 
dir_data_fs_mvnt = '/mnt/ml-lab/dev_users/dev_sc/aod30_mvnt'
dir_data_fs_aod30 = "/mnt/ml-lab/dev_users/dev_sc/fea_aod30_status"

# ml factory
dir_data_mlf_exp = os.path.join(dir_data_mlf_parent, "dev_users", vt_param_mlf_user_id, vt_param_mlf_model_pattern, vt_param_mlf_model_id, vt_param_mlf_exp_id)

dir_data_mlf = dir_data_mlf_exp

# COMMAND ----------

print(dir_data_mlf)

# COMMAND ----------

# MAGIC %md ## s3 data import

# COMMAND ----------

# DBTITLE 1,import - master
df_fs_global_cycle_calendar = spark.read.format("delta").load(os.path.join(dir_data_fs_meta, "d001_global_cycle_calendar"))
df_fsr_dict_meta = spark.read.format("delta").load(os.path.join(dir_data_fs_meta, "d004_fsr_meta/fsr_dict_meta"))
df_fs_serv = spark.read.format("delta").load(os.path.join(dir_data_fs_serv, "serv_mobile_oa_consumer"))
df_fs_mvmt_acct_aod30 = spark.read.format("delta").load(dir_data_fs_mvnt)
df_fs_aod30_status = spark.read.format('delta').load(dir_data_fs_aod30)

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

print("acct wo")
display(df_fs_mvmt_acct_aod30.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## s3.5 data integration

# COMMAND ----------

# MAGIC %md ## s4 data preparation

# COMMAND ----------

# MAGIC %md ### s401 model meta data creation

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

# MAGIC %md ### s402 model unit base creation

# COMMAND ----------

# DBTITLE 1,model unit base
# create model unit base from the feature store
df_model_unit_base = (
    df_fs_serv
    .join(df_fs_aod30_status, ls_param_fs_primary_keys + ls_param_fs_reporting_keys, 
          'left' )
    .filter(f.col('aod30_status').isNull())
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

# MAGIC %md ### s403 model target creation

# COMMAND ----------

# MAGIC %md
# MAGIC Target label table creation based on:
# MAGIC - global cycle calendar
# MAGIC - unit base from feature store exclude customers who are in AOD30
# MAGIC - service from accounts enter into AOD30d + 

# COMMAND ----------

# DBTITLE 1,AOD30 movement
# writeoff movement
df_base_mvmt_srvc_aod30 = (
    df_fs_mvmt_acct_aod30
    .withColumnRenamed("movement_date", "target_date")
    .select("fs_acct_id", "target_date")      
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

# DBTITLE 1,check missing
# checking missing values for the selected features
check_missing_features(
    df_model_raw
    , "fs_srvc_id"
    , ls_param_model_features
    , ls_param_fs_reporting_keys
    , "pct"
)


# COMMAND ----------

# DBTITLE 1,check distinct
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

# DBTITLE 1,model prep data
df_model_prep = df_model_raw

# COMMAND ----------

# MAGIC %md #### rate plan

# COMMAND ----------

# DBTITLE 1,sample data check 01
display(
    df_model_raw
    .groupBy("plan_share_name")
    .agg(f.count("*").alias("cnt"))
    .orderBy(f.desc("cnt"))
)

# COMMAND ----------

# DBTITLE 1,sample data check 02
display(
    df_model_raw
    .groupBy("plan_name_std")
    .agg(f.count("*").alias("cnt"))
    .orderBy(f.desc("cnt"))
)

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

# DBTITLE 1,sample data check 03
display(
    df_model_prep
    .groupBy("plan_family", "plan_name_std")
    .agg(f.countDistinct("fs_srvc_id").alias("cnt"))
    .orderBy("plan_family", f.desc("cnt"))
)

# COMMAND ----------

# DBTITLE 1,sample data check 04
check_distinct_values(
    df_model_prep
    , ["plan_family", "plan_name_std"]
    , ls_param_fs_reporting_keys
)

# COMMAND ----------

# MAGIC %md #### network device brand

# COMMAND ----------

# DBTITLE 1,input data check 01
display(
    df_model_raw
    .groupBy("network_dvc_brand")
    .agg(f.count("*").alias("cnt"))
    .orderBy(f.desc("cnt"))
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

# DBTITLE 1,output data check 01
display(
    df_model_prep
    .groupBy("network_dvc_brand")
    .agg(f.count("*").alias("cnt"))
    .orderBy(f.desc("cnt"))
)

# COMMAND ----------

# MAGIC %md #### network device manufacturer

# COMMAND ----------

# DBTITLE 1,process
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

# MAGIC %md #### network device model

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

# DBTITLE 1,output data check 01
display(
    df_model_prep
    .groupBy("network_dvc_brand", "network_dvc_model_marketing")
    .agg(f.countDistinct("fs_srvc_id").alias("cnt"))
    .orderBy("network_dvc_brand", f.desc("cnt"))
)

# COMMAND ----------

# DBTITLE 1,output data check 02
check_distinct_values(
    df_model_prep
    , ["network_dvc_brand", "network_dvc_model_marketing"]
    , ls_param_fs_reporting_keys
)

# COMMAND ----------

# MAGIC %md #### network device model version

# COMMAND ----------

# DBTITLE 1,process
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

# DBTITLE 1,output data check
display(
    df_model_prep
    .groupBy("network_dvc_manufacturer", "network_dvc_brand", "network_dvc_model_marketing", "network_dvc_model_version")
    .agg(f.countDistinct("fs_srvc_id").alias("cnt"))
    .orderBy("network_dvc_brand", f.desc("cnt"))
)

# COMMAND ----------

# MAGIC %md #### seasonality

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

# MAGIC %md #### finalisation

# COMMAND ----------

df_model = df_model_prep

display(df_model.limit(10))

# COMMAND ----------

check_distinct_values(
    df_model
    , ls_param_model_features_char
    , ls_param_fs_reporting_keys
)

# COMMAND ----------

# MAGIC %md ### s405 model sets creation (sampling)

# COMMAND ----------

# MAGIC %md
# MAGIC Stratified Sampling based on df_model_ssc_meta & target flag to generate:
# MAGIC * training set - model creation
# MAGIC * blending set - model optimisation for stack ensembling
# MAGIC * validation set - model selection
# MAGIC * calibration set - probability calibration
# MAGIC * holdout set - model evaluation

# COMMAND ----------

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

# MAGIC %md ## s5 data export

# COMMAND ----------

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

# COMMAND ----------


