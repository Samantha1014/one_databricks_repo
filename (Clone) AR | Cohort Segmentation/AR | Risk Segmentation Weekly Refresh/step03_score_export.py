# Databricks notebook source
# %restart_python

# COMMAND ----------

# MAGIC %run "./s98_environment_setup"

# COMMAND ----------

### libraries
import os

import pyspark
from pyspark import sql 
from pyspark.sql import Window
from pyspark.sql import functions as f

import mlflow
import h2o

# COMMAND ----------

# MAGIC %run "./s99_utility_functions"

# COMMAND ----------

# system parameters
# data storage
dir_mlf_data_parent = "/mnt/ml-factory-gen-x-dev/dev_users"

# mlflow folder
dir_mlf_mlflow_parent = "/Shared/ml_lab/ml_experiments"

# hold out set foler 
dir_holdout = '/mnt/ml-lab/dev_users/dev_sc/99_misc/df_model_test_v4'

# feature store parameters
vt_param_fs_reporting_cycle_type = "calendar cycle"
ls_param_fs_primary_keys = ["fs_cust_id", "fs_acct_id", "fs_srvc_id"]
ls_param_fs_reporting_keys = ["reporting_cycle_type", "reporting_date"]

# ml factory parameters
# environment
vt_param_mlf_user_id = "dev_sc"

# generic
vt_param_mlf_model_pattern = "classification"

# model
vt_param_mlf_model_id = "mobile_oa_consumer_srvc_aod30d_pred90d"

# experiment
vt_param_mlf_exp_id = "202409_exp2"
vt_param_mlf_exp_set_id = "s1"

# model data parameters
vt_param_mlf_model_data_version_id = "v1"

# model parameters
vt_param_model_h2o_mem_size = "480G"

# COMMAND ----------

vt_param_mlf_mlflow_exp_id = vt_param_mlf_model_id + "_" + vt_param_mlf_exp_id

# directories
dir_mlf_data = f"{dir_mlf_data_parent}/{vt_param_mlf_user_id}/{vt_param_mlf_model_pattern}/{vt_param_mlf_model_id}/{vt_param_mlf_mlflow_exp_id}"

dir_mlf_mlflow = f"{dir_mlf_mlflow_parent}/{vt_param_mlf_model_pattern}/{vt_param_mlf_model_id}"

# COMMAND ----------

mlflow_experiment_entities = mlflow.get_experiment_by_name(os.path.join(dir_mlf_mlflow, vt_param_mlf_mlflow_exp_id))
vt_param_experiment_id = mlflow_experiment_entities.experiment_id

# COMMAND ----------

df_mlflow_experiment = mlflow.search_runs(vt_param_experiment_id)
df_mlflow_experiment = spark.createDataFrame(df_mlflow_experiment)
df_mlflow_experiment = (
    df_mlflow_experiment
    #.filter(f.col("`tags.mlflow.runName`").rlike(vt_param_model_version_id))
)

display(df_mlflow_experiment)

# COMMAND ----------

df_mlflow_experiment_select = (
    df_mlflow_experiment
    .filter(~f.col("`tags.mlflow.runName`").rlike("StackedEnsemble"))
    .filter(~f.col("`tags.mlflow.runName`").rlike("GLM"))
    #.orderBy(f.desc("`metrics.gen_holdout_lift_avg_10`"))
    .withColumn("index", f.row_number().over(Window.orderBy(f.desc("`metrics.gen_holdout_lift_avg_10`"))))
)

df_mlflow_run_select = df_mlflow_experiment_select.filter(f.col("index") == 1)

vt_param_model_run_id = (
    df_mlflow_run_select
    .select("run_id")
    .rdd.map(lambda x: x[0])
    .collect()[0]
)
vt_param_model_data_version_id = (
    df_mlflow_run_select
    .select("`params.gen_data_version`")
    .rdd.map(lambda x: x[0])
    .collect()[0]
)

# COMMAND ----------

display(vt_param_model_data_version_id)

# COMMAND ----------

df_model_holdout = spark.read.format("delta").load(dir_holdout)

# COMMAND ----------

display(df_model_holdout.limit(10))

# COMMAND ----------

display(df_model_holdout
        .groupBy('reporting_date')
        .agg(f.count('*'))
        )

# COMMAND ----------

# df_model_holdout = (
#    df_model_holdout
#     .filter(f.col('reporting_date').isin('2024-04-30'))
# )

# df_model_holdout_b = (
#     df_model_holdout
#     .filter(f.col('reporting_date').isin('2024-04-30', '2024-05-31', '2024-06-30'))
# )

# COMMAND ----------

# MAGIC %md ## s1 data/model preparation

# COMMAND ----------

# initiate H2O clusters
h2o.init(max_mem_size=vt_param_model_h2o_mem_size)

# COMMAND ----------

h2odf_model_holdout = h2o.H2OFrame(df_model_holdout.withColumn("reporting_date", f.date_format(f.col("reporting_date"), "yyyy|MM|dd")).toPandas())

# COMMAND ----------

h2odf_model_holdout["one_app_inactive_days"] = h2odf_model_holdout["one_app_inactive_days"].asnumeric()
h2odf_model_holdout["network_dvc_screen_pixel_ratio"] = h2odf_model_holdout["network_dvc_screen_pixel_ratio"].asnumeric()

# COMMAND ----------

# load selected model for evaluation
h2o_model_eval = mlflow.h2o.load_model('runs:/' + vt_param_model_run_id + '/model')

# load data for evaluation
h2odf_model_eval = h2odf_model_holdout

# COMMAND ----------

h2o_model_eval.params

# COMMAND ----------

df_model_predict = h2o_get_predict(h2odf_model_eval, h2o_model_eval)

# COMMAND ----------

display(df_model_predict.limit(10))

# COMMAND ----------

display(df_model_predict
        .distinct()
        .filter(f.col('fs_srvc_id').isNotNull())
        .groupBy('reporting_date')
        .agg(f.count('*'), f.countDistinct('fs_srvc_id')
             )
        )

# COMMAND ----------

df_output_curr = (df_model_predict
        .distinct()
        .filter(f.col('fs_srvc_id').isNotNull())
        .withColumn('propensity_score_raw', f.col('Y')*100)
        .withColumn(
            "propensity_score_cb"
            , f.col("propensity_score_raw")
        )
        .withColumn(
            "propensity_score"
            , f.col("propensity_score_cb")
        )
        .withColumn('rank', f.col('propensity_score'))
        .withColumn(
                    "propensity_top_ntile"
                    , f.ntile(100).over(
                        Window
                        .orderBy(f.asc("rank"))
                    )
                )
                .withColumn(
                    "propensity_segment_qt"
                    , f.when(
                        f.col("propensity_top_ntile") > 90
                        , f.lit("H")
                    ).when(
                        f.col("propensity_top_ntile") > 70
                        , f.lit("M")
                    ).otherwise(f.lit("L"))
                )
                .withColumn(
                    "propensity_segment_pbty"
                    , f.when(
                        f.col("propensity_score") > 90
                        , f.lit("H")
                    ).when(
                        f.col("propensity_score") > 70
                        , f.lit("M")
                    ).otherwise(f.lit("L"))
                )
                .withColumn('reporting_date', f.to_date('reporting_date', 'yyyy|MM|dd'))
                .select(*ls_param_fs_primary_keys, *ls_param_fs_reporting_keys
                        ,   "propensity_score_raw"
                    , "propensity_score_cb"
                    , "propensity_score"
                    , "propensity_top_ntile"
                    , "propensity_segment_qt"
                    , "propensity_segment_pbty" )
        )

# COMMAND ----------

export_data(
            df = df_output_curr
            , export_path = '/mnt/ml-lab/dev_users/dev_sc/99_misc/predict_aod30d_propensity_v3'
            , export_format = "delta"
            , export_mode = "append"
            , flag_overwrite_schema = True
            , flag_dynamic_partition = True
            , ls_dynamic_partition = ["reporting_date"]
        )

# COMMAND ----------

df_test = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/99_misc/predict_aod30d_propensity_v3')

# COMMAND ----------

display(df_test
        .groupBy('reporting_date', 'reporting_cycle_type')
        .agg(f.count('*')
             , f.countDistinct('fs_srvc_id')
          )
        )

# COMMAND ----------

display(df_test
        .groupBy('reporting_date', 'propensity_segment_qt')
        .agg(f.count('*'), f.countDistinct('fs_srvc_id')
             )
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## one off export to snowflake

# COMMAND ----------

#------------ login to snowflake
password = dbutils.secrets.get(scope = "auea-kv-sbx-dxdtlprdct01", key = "sfdbrsdskey")

options = {
  "sfUrl": "vodafonenz_prod.australia-east.azure.snowflakecomputing.com/", 
  "sfUser": "SVC_LAB_DS_DATABRICKS",
  "pem_private_key": password.replace('\\n', '\n'),
  "sfDatabase": "LAB_ML_STORE",
  "sfSchema": "SANDBOX",
  "sfWarehouse": "LAB_DS_WH_SCALE"
}

# COMMAND ----------

(
    df_test
    .write
    .format("snowflake")
    .options(**options)
    .option("dbtable", "lab_ml_store.sandbox.sc_one_off_aod30d_propensity_score")
    .mode("overwrite")
    .save()
)
