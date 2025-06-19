# Databricks notebook source
# MAGIC %md ## s1 environment setup

# COMMAND ----------

# DBTITLE 1,environment setup
# MAGIC %run "./s98_environment_setup"

# COMMAND ----------

# DBTITLE 1,libraries
### libraries
import os

import pyspark
from pyspark import sql 
from pyspark.sql import Window
from pyspark.sql import functions as f

import mlflow
import h2o
from h2o.automl import H2OAutoML

# COMMAND ----------

# DBTITLE 1,utility functions
# MAGIC %run "./s99_utility_functions"

# COMMAND ----------

# MAGIC %md ## s2 parameters

# COMMAND ----------

# DBTITLE 1,s201 user parameters
# system parameters
# data storage
dir_mlf_data_parent = "/mnt/ml-factory-gen-x-dev/dev_users"

# mlflow folder
dir_mlf_mlflow_parent = "/Shared/ml_lab/ml_experiments"

# feature store parameters
vt_param_fs_reporting_cycle_type = "calendar cycle"
ls_param_fs_primary_keys = ["fs_cust_id", "fs_acct_id", "fs_srvc_id"]
ls_param_fs_reporting_keys = ["reporting_cycle_type", "reporting_date"]

# ml factory parameters
# environment
vt_param_mlf_user_id = "dev_el"

# generic
vt_param_mlf_model_pattern = "classification"

# model
vt_param_mlf_model_id = "mobile_oa_consumer_srvc_writeoff_pred365d"

# experiment
vt_param_mlf_exp_id = "202408_exp1"
vt_param_mlf_exp_set_id = "s2"

# model data parameters
vt_param_mlf_model_data_version_id = "v1"

# model parameters
vt_param_model_h2o_mem_size = "480G"
vt_param_model_h2o_runtime_max = 3600
vt_param_model_h2o_models_max = 50
vt_param_model_h2o_grid_id = "gbm_grid_20240811"
vt_param_model_h2o_seed = 43

# COMMAND ----------

# DBTITLE 1,s202 directories
vt_param_mlf_mlflow_exp_id = vt_param_mlf_model_id + "_" + vt_param_mlf_exp_id

# directories
dir_mlf_data = f"{dir_mlf_data_parent}/{vt_param_mlf_user_id}/{vt_param_mlf_model_pattern}/{vt_param_mlf_model_id}/{vt_param_mlf_mlflow_exp_id}"

dir_mlf_mlflow = f"{dir_mlf_mlflow_parent}/{vt_param_mlf_model_pattern}/{vt_param_mlf_model_id}"

# COMMAND ----------

# DBTITLE 1,s203 MLFlow experiment
# set up experiment log to track
mlflow.set_experiment(experiment_name = os.path.join(dir_mlf_mlflow, vt_param_mlf_mlflow_exp_id))

# COMMAND ----------

# MAGIC %md ## s3 data import

# COMMAND ----------

# model data import
df_model_data_params = spark.read.format("delta").load(os.path.join(dir_mlf_data, f"model_data_params_{vt_param_mlf_model_data_version_id}"))
df_model_train = spark.read.format("delta").load(os.path.join(dir_mlf_data, f"model_train_{vt_param_mlf_model_data_version_id}"))
df_model_valid = spark.read.format("delta").load(os.path.join(dir_mlf_data, f"model_valid_{vt_param_mlf_model_data_version_id}"))
df_model_calibrate = spark.read.format("delta").load(os.path.join(dir_mlf_data, f"model_calibrate_{vt_param_mlf_model_data_version_id}"))
df_model_blend = spark.read.format("delta").load(os.path.join(dir_mlf_data, f"model_blend_{vt_param_mlf_model_data_version_id}"))
df_model_holdout = spark.read.format("delta").load(os.path.join(dir_mlf_data, f"model_holdout_{vt_param_mlf_model_data_version_id}"))

# COMMAND ----------

# MAGIC %md ## s4 model preparation

# COMMAND ----------

# DBTITLE 1,s401 h2o cluster initiation
# initiate H2O clusters
h2o.init(max_mem_size=vt_param_model_h2o_mem_size)

# COMMAND ----------

# MAGIC %md
# MAGIC ### s402 model data preparation

# COMMAND ----------

# DBTITLE 1,data prep 01
df_model_train = (
    df_model_train
    .filter(f.col("reporting_date").between("2022-11-30", "2023-02-28"))
)

df_model_valid = (
    df_model_valid
    .filter(f.col("reporting_date").between("2022-11-30", "2023-02-28"))
)

# COMMAND ----------

# DBTITLE 1,data prep 02
# convert all spark df to h2o df
h2odf_model_train = h2o.H2OFrame(df_model_train.toPandas())
h2odf_model_valid = h2o.H2OFrame(df_model_valid.toPandas())
h2odf_model_calibrate = h2o.H2OFrame(df_model_calibrate.toPandas())
h2odf_model_blend = h2o.H2OFrame(df_model_blend.toPandas())
h2odf_model_holdout = h2o.H2OFrame(df_model_holdout.toPandas())

# COMMAND ----------

# MAGIC %md ### s403 features & target definement

# COMMAND ----------

ls_param_model_x = (
    df_model_train
    .drop(*(ls_param_fs_reporting_keys + ls_param_fs_primary_keys))
    .drop("target_label")
    .columns
)

ls_param_model_y = "target_label"

# COMMAND ----------

# convert data type for performance estimation
h2odf_model_train["one_app_inactive_days"] = h2odf_model_train["one_app_inactive_days"].asnumeric()
h2odf_model_blend["one_app_inactive_days"] = h2odf_model_blend["one_app_inactive_days"].asnumeric()

h2odf_model_train["network_dvc_screen_pixel_ratio"] = h2odf_model_train["network_dvc_screen_pixel_ratio"].asnumeric()
h2odf_model_blend["network_dvc_screen_pixel_ratio"] = h2odf_model_blend["network_dvc_screen_pixel_ratio"].asnumeric()

# COMMAND ----------

# MAGIC %md ## s5 model training

# COMMAND ----------

# DBTITLE 1,hyper parameters
h2o_gbm_hyper_params = {
    "col_sample_rate": [0.4, 0.7, 1.0]
    , "col_sample_rate_per_tree": [0.4, 0.7, 1.0]
    , "learn_rate": [0.01, 0.1]
    , "max_depth": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    , "min_rows": [1, 5, 10, 15, 30, 100]
    , "min_split_improvement": [1e-4, 1e-5]
    , "sample_rate": [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
}

h2o_gbm_search_criteria = {
    'strategy': "RandomDiscrete" 
    , "max_runtime_secs": vt_param_model_h2o_runtime_max
    , 'max_models': vt_param_model_h2o_models_max
    #, "seed": 43
}

# COMMAND ----------

from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.gbm import H2OGradientBoostingEstimator

h2o_gbm_grid_01 = H2OGridSearch(
    H2OGradientBoostingEstimator
    , grid_id = vt_param_model_h2o_grid_id
    , hyper_params = h2o_gbm_hyper_params
    , search_criteria=h2o_gbm_search_criteria
)

h2o_gbm_grid_01.train(
    ls_param_model_x
    , ls_param_model_y
    , training_frame = h2odf_model_train
    , calibrate_model = True
    , calibration_frame = h2odf_model_calibrate
    , ntrees = 10000
)

# COMMAND ----------

h2o_gbm_grid_01.get_grid(sort_by='auc', decreasing=True)

# COMMAND ----------

# MAGIC %md ## s6 model export

# COMMAND ----------

# MAGIC %md ### s601 data parameters

# COMMAND ----------

# model data parameters for logging
vt_param_model_cycle_type = df_model_data_params.collect()[0].model_cycle_type
vt_param_model_freq_type = df_model_data_params.collect()[0].model_freq_type
vt_param_model_predict_days = df_model_data_params.collect()[0].predict_days
vt_param_model_data_version = df_model_data_params.collect()[0].data_version

vt_param_model_train_date_min = df_model_data_params.collect()[0].train_set_date_min
vt_param_model_train_date_max = df_model_data_params.collect()[0].train_set_date_max

vt_param_model_valid_pct = df_model_data_params.collect()[0].valid_set_sample_pct
vt_param_model_valid_date_min = df_model_data_params.collect()[0].valid_set_date_min
vt_param_model_valid_date_max = df_model_data_params.collect()[0].valid_set_date_max

vt_param_model_calibrate_pct = df_model_data_params.collect()[0].calibrate_set_sample_pct
vt_param_model_calibrate_date_min = df_model_data_params.collect()[0].calibrate_set_date_min
vt_param_model_calibrate_date_max = df_model_data_params.collect()[0].calibrate_set_date_max

vt_param_model_blend_date_min = df_model_data_params.collect()[0].blend_set_date_min
vt_param_model_blend_date_max = df_model_data_params.collect()[0].blend_set_date_max

vt_param_model_holdout_date_min = df_model_data_params.collect()[0].holdout_set_date_min
vt_param_model_holdout_date_max = df_model_data_params.collect()[0].holdout_set_date_max

# COMMAND ----------

# MAGIC %md ### s602 model parameters

# COMMAND ----------

# MAGIC %md ### s603 model stats export

# COMMAND ----------

# convert data type for performance estimation
h2odf_model_holdout["one_app_inactive_days"] = h2odf_model_holdout["one_app_inactive_days"].asnumeric()
h2odf_model_valid["one_app_inactive_days"] = h2odf_model_valid["one_app_inactive_days"].asnumeric()

h2odf_model_holdout["network_dvc_screen_pixel_ratio"] = h2odf_model_holdout["network_dvc_screen_pixel_ratio"].asnumeric()
h2odf_model_valid["network_dvc_screen_pixel_ratio"] = h2odf_model_valid["network_dvc_screen_pixel_ratio"].asnumeric()

# COMMAND ----------

vt_param_grid_model_cnt = len(h2o_gbm_grid_01.model_ids)

# COMMAND ----------

for i in range(0, vt_param_grid_model_cnt):
    h2o_model_select = h2o_gbm_grid_01.models[i]
    vt_param_model_nm = h2o_gbm_grid_01.model_ids[i]
    vt_param_run_name_curr = vt_param_mlf_exp_set_id + '_' + vt_param_model_nm

    print(vt_param_run_name_curr)
    with mlflow.start_run(run_name = vt_param_run_name_curr):
        
        #h2o_model_select = h2o.get_model(i)
        
        # Log and save best model (mlflow.h2o provides API for logging & loading H2O models)
        mlflow.h2o.log_model(h2o_model_select, artifact_path="model")
        
        # parameters
        # general parameters
        mlflow.log_param("gen_model_cycle_type", vt_param_model_cycle_type)
        mlflow.log_param("gen_model_freq_type", vt_param_model_freq_type)
        mlflow.log_param("gen_model_predict_days", vt_param_model_predict_days)
        
        mlflow.log_param("gen_data_version", vt_param_model_data_version)
        
        mlflow.log_param("gen_data_train_set_date_min", vt_param_model_train_date_min)
        mlflow.log_param("gen_data_train_set_date_max", vt_param_model_train_date_max)
        
        mlflow.log_param("gen_data_valid_set_sample_pct", vt_param_model_valid_pct)
        mlflow.log_param("gen_data_valid_set_date_min", vt_param_model_valid_date_min)
        mlflow.log_param("gen_data_valid_set_date_max", vt_param_model_valid_date_max)

        mlflow.log_param("gen_data_calibrate_set_sample_pct", vt_param_model_calibrate_pct)
        mlflow.log_param("gen_data_calibrate_set_date_min", vt_param_model_calibrate_date_min)
        mlflow.log_param("gen_data_calibrate_set_date_max", vt_param_model_calibrate_date_max)
        
        mlflow.log_param("gen_data_holdout_set_date_min", vt_param_model_holdout_date_min)
        mlflow.log_param("gen_data_holdout_set_date_max", vt_param_model_holdout_date_max)
        
        mlflow.log_param("gen_data_blend_set_date_min", vt_param_model_blend_date_min)
        mlflow.log_param("gen_data_blend_set_date_max", vt_param_model_blend_date_max)
        
        # h2o parameters
        mlflow.log_param("h2o_runtime_max", vt_param_model_h2o_runtime_max)
        
        # metrics
        # valid set metrics
        h2o_perf_valid = h2o_model_select.model_performance(h2odf_model_valid)

        df_predict_valid = h2o_get_predict(h2odf_model_valid, h2o_model_select)

        df_gainlift_valid = h2o_get_gainlift(
            df_predict_valid
            .withColumn("group", f.col("reporting_date"))
            .withColumn("id", f.col("fs_srvc_id"))
            , 10
        )

        df_gainlift_valid = (
            df_gainlift_valid
            .filter(f.col("ntile") <= 3)
            .groupBy("ntile")
            .agg(
                f.mean("gain").alias("gain_avg")
                , f.mean("lift").alias("lift_avg")
            )
            .orderBy("ntile")
        )

        ls_gain_avg_valid = (
            df_gainlift_valid
            .select("gain_avg")
            .rdd.map(lambda x: x[0])
            .collect()
        )

        ls_lift_avg_valid = (
            df_gainlift_valid
            .select("lift_avg")
            .rdd.map(lambda x: x[0])
            .collect()
        )
        
        mlflow.log_metric("gen_valid_gain_avg_10", ls_gain_avg_valid[0])
        mlflow.log_metric("gen_valid_gain_avg_20", ls_gain_avg_valid[1])
        mlflow.log_metric("gen_valid_gain_avg_30", ls_gain_avg_valid[2])
        mlflow.log_metric("gen_valid_lift_avg_10", ls_lift_avg_valid[0])
        mlflow.log_metric("gen_valid_lift_avg_20", ls_lift_avg_valid[1])
        mlflow.log_metric("gen_valid_lift_avg_30", ls_lift_avg_valid[2])
        mlflow.log_metric("gen_valid_auc", h2o_perf_valid.auc())
        mlflow.log_metric("gen_valid_aucpr", h2o_perf_valid.aucpr())
        mlflow.log_metric("gen_valid_gini", h2o_perf_valid.gini())
        mlflow.log_metric("gen_valid_log_loss", h2o_perf_valid.logloss())
        mlflow.log_metric("gen_valid_rmse", h2o_perf_valid.rmse())
        mlflow.log_metric("gen_valid_r2", h2o_perf_valid.r2())
        
        # holdout set metrics
        h2o_perf_holdout = h2o_model_select.model_performance(h2odf_model_holdout)

        df_predict_holdout = h2o_get_predict(h2odf_model_holdout, h2o_model_select)

        df_gainlift_holdout = h2o_get_gainlift(
            df_predict_holdout
            .withColumn("group", f.col("reporting_date"))
            .withColumn("id", f.col("fs_srvc_id"))
            , 10
        )

        df_gainlift_holdout = (
            df_gainlift_holdout
            .filter(f.col("ntile") <= 3)
            .groupBy("ntile")
            .agg(
                f.mean("gain").alias("gain_avg")
                , f.mean("lift").alias("lift_avg")
            )
            .orderBy("ntile")
        )

        ls_gain_avg_holdout = (
            df_gainlift_holdout
            .select("gain_avg")
            .rdd.map(lambda x: x[0])
            .collect()
        )

        ls_lift_avg_holdout = (
            df_gainlift_holdout
            .select("lift_avg")
            .rdd.map(lambda x: x[0])
            .collect()
        )
        
        mlflow.log_metric("gen_holdout_gain_avg_10", ls_gain_avg_holdout[0])
        mlflow.log_metric("gen_holdout_gain_avg_20", ls_gain_avg_holdout[1])
        mlflow.log_metric("gen_holdout_gain_avg_30", ls_gain_avg_holdout[2])
        mlflow.log_metric("gen_holdout_lift_avg_10", ls_lift_avg_holdout[0])
        mlflow.log_metric("gen_holdout_lift_avg_20", ls_lift_avg_holdout[1])
        mlflow.log_metric("gen_holdout_lift_avg_30", ls_lift_avg_holdout[2])
        mlflow.log_metric("gen_holdout_auc", h2o_perf_holdout.auc())
        mlflow.log_metric("gen_holdout_aucpr", h2o_perf_holdout.aucpr())
        mlflow.log_metric("gen_holdout_gini", h2o_perf_holdout.gini())
        mlflow.log_metric("gen_holdout_log_loss", h2o_perf_holdout.logloss())
        mlflow.log_metric("gen_holdout_rmse", h2o_perf_holdout.rmse())
        mlflow.log_metric("gen_holdout_r2", h2o_perf_holdout.r2())
        
mlflow.end_run()
