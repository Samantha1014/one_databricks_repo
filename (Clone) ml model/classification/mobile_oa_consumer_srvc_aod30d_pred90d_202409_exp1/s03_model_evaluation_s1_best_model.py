# Databricks notebook source
# MAGIC %md ## s0 environment setup

# COMMAND ----------

# DBTITLE 1,environment
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

# COMMAND ----------

# DBTITLE 1,utility functions
# MAGIC %run "./s99_utility_functions"

# COMMAND ----------

# DBTITLE 1,parameters
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
vt_param_mlf_user_id = "dev_sc"

# generic
vt_param_mlf_model_pattern = "classification"

# model
vt_param_mlf_model_id = "mobile_oa_consumer_srvc_aod30d_pred90d"

# experiment
vt_param_mlf_exp_id = "202409_exp1"
vt_param_mlf_exp_set_id = "s1"

# model data parameters
vt_param_mlf_model_data_version_id = "v1"

# model parameters
vt_param_model_h2o_mem_size = "480G"

# COMMAND ----------

# DBTITLE 1,directories
vt_param_mlf_mlflow_exp_id = vt_param_mlf_model_id + "_" + vt_param_mlf_exp_id

# directories
dir_mlf_data = f"{dir_mlf_data_parent}/{vt_param_mlf_user_id}/{vt_param_mlf_model_pattern}/{vt_param_mlf_model_id}/{vt_param_mlf_mlflow_exp_id}"

dir_mlf_mlflow = f"{dir_mlf_mlflow_parent}/{vt_param_mlf_model_pattern}/{vt_param_mlf_model_id}"

# COMMAND ----------

dir_mlf_mlflow

# COMMAND ----------

# DBTITLE 1,mlflow entities 01
mlflow_experiment_entities = mlflow.get_experiment_by_name(os.path.join(dir_mlf_mlflow, vt_param_mlf_mlflow_exp_id))
vt_param_experiment_id = mlflow_experiment_entities.experiment_id

# COMMAND ----------

mlflow_experiment_entities

# COMMAND ----------

vt_param_experiment_id

# COMMAND ----------

# DBTITLE 1,mlflow entities 02
df_mlflow_experiment = mlflow.search_runs(vt_param_experiment_id)
df_mlflow_experiment = spark.createDataFrame(df_mlflow_experiment)
df_mlflow_experiment = (
    df_mlflow_experiment
    #.filter(f.col("`tags.mlflow.runName`").rlike(vt_param_model_version_id))
)

display(df_mlflow_experiment)

# COMMAND ----------

# DBTITLE 1,mlflow entities 03
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

vt_param_model_data_version_id

# COMMAND ----------

# MAGIC %md ## s1 data/model preparation
# MAGIC

# COMMAND ----------

# DBTITLE 1,h2o instance initiation
# initiate H2O clusters
h2o.init(max_mem_size=vt_param_model_h2o_mem_size)

# COMMAND ----------

# DBTITLE 1,data import
df_model_valid = spark.read.format("delta").load(os.path.join(dir_mlf_data, f"model_valid_{vt_param_model_data_version_id}"))
df_model_holdout = spark.read.format("delta").load(os.path.join(dir_mlf_data, f"model_holdout_{vt_param_model_data_version_id}"))

# COMMAND ----------

#df_model_holdout = (
#    df_model_holdout
#    .filter(f.col("reporting_date") == '2023-05-31')
#)

# COMMAND ----------

# DBTITLE 1,holdout data prep 01
#h2odf_model_valid = h2o.H2OFrame(df_model_valid.withColumn("reporting_date", f.date_format(f.col("reporting_date"), "yyyy|MM|dd")).toPandas())
h2odf_model_holdout = h2o.H2OFrame(df_model_holdout.withColumn("reporting_date", f.date_format(f.col("reporting_date"), "yyyy|MM|dd")).toPandas())

# COMMAND ----------

# DBTITLE 1,holdout data prep 02
h2odf_model_holdout["one_app_inactive_days"] = h2odf_model_holdout["one_app_inactive_days"].asnumeric()
h2odf_model_holdout["network_dvc_screen_pixel_ratio"] = h2odf_model_holdout["network_dvc_screen_pixel_ratio"].asnumeric()

# COMMAND ----------

# DBTITLE 1,model import
# load selected model for evaluation
h2o_model_eval = mlflow.h2o.load_model('runs:/' + vt_param_model_run_id + '/model')

# load data for evaluation
h2odf_model_eval = h2odf_model_holdout

# COMMAND ----------

h2o_model_eval.params

# COMMAND ----------

# MAGIC %md ## s2 model evaluation
# MAGIC

# COMMAND ----------

# DBTITLE 1,basic overview
h2o_perf = h2o_model_eval.model_performance(h2odf_model_eval)
h2o_perf

# COMMAND ----------

# DBTITLE 1,gainlift 01
df_model_predict = h2o_get_predict(h2odf_model_eval, h2o_model_eval)

df_model_gainlift = h2o_get_gainlift(
    df_model_predict
    .withColumn("group", f.col("reporting_date"))
    .withColumn("id", f.col("fs_srvc_id"))
    , 10
)

#display(df_base_gainlift.filter(f.col("ntile") <= 3))

display(
    df_model_gainlift
    .filter(f.col("ntile") <= 3)
    .groupBy("ntile")
    .agg(
        f.mean("gain").alias("gain_avg")
        , f.mean("lift").alias("lift_avg")
    )
    .orderBy("ntile")
)

# COMMAND ----------

# DBTITLE 1,gainlift 02
h2o_perf.gains_lift_plot()

# COMMAND ----------

# DBTITLE 1,ROC & PRC
h2o_perf.plot(type = "roc")
h2o_perf.plot(type = "pr")

# COMMAND ----------

# DBTITLE 1,variable importance 01
h2o_model_eval.varimp_plot(20)

# COMMAND ----------

# DBTITLE 1,variable importance 02
df_model_varimp = h2o_model_eval.varimp(use_pandas=True)
display(df_model_varimp)

# COMMAND ----------

# DBTITLE 1,shap 01
h2o_model_eval.shap_summary_plot(h2odf_model_eval)

# COMMAND ----------

# DBTITLE 1,shap 02
h2o_model_eval.shap_summary_plot(h2odf_model_eval, columns = list(df_model_varimp.variable[0:20])[::-1])

# COMMAND ----------

# DBTITLE 1,partial plot 01
 h2o_model_eval.partial_plot(
    data = h2odf_model_eval
    #, server
    , cols = ["credit_score"]
    , nbins = 28
)

# COMMAND ----------

# DBTITLE 1,partial plot 02
 h2o_model_eval.partial_plot(
    data = h2odf_model_eval
    #, server
    , cols = ["credit_score_segment"]
    , nbins = 28
)

# COMMAND ----------

# DBTITLE 1,partial plot 03
 h2o_model_eval.partial_plot(
    data = h2odf_model_eval
    #, server
    , cols = ["network_dvc_hardware_rating_score"]
    , nbins = 28
)

# COMMAND ----------

# DBTITLE 1,partial plot 04
vt_param_top_x_features = 10
ls_param_skip = [99]

for i in range(vt_param_top_x_features):
    if i not in ls_param_skip:
        vt_param_feature_eval = df_model_varimp[i:(i+1)]["variable"].str.split('.').str[0].tolist()
        print(vt_param_feature_eval)
        h2o_model_eval.partial_plot(
            data = h2odf_model_eval
            #, server
            , cols = vt_param_feature_eval
            , nbins = 148
        )
