# Databricks notebook source
# MAGIC %md
# MAGIC ### s01 environment set up 

# COMMAND ----------

# DBTITLE 1,library
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window
import os

# COMMAND ----------

# DBTITLE 1,utilities 00
# MAGIC %run "/Workspace/Users/ethan.li@one.nz/s04_resources/s01_utility_functions/utils_spark_df"

# COMMAND ----------

# DBTITLE 1,utilities 01
# MAGIC %run "../Function"

# COMMAND ----------

# MAGIC %md
# MAGIC #### directories

# COMMAND ----------

# DBTITLE 1,feature store 01
dir_data_parent = "/mnt/feature-store-dev"
dir_data_parent_shared = os.path.join(dir_data_parent, "dev_shared")
dir_data_parent_users = os.path.join(dir_data_parent, "dev_users/dev_sc")

# COMMAND ----------

# DBTITLE 1,feature store 02
dir_data_raw =  os.path.join(dir_data_parent_shared, 'd100_raw')
dir_data_meta = os.path.join(dir_data_parent_users, 'd000_meta')
dir_data_stg = os.path.join(dir_data_parent_users, "d200_staging")
dir_data_int =  os.path.join(dir_data_parent_users, "d200_intermediate")
dir_data_prm =  os.path.join(dir_data_parent_users, "d300_primary")
dir_data_fea =  os.path.join(dir_data_parent_users, "d400_feature")
dir_data_mvmt = os.path.join(dir_data_parent_users, "d500_movement")
dir_data_serv = os.path.join(dir_data_parent_users, "d600_serving")
dir_data_tmp =  os.path.join(dir_data_parent_users, "d999_tmp")

# COMMAND ----------

# MAGIC %md
# MAGIC ### s02 data import

# COMMAND ----------

# MAGIC %run "./raw_coll_action"

# COMMAND ----------

# DBTITLE 1,parameters
vt_reporting_cycle_type = 'rolling cycle'

# COMMAND ----------

# DBTITLE 1,collection action
df_output_curr = (
        df_coll_action
        .withColumn('reporting_date', f.next_day(f.col('coll_complete_date'), 'Sun'))
        .withColumn('reporting_cycle_type', f.lit(vt_reporting_cycle_type))
        .withColumn("data_update_date", f.current_date())
        .withColumn("data_update_dttm", f.current_timestamp())
        #.limit(10)
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### s03 export data

# COMMAND ----------

export_data(
    df=df_output_curr
    , export_path = os.path.join(dir_data_stg, 'stg_collection_action')
    , export_format = 'delta'
    , export_mode = 'overwrite'
    , flag_overwrite_schema = True
    , flag_dynamic_partition = False    
    , ls_dynamic_partition = ['reporting_date', 'reporting_cycle_type']
)
