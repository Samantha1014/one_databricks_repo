# Databricks notebook source
# MAGIC %md
# MAGIC **Objective**
# MAGIC
# MAGIC select ~ 20K one wallet eligible customers to get 50% off the Samsung GalaxyS23FE, criteria 
# MAGIC 1. have $170 welcome boost in their balance 
# MAGIC 2. high propensity to churn 
# MAGIC 3. medium propensity to buy 
# MAGIC 4. on a S20 device, or older 
# MAGIC

# COMMAND ----------

# MAGIC
# MAGIC %md ## s000 environment setup

# COMMAND ----------

# MAGIC %md ### s001 libraries

# COMMAND ----------

# libraries
import os
import pyspark
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window

# COMMAND ----------

# MAGIC
# MAGIC %md ### s002 sf connectivity

# COMMAND ----------

# MAGIC %run "./utility_functions"

# COMMAND ----------

# ------------- Use snowflake utility
sfUtils = sc._jvm.net.snowflake.spark.snowflake.Utils

# ------------ login to snowflake
password = dbutils.secrets.get(scope = "auea-kv-sbx-dxdtlprdct01", key = "sfdbrsdskey")

options = {
  "sfUrl": "vodafonenz_prod.australia-east.azure.snowflakecomputing.com/", 
  "sfUser": "SVC_LAB_DS_DATABRICKS",
  "pem_private_key": password.replace('\\n', '\n'),
  "sfDatabase": "LAB_ML_STORE",
  "sfSchema": "SANDBOX",
  "sfWarehouse": "LAB_DS_WH"
}

# COMMAND ----------

# MAGIC %md ### s003 directories

# COMMAND ----------

dir_fs_data_parent = "/mnt/feature-store-prod-lab"
dir_mls_data_parent = "/mnt/ml-store-prod-lab/classification"

# COMMAND ----------

dir_mls_data_score = os.path.join(dir_mls_data_parent, "d400_model_score")

# COMMAND ----------

dir_fs_data_meta = os.path.join(dir_fs_data_parent, 'd000_meta')
dir_fs_data_raw =  os.path.join(dir_fs_data_parent, 'd100_raw')
dir_fs_data_int =  os.path.join(dir_fs_data_parent, "d200_intermediate")
dir_fs_data_prm =  os.path.join(dir_fs_data_parent, "d300_primary")
dir_fs_data_fea =  os.path.join(dir_fs_data_parent, "d400_feature")
dir_fs_data_target = os.path.join(dir_fs_data_parent, "d500_movement")
dir_fs_data_serv = os.path.join(dir_fs_data_parent, "d600_serving")

# COMMAND ----------

df_wallet_cust = (
     spark.read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """
            select 
                billing_account_number as fs_acct_id
                , MAX(wallet_id) as wallet_id
                , 'Y' as wallet_eligible_flag
            from PROD_WALLET.MODELLED.D_WALLET_CUSTOMER
            where current_record_ind = 1
            and wallet_eligibility_flag = 'Y'
            and service_status_name = 'Active'
            group by 1
        """
    )
    .load()
)

df_wallet_cust = lower_col_names(df_wallet_cust)

display(df_wallet_cust.limit(10))

