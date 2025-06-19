# Databricks notebook source
# MAGIC %md
# MAGIC #library

# COMMAND ----------



# COMMAND ----------

password = dbutils.secrets.get(scope = "auea-kv-sbx-dxdtlprdct01", key = "sfdbrsdskey")
options = {
  "sfUrl": "vodafonenz_prod.australia-east.azure.snowflakecomputing.com", 
  "sfUser": "SVC_LAB_DS_DATABRICKS",
  "pem_private_key": password.replace('\\n', '\n'),
  "sfDatabase": "Prod_Account_Risk",
  "sfSchema": "Modelled",
  "sfWarehouse": "LAB_DS_WH_SCALE"
}


# COMMAND ----------

df_interflow = 
