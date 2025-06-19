# Databricks notebook source
from pyspark.sql import functions as f 

# COMMAND ----------

df_enthicity_output = spark.read.format('parquet').load('dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/enthnicity_output.parquet')

df_census = spark.read.format('csv').option('header', True).load('/FileStore/mnt/ml-lab/dev_shared/census2023/2023_census_population_raw.csv')

df_fea_unitbase = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base')



# COMMAND ----------

display(
    df_fea_unitbase
    .filter(f.col('reporting_date') >= '2024-01-01')
    .filter(f.col('reporting_cycle_type') == 'calendar cycle')
    .join(
        df_enthicity_output
          , f.col('siebel_row_id') == f.col('fs_cust_id'), 'left')
    .groupBy('reporting_date', 'ethnicity')
    .agg(
        f.countDistinct('fs_acct_id')
        )   
)

# COMMAND ----------

#display(df_enthicity_output.limit(100))

# COMMAND ----------

df_port_event = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d100_raw/d104_src/port_events')
df_unit_base = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base')

#df_ifp_base = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/')

df_enthicity_output = spark.read.format('parquet').load('dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/enthnicity_output.parquet')

# COMMAND ----------

display(
    df_unit_base
    .agg(f.max('reporting_date'))
)

# COMMAND ----------

display(
    df_port_event
    .filter(
        f.col('LOSING_SRVC_PROV_NAME')
        .isin(
            'One NZ Mobile', 'One NZ Local'
            , 'Farmside Mobile', 'Farmside Mobile Legacy'
            , 'Mighty Mobile', 'Kogan Mobile'
        )
    )
    .filter(
        f.col('gain_company_name')!= 'Vodafone'
    )
    .filter(f.col('date_key') >= '2024-01-01')
    # .groupBy('date_key')
    # .agg(f.countDistinct('prim_accs_num'))
)    

# COMMAND ----------

# MAGIC %md
# MAGIC ### export to snowflake

# COMMAND ----------

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


(
    df_enthicity_output
    .write
    .format("snowflake")
    .options(**options)
    .option("dbtable", "LAB_ML_STORE.SANDBOX.SC_MOBILE_OA_CONSUMER_ENTHICITY_OUTPUT")
    .mode("append")
    .save()
)
