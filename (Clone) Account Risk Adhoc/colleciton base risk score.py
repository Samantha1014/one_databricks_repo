# Databricks notebook source
# MAGIC %md
# MAGIC ### S01 Library

# COMMAND ----------

### libraries
import pyspark
import os

import re
import numpy as np

from pyspark import sql 
from pyspark.sql import Window
from pyspark.sql import functions as f

# COMMAND ----------

# MAGIC %md
# MAGIC ## S02 Data load

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

# aod report in edw snowflake 
df_aod_base = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """
        select * from prod_edw.raw.custinsight
        """
    )
    .load()
)


# COMMAND ----------

# aod report upload to snowflake 
df_aod_base = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """
        select * from lab_ml_store.sandbox.account_risk_financereport_06052024
        """
    )
    .load()
)

# COMMAND ----------

df_aod_base = spark.read.format('csv').option('header', 'true').load('dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/FinanceReport_01072024021842.csv')


# COMMAND ----------

display(df_aod_base.count()) # 630610

# COMMAND ----------

df_score_base = spark.read.format('delta').load('dbfs:/mnt/ml-store-prod-lab/classification/d400_model_score/mobile_oa_consumer_srvc_writeoff_pred120d/model_version=version_1/reporting_cycle_type=rolling cycle')

vt_max_date = df_score_base.agg(f.max('reporting_date').alias('max_reporting_date')).collect()[0][0]

# COMMAND ----------

display(df_score_base.limit(10))

# COMMAND ----------

display(vt_max_date)

# COMMAND ----------

df_score = df_score_base.filter(f.col('reporting_date') == vt_max_date)  


# COMMAND ----------

# MAGIC %md
# MAGIC ### S03 Transform

# COMMAND ----------

# DBTITLE 1,transform to remove leading 0
df_aod_base = (
    df_aod_base
    .withColumn('ACCOUNT_REF_NO', f.regexp_replace('ACCOUNT_REF_NO', "^0+", "") )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### S04 Development

# COMMAND ----------

df_score_01 = (df_score
               .select('fs_acct_id', 'fs_cust_id', 'reporting_date', 'propensity_score', 'propensity_segment_qt', 'predict_start_date', 'predict_end_date')
               .withColumn('rank', 
                           f.row_number().over(
                               Window.partitionBy('fs_acct_id', 'fs_cust_id')
                               .orderBy(f.desc('propensity_score'))
                               )
                           ) ## get the highest score 
               .filter(f.col('rank') == 1)
               .drop('rank')
               )

# COMMAND ----------

display(df_score_01
        .groupBy('propensity_segment_qt')
        .agg(f.countDistinct('fs_acct_id', 'fs_cust_id'))
        )

# COMMAND ----------

df_score_out = (df_score_01.alias('a')
        .join(df_aod_base, f.col('ACCOUNT_REF_NO') == f.col('fs_acct_id'), 'inner')
        .filter(f.col('TOTAL_BALANCE') >0) 
        .withColumn('aod_sum', f.col('AOD_01TO30') + f.col('AOD_31TO60') + f.col('AOD_61TO90')
                    + f.col('AOD_91TO120') + f.col('AOD_121TO150') + f.col('AOD_151TO180') + f.col('AOD_181PLUS')
                     )
        .filter(f.col('aod_sum') > 0) # filter on only those in debt 
        .select('a.*', 'TOTAL_BALANCE',  'AOD_01TO30', 'AOD_31TO60', 'AOD_61TO90'
                , 'AOD_91TO120', 'AOD_121TO150'	,'AOD_151TO180','AOD_181PLUS'
                )
        .distinct()
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### S05 Check Output

# COMMAND ----------

display(df_score_out.count()) # 50518

# COMMAND ----------

# DBTITLE 1,check score output
display(df_score_out
        .groupBy('propensity_segment_qt')
        .agg(f.countDistinct('fs_acct_id').alias('cnt')
             , f.avg('propensity_score').alias('avg_score')
             , f.avg('TOTAL_BALANCE').alias('avg_bal')
             )
        .withColumn('sum', f.sum('cnt').over(Window.partitionBy()))
        .withColumn('pct', f.col('cnt')/f.col('sum'))
        )

# COMMAND ----------

display(df_score_out)

# COMMAND ----------

# DBTITLE 1,check distinct
display(df_score_out
        .agg(
            f.count('*')
            ,f.countDistinct('fs_acct_id')
        )
        )
