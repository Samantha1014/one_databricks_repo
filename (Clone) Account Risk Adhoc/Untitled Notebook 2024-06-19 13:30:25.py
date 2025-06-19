# Databricks notebook source
import pyspark 
from pyspark.sql import functions as f 

# COMMAND ----------

dbutils.fs.ls('/mnt/feature-store-prod-lab/d600_serving/serv_mobile_oa_consumer/reporting_cycle_type=calendar cycle')

# COMMAND ----------

df_all = spark.read.load('/mnt/feature-store-prod-lab/d600_serving/serv_mobile_oa_consumer/reporting_cycle_type=calendar cycle')

# COMMAND ----------

display(df_all.limit(100))

# COMMAND ----------

df_payment = spark.read.load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_payment_method_cycle_rolling_6/reporting_cycle_type=calendar cycle')

# COMMAND ----------

display(df_payment
        .filter(f.col('reporting_date') == '2024-05-31')
        .select('fs_acct_id'
                , 'reporting_date'
                , 'payment_method_main_code_6cycle'
                , 'payment_method_main_type_6cycle'
                , 'payment_auto_flag_6cycle'
                )
        .distinct()
        .groupBy('payment_method_main_code_6cycle', 'reporting_date')
        .agg(f.countDistinct('fs_acct_id'
                             )
             )
        )
