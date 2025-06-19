# Databricks notebook source
import pyspark
from pyspark import sql 
from pyspark.sql import Window
from pyspark.sql import functions as f
from pyspark.sql.functions import from_unixtime,from_utc_timestamp
from pyspark.sql.functions import date_format

# COMMAND ----------

df_raw_bill = spark.read.format('delta').load('dbfs:/mnt/feature-store-prod-lab/d100_raw/d101_dp_c360/d_billing_account/reporting_cycle_type=rolling cycle')

# bill_delivery_type_desc  to define paper or ebill 

# COMMAND ----------

display(df_raw_bill
        .select('bill_delivery_type_desc', 'billing_account_number', 'bill_payment_method_desc')
        #.distinct()
        .groupBy('bill_delivery_type_desc','bill_payment_method_desc')
        .agg(f.countDistinct('billing_account_number'))
        )

# COMMAND ----------

dbutils.fs.ls('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_payment_method_cycle_rolling_6/reporting_cycle_type=calendar cycle')

# COMMAND ----------

df_payment = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_payment_method_cycle_rolling_6/reporting_cycle_type=calendar cycle')

# COMMAND ----------

display(df_payment
        .filter(f.col('reporting_date') == '2024-05-31')
        .select('fs_cust_id', 'fs_acct_id', 'payment_method_main_type_6cycle')
        .distinct()
        .groupBy('payment_method_main_type_6cycle')
        .agg(f.countDistinct('fs_acct_id'))
        )

# COMMAND ----------

# average days to pay 

dbutils.fs.ls('/mnt/feature-store-prod-lab/d200_staging/d299_src/')
