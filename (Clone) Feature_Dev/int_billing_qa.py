# Databricks notebook source
# MAGIC %md
# MAGIC ### S00001 Library

# COMMAND ----------

import pyspark
import os 
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql.functions import from_unixtime,from_utc_timestamp
from pyspark.sql.functions import date_format
from pyspark.sql.functions import regexp_replace 
from pyspark.sql import Window

# COMMAND ----------

dir_bill_base = '/mnt/feature-store-dev/dev_users/dev_sc/d100_raw/BILL_BASE'
df_bill_base = spark.read.format('delta').load(dir_bill_base)

# COMMAND ----------

dir_oa_base = '/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_unit_base/reporting_cycle_type=calendar cycle'
df_oa_base = spark.read.format('delta').load(dir_oa_base)

# COMMAND ----------

df_oa_base = (
    df_oa_base
    .filter(f.col('reporting_date')>='2021-01-01')
    .select('fs_acct_id')
    .distinct()
)

# COMMAND ----------

# MAGIC %run "./stg_billing_dev"

# COMMAND ----------

dir_oa_consumer_prm = '/mnt/feature-store-prod/d300_primary/d301_mobile_oa_consumer/prm_unit_base/reporting_cycle_type=calendar cycle' 
dir_oa_consumer_int_scc = '/mnt/feature-store-prod/d200_intermediate/d201_mobile_oa_consumer/int_ssc_billing_account/reporting_cycle_type=calendar cycle'
dir_oa_consumer_raw = '/mnt/feature-store-prod/d100_raw/d101_dp_c360/d_billing_account/reporting_cycle_type=rolling cycle'

# COMMAND ----------

# DBTITLE 1,check multi bills per month
 display(df_bill_base_stg_01
        .join(df_oa_base,['fs_acct_id'],'inner')
         .filter(col('longest_duration_bill')!=1) # not the logest duration bill
         .withColumn('transfer_bill', f.when(col('bill_transferred')!=0, 1).otherwise(0)
                     )
         .filter(col('transfer_bill')==0)
         .limit(100)
         ) 
 
display(df_bill_base_stg_01
        .join(df_oa_base,['fs_acct_id'],'inner')
         .filter(col('longest_duration_bill')!=1) # not the logest duration bill
         .withColumn('transfer_bill', f.when(col('bill_transferred')!=0, 1).otherwise(0)
                     )
         .groupBy('transfer_bill')
         .agg(f.countDistinct('fs_acct_id')
             ,f.count('bill_no'))
         ) 
 

# COMMAND ----------

 display(df_bill_base_stg_01.filter(col('fs_acct_id')=='488733691'))
 # 3 bills per month 
 # 1 bill for actual cycle 
 # 1 bill for transfer credit 
 # 1 bill for error bill, bill start and bill end at the same date and bill never close 
