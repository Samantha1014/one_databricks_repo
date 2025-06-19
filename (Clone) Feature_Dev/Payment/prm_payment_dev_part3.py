# Databricks notebook source
# MAGIC %md
# MAGIC ### Payment Part 3
# MAGIC this part calculates the 
# MAGIC 1. payment inteval  variance in the last x cycle 
# MAGIC 2. payment interval avg
# MAGIC 3. median payment interval  
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### S00001 Library

# COMMAND ----------

import pyspark
from pyspark.sql import functions as f
from pyspark.sql.window import Window
from pyspark.sql.functions import col, datediff, lead, avg, var_pop

# COMMAND ----------

# MAGIC %md
# MAGIC ### S00002 Directory

# COMMAND ----------

dir_payment_int_ssc = 'dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d200_intermediate/payment_int_ssc'
df_payment_int_ssc = spark.read.format('delta').load(dir_payment_int_ssc)

# COMMAND ----------

display(df_payment_int_ssc.count())

# COMMAND ----------

display(df_payment_int_ssc.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6 cycle version 

# COMMAND ----------

vt_param_lookback_cycles = 6
vt_param_ssc_reporting_date = '2023-07-31'

# COMMAND ----------


df_prm_pay_days_cycle = (df_payment_int_ssc
        .filter(f.col('item_category') == 'payment')
        .withColumn('next_payment_date', 
                    lead('rec_created_dttm',1)
                    .over(Window.partitionBy('fs_acct_id')
                          .orderBy('rec_created_dttm')))
        .withColumn('days_between_payments', datediff('next_payment_date', 'rec_created_dttm'))
        .groupBy('fs_acct_id')
        .agg(
              f.avg('days_between_payments').alias('pay_days_avg')
             , f.round(f.var_pop('days_between_payments'),2).alias('pay_days_var')
             ,f.median('days_between_payments').alias('pay_days_median')
             )
)


# COMMAND ----------

display(df_prm_pay_days_cycle.filter(f.col('fs_acct_id') =='1001015'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Never Pay Flag

# COMMAND ----------

# directory 
dir_oa_prm = '/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_unit_base/reporting_cycle_type=calendar cycle'
df_oa_prm = spark.read.format('delta').load(dir_oa_prm)
vt_param_ssc_reporting_date = '2023-07-31'

# COMMAND ----------

# oa base and its active period 
df_oa_prm_curr = (
    df_oa_prm
    .filter(f.col('reporting_date')>='2021-01-01')
    .select('reporting_date', 'fs_acct_id', 'fs_cust_id')
    .groupBy('fs_acct_id', 'fs_cust_id')
    .agg(f.min('reporting_date').alias('min_oa_reporting_date')
         , f.max('reporting_date').alias('max_oa_reporting_date')
         )
    .filter(f.col('min_oa_reporting_date') <= vt_param_ssc_reporting_date)
    .filter(f.col('max_oa_reporting_date') >= vt_param_ssc_reporting_date)
)



# COMMAND ----------

 #never pay 
df_oa_cycle = (
    df_oa_prm_curr
    .join(df_payment_int_ssc, ['fs_acct_id'], 'left')
    .withColumn('never_pay_curr', 
                f.when(f.col('item_id').isNull(), 'Y' )
                .otherwise('N') 
                )
    .select('reporting_date', 'fs_acct_id', 'fs_cust_id', 'never_pay_curr')
)
# 516,287 
# 529,165

# COMMAND ----------

display(df_oa_cycle
        .filter( f.col('never_pay_curr') =='Y' )
        #.count()
        .limit(3)
        )
