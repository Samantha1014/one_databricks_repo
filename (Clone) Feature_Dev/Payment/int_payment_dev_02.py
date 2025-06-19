# Databricks notebook source
# MAGIC %md
# MAGIC ###S00001 Library

# COMMAND ----------

import pyspark
from pyspark.sql import functions as f
from pyspark.sql.functions import from_unixtime,from_utc_timestamp, date_format
from pyspark.sql.functions import split, col, size, element_at

# COMMAND ----------

# MAGIC %run "../Function"

# COMMAND ----------

# MAGIC %md
# MAGIC ### S00002 Directory

# COMMAND ----------


dir_payment_base = 'dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d100_raw/PAYMENT_BASE/'
df_payment_base = spark.read.format('delta').load(dir_payment_base) 

dir_oa_prm = '/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_unit_base/reporting_cycle_type=calendar cycle'
df_oa_prim = spark.read.format('delta').load(dir_oa_prm)

dir_global_calendar_meta = 'dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d000_meta/d001_global_cycle_calendar'
df_global_calendar_meta = spark.read.format('delta').load(dir_global_calendar_meta)


# COMMAND ----------

# oa consumer filter to 2021 
df_oa_prim_01 = (
    df_oa_prim
    .select('fs_acct_id')
    .filter(f.col('reporting_date')>='2021-01-01')
    .distinct()
)

display(df_oa_prim_01.count()) # 759457

# COMMAND ----------

ls_param_payment_joining_keys = ['fs_acct_id']
ls_param_payment_int_fields = [
'fs_acct_id', 
'item_id',
'rec_created_dttm', 
'rec_mod_dttm',
'rec_effective_dttm',
'item_amount',
'rec_created_month',
'onenz_auto_pay',
'payment_method',
'item_category'] 


# COMMAND ----------

# inner join with OA consumer to narrow down 
df_payment_int = (
    df_payment_base
    .join(df_oa_prim_01, ls_param_payment_joining_keys, 'inner')
    .withColumn(
        'onenz_auto_pay'
        , f.when(f.col('payment_event_type').isin('/event/billing/payment/dd', '/event/billing/payment/cc' ), 'Y')
        .otherwise('N')
    )
    .withColumn('payment_method', element_at(split(f.col('payment_event_type'), '/'), size(split(f.col('payment_event_type'), '/'))))
    .withColumn(
        'item_category'
        , f.when(f.col('item_poid_type')=='/item/adjustment', f.lit('adjustment'))
        .when( f.col('payment_event_type') == '/event/billing/payment/failed', f.lit('fail payment'))
                #.when( f.col('item_poid_type').isNull(), f.lit('no payment'))
        .when( f.col('item_poid_type') == '/item/payment' , f.lit('payment'))
        .otherwise(f.lit('other'))
    )
    .withColumnRenamed('item_poid_id0', 'item_id')
    .select(ls_param_payment_int_fields)
         #.fillna(value = 'other', subset = ['payment_method'])
         # use right join since there is never payer 
)

# COMMAND ----------

vt_param_ssc_reporting_date = '2023-07-31'
vt_param_ssc_reporting_cycle_type = 'calendar cycle'
vt_param_lookback_cycles = 6
vt_param_lookback_cycle_unit_type = "months"
vt_param_lookback_units_per_cycle = 1

# COMMAND ----------

df_param_partition_curr = get_lookback_cycle_meta(
    df_calendar=df_global_calendar_meta
    , vt_param_date=vt_param_ssc_reporting_date
    , vt_param_lookback_cycles=vt_param_lookback_cycles
    , vt_param_lookback_cycle_unit_type=vt_param_lookback_cycle_unit_type
    , vt_param_lookback_units_per_cycle=vt_param_lookback_units_per_cycle
)

display(df_param_partition_curr)

df_param_partition_summary_curr = (
    df_param_partition_curr
    .agg(
        f.min("reporting_start_date").alias("date_min")
        , f.max("reporting_end_date").alias("date_max")
        # , f.max('reporting_start_date').alias('reporting_start_date')
    )
)

vt_param_partition_start_date = df_param_partition_summary_curr.collect()[0].date_min
vt_param_partition_end_date = df_param_partition_summary_curr.collect()[0].date_max
# vt_param_cycle_start_date = df_param_partition_summary_curr.collect()[0].reporting_start_date 

# COMMAND ----------

print(vt_param_partition_start_date)
print(vt_param_partition_end_date)

# COMMAND ----------

df_payment_int_curr =  (df_payment_int
                       .withColumn('reporting_date', f.lit(vt_param_ssc_reporting_date))
                       .filter(f.col('rec_created_dttm') <= vt_param_partition_end_date)
                       .filter(f.col('rec_created_dttm')>= vt_param_partition_start_date)
                       .select('reporting_date', *ls_param_payment_int_fields)
                    #    .select(f.col('reporting_date'), *[f.col(c) for c in ls_param_payment_int_fields])
                       )

# COMMAND ----------

display(df_payment_int_curr.count()) 
3,822,504

# COMMAND ----------

df_output_curr = (
    df_payment_int_curr
)

# COMMAND ----------

# dbutils.fs.rm('dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d200_intermediate/payment_int_ssc', True)

# COMMAND ----------

(
    df_output_curr
    .write
    .format("delta")
    .mode("overwrite")
    .partitionBy("rec_created_month")
    .option("overwriteSchema", "true")
    .option("partitionOverwriteMode", "dynamic")
    .save("dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d200_intermediate/payment_int_ssc")
)
