# Databricks notebook source
# MAGIC %md
# MAGIC ###S00001 Library

# COMMAND ----------

import pyspark
from pyspark.sql import functions as f
from pyspark.sql.functions import from_unixtime,from_utc_timestamp, date_format
from pyspark.sql.functions import split, col, size

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
        f.min("partition_date").alias("date_min")
        , f.max("partition_date").alias("date_max")
        , f.max('reporting_start_date').alias('reporting_start_date')
    )
)

vt_param_partition_start_date = df_param_partition_summary_curr.collect()[0].date_min
vt_param_partition_end_date = df_param_partition_summary_curr.collect()[0].date_max
vt_param_cycle_start_date = df_param_partition_summary_curr.collect()[0].reporting_start_date 

# COMMAND ----------

ls_param_usg_joining_keys = ['fs_acct_id']

# COMMAND ----------

# right join with OA consumer to narrow down 
df_payment_int = (df_payment_base
                  .join(df_oa_prim_01, ls_param_usg_joining_keys, 'right')
        .withColumn('onenz_auto_pay', f.when(f.col('payment_event_type').isin('/event/billing/payment/dd', '/event/billing/payment/cc' ), 'Y')
                                        .otherwise('N')
                    )
        .withColumn('payment_method',  split(f.col('payment_event_type'), "/").getItem(size(split(f.col('payment_event_type'), "/")) - 1))
        .withColumn('payment_category', 
                    f.when(f.col('item_poid_type')=='/item/adjustment', f.lit('adjustment'))
                    .when( f.col('payment_event_type') == '/event/billing/payment/failed', f.lit('fail payment'))
                    .when( f.col('item_poid_type').isNull(), f.lit('no payment'))
                    .when( f.col('item_poid_type') == '/item/payment' , f.lit('payment'))
                    .otherwise(f.lit('other'))
                    )
        # .fillna(value = 'other', subset = ['payment_method'])
         # use right join since there is never payer 
        )

# COMMAND ----------

display(df_payment_int.limit(10))

# COMMAND ----------

# DBTITLE 1,adjustment
# adjustment records only 
df_adjustment_int = (
    df_payment_int
    .filter(col('item_poid_type')=='/item/adjustment')
)

# COMMAND ----------

display(df_adjustment_int.limit(10))

# COMMAND ----------

df_adjustment_int_01 = (
    df_adjustment_int
    # .withColumn('reporting_date', f.lit(vt_param_ssc_reporting_date))
    .filter(f.col('rec_created_dttm') >= vt_param_cycle_start_date)
    .filter(f.col('rec_created_dttm') <= vt_param_ssc_reporting_date )
    .groupBy('fs_acct_id')
    .agg(f.count('item_poid_id0').alias('ajustment_cnt_curr')
         , f.sum('item_amount').alias('adjustment_totl_curr')
         )
    .withColumn('reporting_date', f.lit(vt_param_ssc_reporting_date))
)

# COMMAND ----------

display(df_adjustment_int_01.limit(10))

# COMMAND ----------

# DBTITLE 1,fail payment
df_fail_payment_int = (
     df_payment_int
    .filter(f.col('item_poid_type')!='/item/adjustment')
    .filter(f.col('payment_event_type') == '/event/billing/payment/failed')
)

# display(df_fail_payment_int.count())

# COMMAND ----------

df_fail_payment_int_01 = (
    df_fail_payment_int
    .filter(f.col('rec_created_dttm') >= vt_param_cycle_start_date)
    .filter(f.col('rec_created_dttm') <= vt_param_ssc_reporting_date )
    .groupBy('fs_acct_id')
    .agg(f.count('item_poid_id0').alias('fail_pay_cnt_curr')
         )
    .withColumn('reporting_date', f.lit(vt_param_ssc_reporting_date))
)

# COMMAND ----------

df_fail_payment_cycle_01 = (
    df_fail_payment_int
    .filter(f.col('rec_created_dttm') >= vt_param_partition_start_date)
    .filter(f.col('rec_created_dttm') <= vt_param_partition_end_date )
    .groupBy('fs_acct_id')
    .agg(f.count('item_poid_id0').alias(f'fail_pay_cnt_c{vt_param_lookback_cycles}')
         )
    .withColumn('reporting_date', f.lit(vt_param_ssc_reporting_date))
)

# COMMAND ----------

display(df_fail_payment_cycle_01.limit(3))

# COMMAND ----------

# payment records only 
df_payment_int_01 = (
    df_payment_int
    .filter( (f.col('item_poid_type')!='/item/adjustment') | f.col('item_poid_type').isNull() ) # or null 
    .filter( (f.col('payment_event_type') != '/event/billing/payment/failed')  | f.col('item_poid_type').isNull())
    )

# display(df_payment_int_01.count()) # 22038079

# COMMAND ----------

display(df_payment_int_01
        .select('fs_acct_id', 'item_poid_id0')
        .filter(f.col('item_poid_id0').isNull())
        .distinct()
        .count()
        ) 

# COMMAND ----------

display(df_payment_int_01.filter(f.col('fs_acct_id') =='1012777'))

# COMMAND ----------

display(
    df_payment_int_01
    .groupBy('rec_created_month')
    .agg(f.count('fs_acct_id'))
    # .filter(f.col('method_cnt')>1)
     #.filter(f.col('rec_created_month') == '2023-07')
    # .count()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### current attributes 

# COMMAND ----------

df_payment_int_01_latest = (
    df_payment_int_01
    .filter(f.col('rec_created_dttm') <= vt_param_ssc_reporting_date)
    .withColumn('last_payment_method_rnk', f.row_number().over(Window.partitionBy('fs_acct_id').orderBy(f.desc('rec_created_dttm'))))
    .filter(f.col('last_payment_method_rnk') ==1)
    .select('fs_acct_id', 'rec_created_dttm','item_amount', 'payment_method') 
    .withColumnRenamed('rec_created_dttm', 'last_payment_date')
    .withColumnRenamed('item_amount', 'last_payment_amount')
    .withColumnRenamed('payment_method', 'last_payment_method')
    # .filter(f.col('fs_acct_id') =='405980390')    
)
   

# COMMAND ----------

display(df_payment_int_01_latest.filter(f.col('fs_acct_id') == '1012777'))

# COMMAND ----------

df_payment_int_02 = (
    df_payment_int_01
    .filter(f.col('rec_created_dttm') >= vt_param_cycle_start_date )
    .filter(f.col('rec_created_dttm') <= vt_param_ssc_reporting_date)
    .groupBy('fs_acct_id')
    .agg(
        f.count('item_poid_id0').alias('payment_cnt_curr')
         , f.round(f.sum('item_amount'),2).alias('payment_total_curr')
         )
    .withColumn('payment_avg_curr', f.round(f.col('payment_total_curr')/ f.col('payment_cnt_curr'),2))
    .join(df_payment_int_01_latest, ls_param_usg_joining_keys, 'right')
     # include those who has not pay at current cycle 
   #  .filter(f.col('fs_acct_id') == '405980390')
)
  

# COMMAND ----------

display(df_payment_int_02.filter(f.col('fs_acct_id')=='1012777'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### cycle version 

# COMMAND ----------

display(df_payment_int_01.filter(f.col('fs_acct_id') == '1012777'))

# COMMAND ----------

# 6 cycle sum 
df_payment_int_cycle_01 = (df_payment_int_01
        .withColumn('reporting_date', f.lit(vt_param_ssc_reporting_date))
        .filter(f.col('rec_created_dttm')>= vt_param_partition_start_date)
        .filter(f.col('rec_created_dttm')<= vt_param_partition_end_date)
        .groupBy('reporting_date', 'fs_acct_id')
        .agg(
                f.count(f.col('item_poid_id0')).alias(f'payment_cnt_{vt_param_lookback_cycles}_cycle' )
                ,f.round(f.sum('item_amount'),2).alias(f'payment_sum_{vt_param_lookback_cycles}_cycle')
                , f.round(f.avg('item_amount'),2).alias(f'payment_avg_{vt_param_lookback_cycles}_cycle')
                , f.sum('onenz_auto_pay').alias(f'auto_pay_cnt_{vt_param_lookback_cycles}_cycle')
        )
        #.withColumn('partition_start_date', f.lit(vt_param_partition_start_date))
        #.withColumn('partition_end_date', f.lit(vt_param_partition_end_date))
        .withColumn(f'auto_pay_flag_{vt_param_lookback_cycles}_cycle', 
                    f.when( f.col(f'payment_cnt_{vt_param_lookback_cycles}_cycle') == f.col(f'auto_pay_cnt_{vt_param_lookback_cycles}_cycle'), 
                           1)
                    .otherwise(0)
        )
        .join(df_payment_int_02, ls_param_usg_joining_keys, 'right')
)
        

# COMMAND ----------

display(df_payment_int_cycle_01.count()) # 656674

# COMMAND ----------

# 6 cycle most frequent payment method 
df_payment_int_cycle_02 =  (df_payment_int_01
        .filter(f.col('rec_created_dttm')>= vt_param_partition_start_date)
        .filter(f.col('rec_created_dttm')<= vt_param_partition_end_date)
        .groupBy('fs_acct_id', 'payment_method')
        .agg(
                f.count(f.col('item_poid_id0')).alias(f'payment_method_cnt_{vt_param_lookback_cycles}_cycle')
        )
     .orderBy(f.desc(f'payment_method_cnt_{vt_param_lookback_cycles}_cycle'))
     .groupBy('fs_acct_id')
     .agg(
         f.first('payment_method').alias('most_freq_payment_method')
     )
      .join(df_payment_int_cycle_01, ls_param_usg_joining_keys, 'right')
)


# COMMAND ----------

display(df_payment_int_cycle_02
        .filter(f.col('last_payment_date').isNull()
        )
)

# COMMAND ----------

display(df_payment_int_cycle_02
        .filter(f.col('most_freq_payment_method').isNull())
         #.count()  #102,529
        .groupBy(date_format(f.col('last_payment_date'), 'yyyy-MM'))
        .agg(
            f.count('fs_acct_id').alias('cnt')
        )
        .withColumn('sum', f.sum('cnt').over(Window.partitionBy()))
     .withColumn('pct', f.col('cnt')/f.col('sum'))
) # 102529

# COMMAND ----------

display(df_payment_int_cycle_02.filter(f.col('fs_acct_id') == '1012777'))

# COMMAND ----------

display(df_payment_int_cycle_01)

# COMMAND ----------

display(df_payment_int_cycle_01
        .groupBy('payment_cnt_6_cycle')
        .agg(f.count('fs_acct_id').alias('cnt'))
        #.withColumn('sum', f.sum('cnt'))
       # .withColumn('pct', f.col('cnt')/ f.col('sum'))
        )

# COMMAND ----------

df_payment_int_03 = (df_payment_int_01
        .withColumn('reporting_date', f.lit(vt_param_ssc_reporting_date))
        .filter(f.col('rec_created_dttm')>= vt_param_partition_start_date)
        .filter(f.col('rec_created_dttm')<= vt_param_partition_end_date)
        .filter(f.col('payment_event_type')== '/event/billing/payment/failed')
        # .filter(f.col('fs_acct_id') == '484052416')
        .groupBy('reporting_date', 'fs_acct_id')
        .agg(
                f.count('item_poid_id0').alias(f'fail_payment_cnt_{vt_param_lookback_cycles}_cycle' )
                , f.max('rec_created_dttm').alias('last_fail_payment_date')
                 #,f.round(f.sum('item_amount'),2).alias(f'fail_payment_sum_{vt_param_lookback_cycles}_cycle')
                # , f.round(f.avg('item_amount'),2).alias(f'fail_payment_avg_{vt_param_lookback_cycles}_cycle')
        )
)

# COMMAND ----------

display(df_payment_int_01
        .withColumn('reporting_date', f.lit(vt_cycle_end_date))
        .filter(f.col('rec_created_dttm')>= vt_param_partition_start_date)
        .filter(f.col('rec_created_dttm')<= vt_param_partition_end_date)
        # .filter(f.col('payment_event_type')!= '/event/billing/payment/failed')
        .filter(f.col('fs_acct_id') == '484052416')
        .groupBy('reporting_date', 'fs_acct_id')
        .agg(
                f.count(
                        f.when(f.col('payment_event_type') != '/event/billing/payment/failed',
                        f.col('item_poid_id0'))).alias('payment_cnt_6_cycle')
                , f.sum('item_amount').alias('payment_sum_6_cycle')
                , f.count(
                        f.when(f.col('payment_event_type') == '/event/billing/payment/failed', f.col('item_poid_id0'))
                ).alias('payment_fail_cnt_6_cycle')
        )
        .withColumn('payment_avg_6_cycle', f.col('payment_sum_6_cycle')/ f.col('payment_cnt_6_cycle') )
        
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### cycle compare

# COMMAND ----------

display(df_test.limit(10))
