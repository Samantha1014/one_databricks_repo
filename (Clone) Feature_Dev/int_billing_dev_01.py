# Databricks notebook source
# MAGIC %md
# MAGIC ### S00001 Library

# COMMAND ----------

import pyspark
# import os 
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql.functions import from_unixtime,from_utc_timestamp
from pyspark.sql.functions import date_format, last_day, datediff
from pyspark.sql.functions import regexp_replace, last
from pyspark.sql.functions import lag, lead
from pyspark.sql.functions import col, month, dayofmonth, when, add_months
from pyspark.sql import Window

# COMMAND ----------

# %run "../Function"

# COMMAND ----------

dir_global_calendar_meta = 'dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d000_meta/d001_global_cycle_calendar'
df_global_calendar_meta = spark.read.format('delta').load(dir_global_calendar_meta)

# COMMAND ----------

dir_oa_prm = '/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_unit_base/reporting_cycle_type=calendar cycle'
df_oa_prim = spark.read.format('delta').load(dir_oa_prm)

# COMMAND ----------

dir_bill_base = '/mnt/feature-store-dev/dev_users/dev_sc/d100_raw/BILL_BASE'
df_bill_base_stg = spark.read.format('delta').load(dir_bill_base)

# COMMAND ----------

# MAGIC %md
# MAGIC ###S00002 Bill Base Transform

# COMMAND ----------

# DBTITLE 1,Add Transform Column
# filter to include longest duration bill 
df_bill_base_int_01 =( 
    df_bill_base_stg
    #.withColumn('bill_cycle_end_dttm',
    #             f.date_sub(f.col('bill_end_dttm'),1) 
    #             ) # 1st march to 28th feb 
    #.withColumn('bill_cycle_end_month', 
    #            date_format(col('bill_cycle_end_dttm'), 'yyyy-MM')
    #            )
    .withColumn('bill_period',
                 f.datediff(col('bill_end_dttm'), col('bill_start_dttm'))
                 )
    .withColumn('bill_current_charge', 
                f.col('bill_total_due') - f.col('bill_previous_total')
                )
    .withColumn('cnt_bill_per_month', 
                f.count('bill_no').over(Window.partitionBy('fs_acct_id', 'bill_end_month'))
                )
    .withColumn('transfer_bill_exist', 
                f.when( (col('cnt_bill_per_month')>1) & (col('bill_transferred') != 0),1 )
                .otherwise(0)
                )
    .withColumn('longest_duration_bill', 
                f.row_number().over(Window.partitionBy('fs_acct_id', 'bill_end_month')
                                    .orderBy(f.desc('bill_period'))
                                    )
                )
    .withColumn('next_record_previous_total', 
                lead('bill_previous_total', 1,0)
                .over(Window.partitionBy('fs_acct_id')
                      .orderBy('bill_end_dttm'))
                )
    .withColumn('calculated_recvd', 
                f.col('bill_total_due') - f.col('next_record_previous_total')
                ) # calcualted receive will include adjustment + payment 
    #.filter(col('longest_duration_bill') == 1)
    # .withColumn('reporting_date', last_day("bill_cycle_end_dttm"))
)



# COMMAND ----------

display(df_bill_base_int_01
        .filter(col('fs_acct_id')=='468965894'))

# COMMAND ----------

# oa consumer filter to 2021 
df_oa_prim_01 = (
    df_oa_prim
    .select('fs_acct_id')
    .filter(col('reporting_date')>='2021-01-01')
    .distinct()
)

# COMMAND ----------

# narrow down to oa consumer 
df_bill_base_int_02 = (
    df_bill_base_int_01
    .join(df_oa_prim_01, ['fs_acct_id'], 'inner')
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parameter

# COMMAND ----------

vt_reporting_date = '2022-07-31'
vt_param_cycle = 6

# COMMAND ----------

# MAGIC %md
# MAGIC ### Int Layer Stage 1

# COMMAND ----------

df_bill_base_int_03 = (
    df_bill_base_int_02
    .filter(col('bill_end_dttm')<= vt_reporting_date)
    .withColumn('reporting_date', f.lit(vt_reporting_date))
    .withColumn('last_bill_cycle_rnk', 
                f.row_number()
                .over(Window.partitionBy('fs_acct_id')
                      .orderBy(f.desc('bill_end_dttm'))
                      )
                )
    .filter(col('last_bill_cycle_rnk')<= vt_param_cycle )
     .withColumn('bill_cumsum_charge', 
                 f.sum('bill_current_charge')
               .over(Window.partitionBy('fs_acct_id')
                      .orderBy('bill_end_dttm')
                      .rowsBetween(Window.unboundedPreceding, Window.currentRow)
                      )
                )
     .withColumn('bill_cumsum_count', f.count('bill_no')
                 .over(Window.partitionBy('fs_acct_id')
                       .orderBy('bill_end_dttm')
                       .rowsBetween(Window.unboundedPreceding, Window.currentRow)
                       )
                )
    .withColumn('bill_cumsum_avg',
                 f.col('bill_cumsum_charge')/f.col('bill_cumsum_count')
                )
    .withColumn('overdue_days', 
                f.when( # if bill not close or close after reporting date 
                     (col('bill_closed_dttm')=='1970-01-01T12:00:00.000+00:00') |  ( col('bill_closed_dttm') >= col('reporting_date') ),
                                       datediff(col('reporting_date'), col('due_dttm'))
                      )
                .otherwise( # bill close before reporting date 
                    datediff(col('bill_closed_dttm'), col('due_dttm'))
                          )
                )
    .withColumn('flag_late_pay',
                 f.when( col('overdue_days') >0, 1)
                .otherwise(0)
                )
    .withColumn('flag_early_pay', 
                f.when(col('overdue_days')<0,1 )
                .otherwise(0)
                )
    .withColumn('flag_ontime_pay', 
                f.when(col('overdue_days')==0,1 )
                .otherwise(0)
                )
    .withColumn('cnt_overdue_bills', 
                f.when( (col('bill_closed_dttm') >= col('reporting_date'))
                       | (col('bill_closed_dttm') =='1970-01-01T12:00:00.000+00:00')
                       , 1)
                .otherwise(0)                
                )
    .withColumn('cnt_paid_bills', 
                f.when(col('cnt_overdue_bills') ==0,1)
                .otherwise(0)
                 )
    .withColumn('flag_partial_pay',  f.when(col('next_record_previous_total') >0, 1
                                       )
                .otherwise(0)
                )
    .withColumn('flag_full_pay', f.when(col('next_record_previous_total') <=0, 1
                                        )
                .otherwise(0) # flag for full or over pay 
                )
    .withColumn('rto_pay', col('calculated_recvd')/col('bill_total_due')
                ) 
    .drop('bill_poid_id0', 'bill_account_obj_id0')         
   #  .filter(col('fs_acct_id')=='468965894')
)


# COMMAND ----------

display(df_bill_base_int_03.filter(col('fs_acct_id')=='468965894'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Stage 2 Prim Layer - Aggregate

# COMMAND ----------

# MAGIC %md
# MAGIC ### current cycle 

# COMMAND ----------

vt_param_cycle

# COMMAND ----------

# get cycle roll up 


display(df_bill_base_int_03
        .groupBy('reporting_date', 'fs_acct_id')
        .agg(
        f.min('bill_start_dttm').alias('earlist_bill_start_dttm')
         , f.min('bill_end_dttm').alias('earlist_bill_end_dttm')
         ,f.max('bill_start_dttm').alias('last_bill_start_dttm')   
         ,f.max('bill_end_dttm').alias('last_bill_end_dttm')
         ,f.count('bill_no').alias('cnt_no_bill')
         ,f.round(f.avg('bill_current_charge'),2).alias('|avg_bill_charge' + str(vt_param_cycle))
         ,f.round(f.median('bill_current_charge'),2).alias('med_bill_charge' + str(vt_param_cycle))
         ,f.max(
                 f.when(col('last_bill_cycle_rnk')==1, 
                        f.col('bill_total_due'))
                 ).alias('account_total_due')
         ,f.sum('bill_current_charge').alias('total_bill_charge')
         ,f.round(f.avg('overdue_days'),2).alias('avg_od_days')
         ,f.sum('flag_late_pay').alias('cnt_late_pay')
         ,f.sum('flag_early_pay').alias('cnt_early_pay')
         ,f.sum('flag_ontime_pay').alias('cnt_ontime_pay')
         ,f.sum('cnt_overdue_bills').alias('cnt_overdue_bills')
         ,f.sum('cnt_paid_bills').alias('cnt_paid_bills')
         ,f.sum('flag_partial_pay').alias('cnt_partial_pay')
         ,f.sum('flag_full_pay').alias('cnt_full_pay')
         ,f.avg('rto_pay').alias('avg_rto_pay')
        )
        .withColumn('rto_outstanding_amt', f.round(f.col('account_total_due')/ f.col('total_bill_charge'),2))
        .withColumn('rto_outstanding_cnt', f.round( f.col('cnt_overdue_bills')/ f.col('cnt_no_bill'),2))
        .withColumn('pct_pay_ontime', f.round( f.col('cnt_ontime_pay') / f.col('cnt_no_bill'),2 ))
        .withColumn('pct_pay_early', f.round(f.col('cnt_early_pay')/ f.col('cnt_no_bill'),2 ))
        .withColumn('pct_pay_late', f.round(f.col('cnt_late_pay')/ f.col('cnt_no_bill'),2))
        .withColumn('pct_full_pay', f.round(f.col('cnt_full_pay') / f.col('cnt_no_bill'),2 ))
        .withColumn('pct_partial_pay', f.round(f.col('cnt_partial_pay') / f.col('cnt_no_bill'),2 ))
        .filter(col('fs_acct_id') == '468965894')      
        )
 #latest cycle rto_pay 
 # current records 
# suffix bc 

# COMMAND ----------

# MAGIC %md
# MAGIC ### cycle common 

# COMMAND ----------

# MAGIC %md
# MAGIC ### cycle comparsion 

# COMMAND ----------

vt_param_partition = 3 

# COMMAND ----------


