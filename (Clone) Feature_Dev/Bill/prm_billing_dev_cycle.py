# Databricks notebook source
# MAGIC %md
# MAGIC ### S00001 Library

# COMMAND ----------

import pyspark
import os 
from pyspark import sql
from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.sql.functions import from_unixtime,from_utc_timestamp
from pyspark.sql.functions import date_format, last_day, datediff
from pyspark.sql.functions import regexp_replace, last
from pyspark.sql.functions import lag, lead
from pyspark.sql.functions import col, month, dayofmonth, when, add_months
from pyspark.sql import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ### S00002 Directory

# COMMAND ----------

dir_oa_prm = '/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_unit_base/reporting_cycle_type=calendar cycle'
df_oa_prm = spark.read.format('delta').load(dir_oa_prm)

# COMMAND ----------

dir_bill_int = 'dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d200_intermediate/bill_int_scc'
df_bill_int = spark.read.format('delta').load(dir_bill_int)
display(df_bill_int.count()) # 3,051,818

# COMMAND ----------

# MAGIC %md
# MAGIC ###S00003 Development 

# COMMAND ----------

# DBTITLE 1,parameter
vt_param_ssc_reporting_date = '2023-07-31'
vt_param_cycle = 6
vt_partition_cycle = 3

# COMMAND ----------

# DBTITLE 1,unit base
df_oa_prm_curr = (
    df_oa_prm
    .filter(f.col('reporting_date')==vt_param_ssc_reporting_date)
    .select('reporting_date', 'fs_acct_id', 'fs_cust_id')
    .distinct()
)

# COMMAND ----------

display(df_bill_int.limit(10))

# COMMAND ----------

# DBTITLE 1,inner join to unit base
df_bill_prm = (
    df_oa_prm_curr.alias('a')
    .join(df_bill_int.alias('b'), ['fs_acct_id'], 'inner')
    .select('b.*')
)


# COMMAND ----------

# DBTITLE 1,Add Transform Column
df_bill_prm_01 = (
    df_bill_prm
    .withColumn('bill_period', 
                f.datediff(f.col('bill_end_dttm'), f.col('bill_start_dttm'))
                )
    .withColumn('bill_current_charge', 
                f.col('bill_total_due') - f.col('bill_previous_total')
                )
    .withColumn(
        'next_record_previous_total', 
                lead('bill_previous_total', 1)
                .over(Window.partitionBy('fs_acct_id')
                      .orderBy('bill_end_dttm'))
                )
    .withColumn('calculated_recvd', 
                f.col('bill_total_due') - f.col('next_record_previous_total')
                )
                )
    

# COMMAND ----------

display(df_bill_prm_01
        .filter(f.col('fs_acct_id')=='468965894'))


# COMMAND ----------

# MAGIC %md
# MAGIC ### 6 Cycle Features

# COMMAND ----------

# DBTITLE 1,Adding Helper Flag
# due date before reporting date for complete bill cycle 
df_bill_prm_02 = (
    df_bill_prm_01
    # .filter(f.col('last_bill_cycle_rnk') ==1)
        .withColumn(
                'overdue_days', 
                f.when( # if bill not close or close after reporting date 
                     (f.col('bill_closed_dttm')=='1970-01-01T12:00:00.000+00:00') | 
                      (  (f.col('bill_closed_dttm') > f.col('reporting_date') ) ) ,
                                       datediff(f.col('reporting_date'), f.col('bill_due_dttm'))
                      )
                .otherwise( # bill close before reporting date 
                    datediff(f.col('bill_closed_dttm'), f.col('bill_due_dttm'))
                          )
                    )
        .withColumn('flag_pay_bills', 
                    f.when( (f.col('bill_closed_dttm')=='1970-01-01T12:00:00.000+00:00') | 
                      ( (f.col('bill_closed_dttm') > f.col('reporting_date') ) )
                      , 'overdue'
                    ).otherwise('paid')
                    )
         .withColumn('flag_pay_time', 
                    f.when( (f.col('overdue_days') >0) 
                           & (f.col('flag_pay_bills') == 'overdue'),
                            'miss') # if overdue bill means payment is missed 
                    .when( (f.col('overdue_days') >0) 
                           & (f.col('flag_pay_bills') == 'paid'), 
                            'late')
                    .when(f.col('overdue_days')<0, 'early' )
                    .when(f.col('overdue_days') ==0, 'ontime')
                    )
        .withColumn('flag_pay_status',  
                    f.when(f.col('flag_pay_bills') =='paid', 'full_paid') # if pay already, means bill is closed, then it is paid in full 
                    .when(f.col('calculated_recvd') ==0, 'no_pay') # if calculated receivd is 0, then means no payment made 
                    .when(f.col('next_record_previous_total') >0, 'partial_paid')
                    .when(f.col('next_record_previous_total') ==0, 'full_paid' )
                    .when(f.col('next_record_previous_total') <0, 'over_paid')
                ) 
)


# COMMAND ----------

# DBTITLE 1,Overview for Helper flag
display(df_bill_prm_02
        .groupBy('flag_pay_status','flag_pay_bills', 'flag_pay_time' )
        .agg(f.countDistinct('fs_acct_id')
             , f.count('bill_no')
             )
        )

# COMMAND ----------

# DBTITLE 1,Check Edge Case Proportion
display(df_bill_prm_02
        .filter(f.col('flag_pay_status') =='over_paid'
                )
        .filter(f.col('flag_pay_bills')=='overdue')
        .filter(f.col('flag_pay_time') =='miss')
        #.count()
        )

# COMMAND ----------

# DBTITLE 1,Check Edge Case
display(df_bill_prm_02.filter(f.col('fs_acct_id')
                              =='452243987'
                              )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pivot on flag pay bills

# COMMAND ----------

# DBTITLE 1,pivot for paid vs. overdue
df_bill_prm_03 = (df_bill_prm_02
        .groupBy('fs_acct_id', 'reporting_date' )
        .pivot('flag_pay_bills' )
        .agg(
            f.count('bill_no').alias('bill_cnt')
             , f.round(f.sum('bill_current_charge'),2).alias('bill_charge')
             , f.collect_list('bill_no').alias('bill_list')
             )
        .fillna(0)
        .withColumn('bill_cnt_total',
                     f.col('paid_bill_cnt') + f.col('overdue_bill_cnt')
                     )
        .withColumn('bill_charge_total',  
                    f.col('paid_bill_charge') + f.col('overdue_bill_charge')
                    )
        .withColumn('bill_charge_avg', 
                    f.round(f.col('bill_charge_total')/f.col('bill_cnt_total'),2)
                    )
        .withColumn('overdue_amount_rto', 
                    f.round( f.col('overdue_bill_charge') / f.col('bill_charge_total') ,2)
                    )
        .withColumn('overdue_cnt_rto', 
                     f.round( f.col('overdue_bill_cnt') / f.col('bill_cnt_total'), 2)
                    )
        )

# COMMAND ----------

# DBTITLE 1,rename function
def rename_columns_with_suffix(df: DataFrame, suffix: str
                               )-> DataFrame:
    df_out = (
        df.
        select(
           *[f.col(col_name).alias(col_name if col_name in ['fs_acct_id', 'reporting_date'] else f"{col_name}{suffix}")
            for col_name in df.columns 
           ]
        )
    )
    return df_out

# COMMAND ----------

# DBTITLE 1,get suffix
suffix = '_'+ str(vt_param_cycle)+ 'cycle'

# COMMAND ----------

# DBTITLE 1,rename with suffix
df_bill_prm_03 = rename_columns_with_suffix(df_bill_prm_03, suffix)

# COMMAND ----------

# DBTITLE 1,Check Example
display(df_bill_prm_02.filter(f.col('fs_acct_id')==482028945))

# COMMAND ----------

# DBTITLE 1,Check Case with Credit Overdue
display(df_bill_prm_03
        .filter(f.col('overdue_bill_charge') <0 )
) # around 1.9K has credit bills 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pivot on flag_pay_time

# COMMAND ----------

df_bill_prm_04 = (df_bill_prm_02
        .withColumn('bill_cnt_total', 
                    f.count('bill_no')
                    .over(Window.partitionBy('fs_acct_id')
                          )
                    )         
        .groupBy('fs_acct_id', 'reporting_date' )
        .pivot('flag_pay_time' )
        .agg(
            f.count('bill_no').alias('bill_cnt')
            , f.round(
                f.count('bill_no') / f.max('bill_cnt_total'),2
                ).alias('bill_cnt_pct')
        )
        .fillna(0)
)

# COMMAND ----------

# DBTITLE 1,rename with suffix
df_bill_prm_04 = rename_columns_with_suffix(df_bill_prm_04, suffix)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pivot on flag_pay_status  

# COMMAND ----------

df_bill_prm_05 = (df_bill_prm_02 
        .withColumn('bill_cnt_total', 
                    f.count('bill_no')
                    .over(Window.partitionBy('fs_acct_id')
                          )
                    )               
        .groupBy('fs_acct_id', 'bill_cnt_total', 'reporting_date')
        .pivot('flag_pay_status')
        .agg(
                f.count('bill_no').alias('bill_cnt')
                , f.round(
                        f.count('bill_no')/ f.max('bill_cnt_total')
                        ,2)
                .alias('bill_cnt_pct')
                )
        #.withColumn('')
        .fillna(0)
        # .withColumnRenamed('null_bill_cnt', 'unknown_bill_cnt')
        )

# COMMAND ----------

# DBTITLE 1,rename with suffix
df_bill_prm_05 = rename_columns_with_suffix(df_bill_prm_05, suffix)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Common 6 Cycle Features

# COMMAND ----------

df_bill_prm_06 = (df_bill_prm_02
        .groupBy('fs_acct_id', 'reporting_date')
        .agg(
            f.round(f.avg('overdue_days'),2)
             .alias(f'bill_overdue_days_avg_{vt_param_cycle}cycle')
             , f.median('overdue_days')
             .alias(f'bill_overdue_days_med_{vt_param_cycle}cycle')
             , f.round(f.median('bill_current_charge'),2)
             .alias(f'bill_total_charge_med_{vt_param_cycle}cycle')
             , f.round(f.avg('bill_period'),2)
             .alias(f'bill_period_avg_{vt_param_cycle}cycle')
             ) # 
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cycle Compare Trend

# COMMAND ----------

# MAGIC %md
# MAGIC #### compare late payment rate 

# COMMAND ----------

vt_partition_cycle = 3

# COMMAND ----------

display(df_bill_prm_02
        .withColumn('max_bill_cycle', 
                    f.max('last_bill_cycle_rnk')
                    .over(Window.partitionBy('fs_acct_id'))
                    )
        .groupBy('max_bill_cycle')
        .agg(f.countDistinct('fs_acct_id'))
        )


# COMMAND ----------

# DBTITLE 1,LPT
# there are still 9% of the accounts with has less than 6 bills -new joiners 
# to compensate this, calculate the occurance rate of late pay within the available cycle (even if it is less than 3)

df_bill_prm_first_3= (df_bill_prm_02
        .withColumn('bill_rnk_asc', f.row_number()
                    .over(Window.partitionBy('fs_acct_id').orderBy('bill_due_dttm'))
                    )
        .filter(f.col('bill_rnk_asc') <= f.lit(vt_partition_cycle))
        .groupBy('fs_acct_id')
        .agg(
            f.round(
                f.sum(f.when(f.col('flag_pay_time') == 'late',1)
                        .otherwise(0)
                     ) / 
            f.max('last_bill_cycle_rnk'),2
            )
            .alias(f'late_payment_rate_first_{vt_partition_cycle}cycle')
        )
        )

df_bill_prm_last_3= (df_bill_prm_02
                      .filter(f.col('last_bill_cycle_rnk') <=f.lit(vt_partition_cycle))
                      .groupBy('fs_acct_id')
                      .agg(
                            f.round(
                             f.sum(f.when(f.col('flag_pay_time') == 'late',1)
                                .otherwise(0)
                                  ) / 
                            f.max('last_bill_cycle_rnk'),2
                                  )
                                .alias(f'late_payment_rate_last_{vt_partition_cycle}cycle')
                            )
                      )
        

df_bill_prm_compare_01 = (
    df_bill_prm_first_3
    .join(df_bill_prm_last_3, ['fs_acct_id'], 'inner')
    .withColumn(f'late_payment_rate_change_{vt_param_cycle}cycle', 
                f.round(
                    f.col('late_payment_rate_last_3cycle') - f.col('late_payment_rate_first_3cycle')
                    ,2)
                )
)

del df_bill_prm_last_3
del df_bill_prm_first_3

# COMMAND ----------

display(df_bill_prm_compare_01)

# COMMAND ----------

# DBTITLE 1,check cases
display(df_bill_prm_02.filter(f.col('fs_acct_id') ==346819763))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Combine Together

# COMMAND ----------

# DBTITLE 1,join together
display(
    df_bill_prm_03
    .join(df_bill_prm_06, ['fs_acct_id', 'reporting_date'], 'inner')
    .join(df_bill_prm_04, ['fs_acct_id', 'reporting_date'], 'inner')
    .join(df_bill_prm_05, ['fs_acct_id', 'reporting_date'], 'inner')
)

# COMMAND ----------

# DBTITLE 1,Test Case
display(df_bill_prm_02
        .filter(f.col('fs_acct_id') ==421509313)
        )
        # bill start in 2019 and end in 2023, has long bill period, but in bill image, this bill only last for 1 months 

# COMMAND ----------

# DBTITLE 1,Check Count
display(df_bill_prm_02.count()) # 2980630
display(df_bill_prm_02
        .agg(f.countDistinct('fs_acct_id'))
        )
display(df_bill_prm_03.count()) # 520672
