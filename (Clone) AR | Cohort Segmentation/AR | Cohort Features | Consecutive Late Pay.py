# Databricks notebook source
# MAGIC %md
# MAGIC #### library

# COMMAND ----------

import os
import pyspark
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window
from datetime import datetime, timedelta

# COMMAND ----------

# MAGIC %run "./utility_functions"

# COMMAND ----------

# MAGIC %md
# MAGIC #### directory

# COMMAND ----------

dir_edw_data_parent = "/mnt/prod_edw/raw/cdc"
dir_brm_data_parent = "/mnt/prod_brm/raw/cdc"
dir_fs_data_parent = "/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer"
dir_fs_data_stg = '/mnt/feature-store-prod-lab/d200_staging/d299_src'
dir_fs_data_prm = '/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/'
dir_fs_data_srvc = '/mnt/feature-store-prod-lab/d600_serving/serv_mobile_oa_consumer/'

# COMMAND ----------

# MAGIC %md
# MAGIC #### load data

# COMMAND ----------

df_master = spark.read.format('delta').load(os.path.join(dir_fs_data_srvc,'reporting_cycle_type=calendar cycle'))
df_prm_bill = spark.read.format('delta').load(os.path.join(dir_fs_data_prm, 'prm_bill_cycle_billing_6'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### parameters

# COMMAND ----------

vt_reporting_date = '2024-03-31'
vt_reporting_cycle_type = 'calendar cycle'
vt_param_bill_lookback_billing_cycles = 6
vt_param_bill_late_chronic_threshold = vt_param_bill_lookback_billing_cycles * 2/3
vt_param_bill_late_sporadic_threshold = vt_param_bill_lookback_billing_cycles * 1/3
vt_param_bill_late_mid_threshold = vt_param_bill_lookback_billing_cycles/2 
vt_param_bill_late_length_sequence_threshold = vt_param_bill_late_mid_threshold/3*2

# COMMAND ----------

print(vt_param_bill_late_chronic_threshold)
print(vt_param_bill_late_sporadic_threshold)
print(vt_param_bill_late_mid_threshold)
print(vt_param_bill_late_length_sequence_threshold)

# COMMAND ----------

ls_param_bill_joining_keys = ['fs_acct_id', 'reporting_date']
ls_param_prm_keys = ['fs_cust_id', 'fs_acct_id', 'fs_srvc_id']
ls_reporting_date_keys = ['reporting_date', 'reporting_cycle_type']

# COMMAND ----------

# MAGIC %md
# MAGIC #### transformation 

# COMMAND ----------

# DBTITLE 1,bill base 00
df_base_bill_00_curr = (df_prm_bill
                .filter(
                    (f.col("reporting_date") == vt_reporting_date)
                    & (f.col("reporting_cycle_type") == vt_reporting_cycle_type)
                    & (f.col("bill_end_date") <= vt_reporting_date)
                )
                #.select(ls_param_bill_fields)
                .withColumn(
                    "previous_total_due"
                    , f.lag("total_due", 1).over(
                        Window
                        .partitionBy(*ls_param_bill_joining_keys)
                        .orderBy("bill_end_date")
                    )
                )
                .withColumn(
                    "total_due_delta"
                    , f.col("total_due")/f.col("previous_total_due")
                )
                .withColumn(
                    "current_charge"
                    , f.col('total_due') - f.col('previous_total')
                )
                .withColumn(
                    "previous_current_charge"
                    , f.lag("current_charge", 1).over(
                        Window
                        .partitionBy(*ls_param_bill_joining_keys)
                        .orderBy("bill_end_date")
                    )
                )
                .withColumn(
                    "current_charge_delta"
                    , f.col("current_charge")/f.col("previous_current_charge")
                )
                .filter(
                    (f.col("billing_cycle_finish_flag") == 'Y')
                    & (f.col("billing_cycle_index") <= vt_param_bill_lookback_billing_cycles)
                )
            )

# COMMAND ----------

# DBTITLE 1,bill base 01
df_base_bill_01_curr = (
                df_base_bill_00_curr
                .withColumn(
                    "bill_close_flag"
                    , f.when(
                        (f.col("bill_close_date") <= '1970-01-31')
                        | (f.col("bill_close_date") > vt_reporting_date)
                        , f.lit('N')
                    )
                    .otherwise(f.lit('Y'))
                )
                .withColumn(
                    "bill_overdue_days"
                    , f.when(
                        f.col("bill_close_flag") == 'N'
                        , f.datediff("reporting_date", "bill_due_date")
                    )
                    .otherwise(f.datediff("bill_close_date", "bill_due_date"))
                )
                .withColumn(
                    "bill_payment_timeliness_status"
                    , f.when(
                        (f.col("total_due") <= 0)
                        , f.lit("credit_bill")
                    )
                    .when(
                        (f.col("bill_close_flag") == 'N')
                        , f.lit("miss")
                    )
                    .when(
                        (f.col("bill_overdue_days") > 3)
                        , f.lit("late")
                    )
                    .when(
                        f.col("bill_overdue_days") < 0
                        , f.lit("early")
                    )
                    .when(
                        (f.col("bill_overdue_days") >= 0)
                        & (f.col("bill_overdue_days") <= 3)
                        , f.lit("ontime")
                    )
                    .otherwise(f.lit("unknown"))
                )  
            )

# COMMAND ----------

# DBTITLE 1,check example
display(df_master
        .filter(f.col('reporting_date') == vt_reporting_date)
        .filter(f.col('reporting_cycle_type') == vt_reporting_cycle_type)
        .filter(f.col('bill_cnt_late_tot_6bmnth') >2)
        .select('fs_acct_id', 'fs_cust_id', 'bill_cnt_late_tot_6bmnth')
        )

# COMMAND ----------

# DBTITLE 1,late pay consecutive sequence
df_consecutive_late_pay =(df_base_bill_01_curr
        #.filter(f.col('bill_payment_timeliness_status').isin('late', 'miss'))
        #.filter(f.col('fs_acct_id') == '724563')   
        .select('reporting_date'
                , 'reporting_cycle_type'
                , 'fs_cust_id'
                , 'fs_acct_id'
                , 'bill_no'
                , 'billing_cycle_index'
                , 'bill_due_date'
                , 'bill_close_date'
                , 'total_due'
                , 'previous_total'
                , 'current_charge'
                , 'total_due_delta'
                , 'current_charge_delta'
                , 'bill_close_flag'
                , 'bill_overdue_days'
                , 'bill_payment_timeliness_status'
                )
        .withColumn('_is_late', f.when( f.col('bill_payment_timeliness_status').isin( 'late', 'miss')
                                       , 1
                                       )
                                 .otherwise(0)
        )
        .withColumn('lag_is_late'
                    , f.lag('_is_late').over(
                    Window
                    .partitionBy(*ls_param_bill_joining_keys)
                    .orderBy('bill_due_date')
                    )
        )
        .withColumn('status_change', 
                    f.when(  (f.col('_is_late') != f.col('lag_is_late')) 
                                | f.col('lag_is_late').isNull()
                                , 1
                    )
                     .otherwise(0)                    
                    )
        .withColumn('group_id', f.sum('status_change').over(
                                  Window
                                  .partitionBy(*ls_param_bill_joining_keys)
                                  .orderBy('bill_due_date'))
        )
        .filter(f.col('_is_late') == 1)
        .groupBy(*ls_param_bill_joining_keys)
        .agg(f.countDistinct('group_id').alias('num_consecutive_late_pay')
             #, f.sum('_is_late').alias('cnt_late_pay_6bcycle')
             , f.countDistinct('bill_no').alias('cnt_late_pay_6bcycle')
             , f.avg('bill_overdue_days').alias('avg_overdue_days_6bcycle')
             )
        .withColumn('avg_late_sequence_length',
                    f.when(f.col('num_consecutive_late_pay') == 0,0)
                     .otherwise(f.col('cnt_late_pay_6bcycle') / f.col('num_consecutive_late_pay'))
        )
        .withColumn('payment_behavior_label', 
                      f.when(f.col('cnt_late_pay_6bcycle') >= vt_param_bill_late_chronic_threshold
                             , 'Chronic Late Payer')
                       .when(f.col('cnt_late_pay_6bcycle') <= vt_param_bill_late_sporadic_threshold
                             , 'Sporadic Late Payer')
                       .when(
                              (f.col('cnt_late_pay_6bcycle') == vt_param_bill_late_mid_threshold) 
                                & (f.col('avg_late_sequence_length') > vt_param_bill_late_length_sequence_threshold),
                                'Chronic Late Payer'
                        )
                        .when(
                                (f.col('cnt_late_pay_6bcycle') == vt_param_bill_late_mid_threshold) 
                                & (f.col('avg_late_sequence_length') <= vt_param_bill_late_length_sequence_threshold),
                                'Sporadic Late Payer'
                        )
                        .otherwise('Moderate Late Payer')  
        )
)

# COMMAND ----------

# DBTITLE 1,number for threshold
df_consecutive_late_pay =(df_base_bill_01_curr
        #.filter(f.col('bill_payment_timeliness_status').isin('late', 'miss'))
        #.filter(f.col('fs_acct_id') == '724563')   
        .select('reporting_date'
                , 'reporting_cycle_type'
                , 'fs_cust_id'
                , 'fs_acct_id'
                , 'bill_no'
                , 'billing_cycle_index'
                , 'bill_due_date'
                , 'bill_close_date'
                , 'total_due'
                , 'previous_total'
                , 'current_charge'
                , 'total_due_delta'
                , 'current_charge_delta'
                , 'bill_close_flag'
                , 'bill_overdue_days'
                , 'bill_payment_timeliness_status'
                )
        .withColumn('_is_late', f.when( f.col('bill_payment_timeliness_status').isin( 'late', 'miss')
                                       , 1
                                       )
                                 .otherwise(0)
        )
        .withColumn('lag_is_late'
                    , f.lag('_is_late').over(
                    Window
                    .partitionBy(*ls_param_bill_joining_keys)
                    .orderBy('bill_due_date')
                    )
        )
        .withColumn('status_change', 
                    f.when(  (f.col('_is_late') != f.col('lag_is_late')) 
                                | f.col('lag_is_late').isNull()
                                , 1
                    )
                     .otherwise(0)                    
                    )
        .withColumn('group_id', f.sum('status_change').over(
                                  Window
                                  .partitionBy(*ls_param_bill_joining_keys)
                                  .orderBy('bill_due_date'))
        )
        .filter(f.col('_is_late') == 1)
        .groupBy(*ls_param_bill_joining_keys)
        .agg(f.countDistinct('group_id').alias('num_consecutive_late_pay')
             #, f.sum('_is_late').alias('cnt_late_pay_6bcycle')
             , f.countDistinct('bill_no').alias('cnt_late_pay_6bcycle')
             , f.avg('bill_overdue_days').alias('avg_overdue_days_6bcycle')
             )
        .withColumn('avg_late_sequence_length',
                    f.when(f.col('num_consecutive_late_pay') == 0,0)
                     .otherwise(f.col('cnt_late_pay_6bcycle') / f.col('num_consecutive_late_pay'))
        )
        .withColumn('payment_behavior_label', 
                      f.when(f.col('cnt_late_pay_6bcycle') >= 4
                             , 'Chronic Late Payer')
                       .when(f.col('cnt_late_pay_6bcycle') <= 2
                             , 'Sporadic Late Payer')
                       .when(
                              (f.col('cnt_late_pay_6bcycle') == 3) 
                                & (f.col('avg_late_sequence_length') > 2),
                                'Chronic Late Payer'
                        )
                        .when(
                                (f.col('cnt_late_pay_6bcycle') == 3) 
                                & (f.col('avg_late_sequence_length') <= 2),
                                'Sporadic Late Payer'
                        )
                        .otherwise('Moderate Late Payer')  
        )
)

# COMMAND ----------

print(vt_param_bill_late_chronic_threshold)
print(vt_param_bill_late_sporadic_threshold)
print(vt_param_bill_late_mid_threshold)

print(vt_param_bill_late_mid_threshold)
print(vt_param_bill_late_length_sequence_threshold)

# COMMAND ----------

# DBTITLE 1,unit base left join late pay consec
df_output_curr = (df_master
        .filter(f.col('reporting_cycle_type') == vt_reporting_cycle_type)
        .filter(f.col('reporting_date') == vt_reporting_date)
        .select(*ls_reporting_date_keys, *ls_param_prm_keys)
        .join(df_consecutive_late_pay, ls_param_bill_joining_keys, 'left')
        .withColumn("num_consecutive_late_pay", 
                          f.when(f.col('num_consecutive_late_pay').isNull()
                                 , 0)
                          .otherwise(f.col('num_consecutive_late_pay')))
        .withColumn("cnt_late_pay_6bcycle", 
                          f.when(f.col('cnt_late_pay_6bcycle').isNull()
                                 , 0)
                          .otherwise(f.col('cnt_late_pay_6bcycle')))
        .withColumn("avg_late_sequence_length", 
                          f.when(f.col('avg_late_sequence_length').isNull()
                                 , 0)
                          .otherwise(f.col('avg_late_sequence_length')))
        .withColumn("avg_overdue_days_6bcycle", 
                          f.when(f.col('avg_overdue_days_6bcycle').isNull()
                                 , 0)
                          .otherwise(f.col('avg_overdue_days_6bcycle')))
        .withColumn("payment_behavior_label", 
                          f.when(f.col('payment_behavior_label').isNull()
                                 , "On Time Payer")
                          .otherwise(f.col('payment_behavior_label')))
)        

# COMMAND ----------

display(df_output_curr)

# COMMAND ----------

display(df_output_curr
        .groupBy('cnt_late_pay_6bcycle', 'reporting_date')
        .agg(f.count('*'))
        )

# COMMAND ----------

# MAGIC %md
# MAGIC #### export data

# COMMAND ----------

dbutils.fs.ls('/mnt/ml-lab/dev_users/dev_sc/99_misc/cohort_seg/fea_payment_behavior')

# COMMAND ----------

export_data(
            df = df_output_curr
            , export_path = '/mnt/ml-lab/dev_users/dev_sc/99_misc/cohort_seg/fea_payment_behavior'
            , export_format = "delta"
            , export_mode = "append"
            , flag_overwrite_schema = True
            , flag_dynamic_partition = True
            , ls_dynamic_partition = ["reporting_date"]
        )

# COMMAND ----------

display(df_output_curr
        .groupBy('reporting_date', 'payment_behavior_label')
        .agg(f.countDistinct('fs_acct_id')
             , f.avg('avg_overdue_days_6bcycle')
             , f.count('*')
             )
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Data

# COMMAND ----------

df_test = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/99_misc/cohort_seg/fea_payment_behavior')

# COMMAND ----------

display(df_test
        .groupBy('reporting_date'
                 , 'payment_behavior_label'
                 )
        .agg(f.countDistinct('fs_acct_id')
             , f.count('*').alias('cnt')
             )
        .withColumn('sum', f.sum('cnt').over(Window.partitionBy('reporting_date')))
        .withColumn('pct', f.col('cnt')/ f.col('sum'))
        )

# COMMAND ----------

display(df_test
        .groupBy('cnt_late_pay_6bcycle', 'reporting_date')
        .agg(f.count('*'))
        )

# COMMAND ----------

display(df_consecutive_late_pay
        .groupBy('reporting_date'
                 , 'payment_behavior_label'
                 )
        .agg(f.countDistinct('fs_acct_id')
             , f.count('*').alias('cnt')
             )
        .withColumn('sum', f.sum('cnt').over(Window.partitionBy()))
        .withColumn('pct', f.col('cnt')/ f.col('sum'))
        )

# COMMAND ----------

display(df_consecutive_late_pay
        .groupBy('reporting_date', 'num_consecutive_late_pay', 'cnt_late_pay_6bcycle', 'avg_late_sequence_length'
                 , 'payment_behavior_label'
                 )
        .agg(f.countDistinct('fs_acct_id')
             , f.count('*')
             )
        )

# COMMAND ----------

display(df_consecutive_late_pay
        .filter(f.col('num_consecutive_late_pay') == 2)
        .filter(f.col('cnt_late_pay_6bcycle') == 2)
        )

# COMMAND ----------

display(df_base_bill_01_curr
        #.filter(f.col('bill_payment_timeliness_status').isin('late', 'miss'))
        .filter(f.col('fs_acct_id') == '1103348')   
        .select('reporting_date'
                , 'reporting_cycle_type'
                , 'fs_cust_id'
                , 'fs_acct_id'
                , 'bill_no'
                , 'billing_cycle_index'
                , 'bill_due_date'
                , 'bill_close_date'
                , 'total_due'
                , 'previous_total'
                , 'current_charge'
                , 'total_due_delta'
                , 'current_charge_delta'
                , 'bill_close_flag'
                , 'bill_overdue_days'
                , 'bill_payment_timeliness_status'
                )
        .withColumn('_is_late', f.when( f.col('bill_payment_timeliness_status').isin( 'late', 'miss')
                                       , 1
                                       )
                                 .otherwise(0)
        )
        .withColumn('lag_is_late'
                    , f.lag('_is_late').over(
                    Window
                    .partitionBy('fs_acct_id', 'reporting_date')
                    .orderBy('bill_due_date')
                    )
        )
        .withColumn('status_change', 
                    f.when( (f.col('_is_late') != f.col('lag_is_late')) | f.col('lag_is_late').isNull()
                            , 1)
                     .otherwise(0)                    
                    )
        .withColumn('group_id', f.sum('status_change').over(Window.partitionBy('fs_acct_id', 'reporting_date').orderBy('bill_due_date'))
        ))
