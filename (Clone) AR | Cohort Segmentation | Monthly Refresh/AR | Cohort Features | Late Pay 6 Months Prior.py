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

# MAGIC %run "../utility_functions"

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

#vt_reporting_date = '2024-03-31'
vt_reporting_date_lookback_months = 6 
vt_reporting_cycle_type = 'calendar cycle'
vt_param_bill_lookback_billing_cycles = 6
# vt_param_bill_late_chronic_threshold = vt_param_bill_lookback_billing_cycles * 2/3
# vt_param_bill_late_sporadic_threshold = vt_param_bill_lookback_billing_cycles * 1/3
# vt_param_bill_late_mid_threshold = vt_param_bill_lookback_billing_cycles/2 
# vt_param_bill_late_length_sequence_threshold = vt_param_bill_late_mid_threshold/3*2

# COMMAND ----------

ls_param_bill_joining_keys = ['fs_acct_id', 'reporting_date']
ls_param_prm_keys = ['fs_cust_id', 'fs_acct_id', 'fs_srvc_id']
ls_reporting_date_keys = ['reporting_date', 'reporting_cycle_type']

# COMMAND ----------

#ls_reporting_date = [ '2024-04-30', '2024-05-31', '2024-06-30', '2024-07-31', '2024-08-31', '2024-09-30'] 
ls_reporting_date = [ '2024-10-31'] 
df_reporting_dates = spark.createDataFrame([(d,) for d in ls_reporting_date], ["snapshot_date"])

# COMMAND ----------

display(df_reporting_dates)

# COMMAND ----------

# MAGIC %md
# MAGIC #### transformation 

# COMMAND ----------

# DBTITLE 1,bill base 00
df_base_bill_00_curr = (df_prm_bill
                .filter(f.col('reporting_cycle_type') == vt_reporting_cycle_type)
                .crossJoin(df_reporting_dates)
                .filter(
                    (f.col("reporting_date") == 
                     f.last_day(f.add_months(f.col('snapshot_date'), -6)))  # 6 months prior's reporting date 
                    & (f.col("bill_end_date") <= f.col('snapshot_date'))
                )
                .withColumnRenamed('snapshot_date', 'reporting_date_curr')
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

display(df_base_bill_00_curr.limit(10)
        )

# COMMAND ----------

display(df_base_bill_00_curr
        .groupBy('reporting_date', 'reporting_date_curr')
        .agg(f.countDistinct('fs_acct_id')
             , f.count('*')
             )
        )

# COMMAND ----------

# DBTITLE 1,bill base 01
df_base_bill_01_curr = (
                df_base_bill_00_curr
                #.drop('snapshot_date')
                #.crossJoin(df_reporting_dates)
                .withColumn(
                    "bill_close_flag"
                    , f.when(
                        (f.col("bill_close_date") <= '1970-01-31')
                        | (f.col("bill_close_date") > f.col('reporting_date'))
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

display(df_base_bill_01_curr
        .groupBy('reporting_date', 'reporting_date_curr')
        .agg(f.countDistinct('fs_acct_id')
             )
        )

# COMMAND ----------

# DBTITLE 1,late pay 6 months prior for 6 cycle
df_late_pay_6mp = (df_base_bill_01_curr
        #.filter(f.col('bill_payment_timeliness_status').isin('late', 'miss'))
        #.filter(f.col('fs_acct_id') == '724563')   
        .select('reporting_date'
                , 'reporting_cycle_type'
                , 'reporting_date_curr'
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
        .filter(f.col('_is_late') == 1)
        .groupBy(*ls_param_bill_joining_keys, 'reporting_date_curr') 
        .agg(
              f.countDistinct('bill_no').alias('cnt_late_pay_6bcycle_6mp')
             , f.avg('bill_overdue_days').alias('avg_overdue_days_6bcycle_6mp')
             )
        .withColumnRenamed('reporting_date', 'reporting_date_6mp')
        .withColumnRenamed('reporting_date_curr', 'reporting_date')
        )

# COMMAND ----------

# DBTITLE 1,unit base left join late pay 6mp
df_output_curr = (df_master
        .filter(f.col('reporting_cycle_type') == vt_reporting_cycle_type)
        .join(df_reporting_dates, f.col('reporting_date') == f.col('snapshot_date'), 'inner')
        .drop('snapshot_date')
        #.filter(f.col('reporting_date') == vt_reporting_date)
      # .filter(f.col('reporting_date').)
        .select(*ls_reporting_date_keys, *ls_param_prm_keys)
        .join(df_late_pay_6mp, ls_param_bill_joining_keys, 'left')
        .withColumn("cnt_late_pay_6bcycle_6mp", 
                          f.when(f.col('cnt_late_pay_6bcycle_6mp').isNull()
                                 , 0)
                          .otherwise(f.col('cnt_late_pay_6bcycle_6mp')))
       .withColumn("avg_overdue_days_6bcycle_6mp", 
                          f.when(f.col('avg_overdue_days_6bcycle_6mp').isNull()
                                 , 0)
                          .otherwise(f.col('avg_overdue_days_6bcycle_6mp')))
       .withColumn("reporting_date_6mp", 
                          f.when(f.col('reporting_date_6mp').isNull()
                                 ,f.last_day(f.add_months('reporting_date' ,-6)))
                          .otherwise(f.col('reporting_date_6mp')))
              )        

# COMMAND ----------

display(df_output_curr.limit(10))

# COMMAND ----------

display(df_output_curr)

# COMMAND ----------

display(df_output_curr
        .groupBy('cnt_late_pay_6bcycle_6mp', 'reporting_date')
        .agg(f.count('*')
            , f.countDistinct('fs_acct_id')
            , f.countDistinct('fs_srvc_id')
             )
        )

# COMMAND ----------

# MAGIC %md
# MAGIC #### export data

# COMMAND ----------

dbutils.fs.ls('/mnt/ml-lab/dev_users/dev_sc/99_misc/cohort_seg/fea_late_pay_6mp')

# COMMAND ----------

# display(df_output_curr
#         .agg(f.max('reporting_date'))
#         )

# COMMAND ----------

export_data(
            df = df_output_curr
            , export_path = '/mnt/ml-lab/dev_users/dev_sc/99_misc/cohort_seg/fea_late_pay_6mp'
            , export_format = "delta"
            , export_mode = "append"
            , flag_overwrite_schema = True
            , flag_dynamic_partition = True
            , ls_dynamic_partition = ["reporting_date"]
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Data

# COMMAND ----------

df_test = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/99_misc/cohort_seg/fea_late_pay_6mp')

# COMMAND ----------

display(df_test.limit(10))

# COMMAND ----------

display(df_test
        .groupBy('reporting_date'
                 )
        .agg(f.countDistinct('fs_acct_id')
             , f.count('*').alias('cnt')
             , f.countDistinct('fs_srvc_id')
             )
        )

# COMMAND ----------

display(df_test
        .groupBy('reporting_date', 'cnt_late_pay_6bcycle_6mp'
                 )
        .agg(f.countDistinct('fs_acct_id')
             , f.count('*').alias('cnt')
             )
        .withColumn('sum', f.sum('cnt').over(Window.partitionBy('reporting_date')))
        .withColumn('pct', f.col('cnt')/ f.col('sum'))
        )
