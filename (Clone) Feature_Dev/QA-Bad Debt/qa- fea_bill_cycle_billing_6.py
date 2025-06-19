# Databricks notebook source
import pyspark
import os
import re
import numpy as np
from pyspark import sql 
from pyspark.sql import Window
from pyspark.sql import functions as f
from pyspark.sql.types import IntegerType, StructType
from pyspark.sql.functions import monotonically_increasing_id, row_number
from itertools import islice, cycle
from pyspark.sql.functions import regexp_replace 
from pyspark.sql.functions import percentile_approx

# COMMAND ----------

dir_data_fs_support = "/mnt/feature-store-dev/dev_users/dev_el/2024q4_moa_account_risk"
df_fs_bill = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_bill_cycle_billing_6"))
#df_fs_aod = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_aod"))
#df_fs_payment_01 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_cycle_rolling_3"))
#df_fs_payment_02 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_cycle_rolling_6"))
#df_fs_payment_03 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_method_cycle_rolling_6"))
#df_fs_payment_04 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_latest"))
#df_fs_credit_score = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_credit_score"))
df_fs_oa = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base')

#df_payment_base = spark.read.format('delta').load('/mnt/feature-store-dev/dev_users/dev_sc/d100_raw/PAYMENT_BASE')

# COMMAND ----------

display(df_fs_bill.limit(100))

# COMMAND ----------

# DBTITLE 1,check bill payment timeless status flag
df_bill_result = (df_fs_bill
        .filter(f.col('reporting_cycle_type') == 'calendar cycle')
        .groupBy('reporting_date')
        .pivot('bill_payment_timeliness_status')
        .agg(f.countDistinct('fs_acct_id'))
        )


# display(df_bill_result) 

# make a list of columns to sum 
columns_to_sum = [col for col in df_bill_result.columns if col != 'reporting_date']

print(columns_to_sum)

# check columns 
df_bill_result = (df_bill_result
         .withColumn('total', sum(f.col(col) for col in columns_to_sum))
)


for column in columns_to_sum: 
    df_bill_result = df_bill_result.withColumn(column+'pct', f.col(column) / f.col('total') )


# COMMAND ----------

display(df_bill_result)

# COMMAND ----------

display(df_fs_bill
        .filter(f.col('reporting_cycle_type') == 'calendar cycle')
        .groupBy('reporting_date')
        .pivot('bill_payment_timeliness_status')
        .agg(f.countDistinct('fs_acct_id')
             
             )

        )

# COMMAND ----------

# DBTITLE 1,check credit bill
display(df_fs_bill
        .filter(f.col('reporting_date') == '2024-03-31')
        .filter(f.col('bill_payment_timeliness_status') == 'credit_bill' )
        .filter(f.col('bill_charge_amt') >0 )
        .filter(f.col('bill_overdue_days') >0)
        # .agg(f.avg('bill_overdue_days'))
        )

# COMMAND ----------

display(df_fs_bill
        .filter(f.col('reporting_date') == '2024-03-31')
        .filter(f.col('bill_payment_timeliness_status').isNull())
        )
        # case when there is only bills in certain period 

# COMMAND ----------

# DBTITLE 1,cat varibale
ls_test_field_cat = ['bill_close_flag']

# COMMAND ----------

display(df_fs_bill
        .groupBy('reporting_date')
        .pivot(ls_test_field_cat)
        .agg(f.countDistinct('fs_acct_id')
             , f.countDistinct('bill_no')
             )
        )

# COMMAND ----------

ls_test_field_numeric = ['bill_cnt_tot_6bmnth', 'bill_due_amt_tot_6bmnth'
                         , 'bill_due_amt_avg_6bmnth'
                         , 'bill_charge_amt_tot_6bmnth', 'bill_charge_amt_avg_6bmnth'
                         , 'bill_carryover_bal_avg_6bmnth' ]

# COMMAND ----------

ls_test = ['bill_due_amt_tot_6bmnth']

# COMMAND ----------

for i in ls_test_field_numeric:     
    df_result = (df_fs_bill
            .groupBy('reporting_date')
            .agg(
                f.sum(i).alias('sum'),
                f.mean(i).alias('mean'),
                f.percentile_approx(i, 0.25,100).alias('25pct'),
                f.percentile_approx(i, 0.79,100).alias('75pct'),
                f.percentile_approx(i, 0.95,100).alias('95pct'),
                f.percentile_approx(i, 0.99,100).alias('99pct'),
                f.median(i).alias('median'),
                f.stddev(i).alias('stddev'),
                f.min(i).alias('min'),
                f.max(i).alias('max'), 
                f.countDistinct('fs_acct_id'), 
                f.countDistinct('bill_no')
                )
    )
    print(i)
    display(df_result)

# COMMAND ----------

ls_test_field_numeric_time = ['bill_cnt_early_tot_6bmnth', 'bill_charge_amt_early_tot_6bmnth'
                              , 'bill_overdue_days_early_avg_6bmnth', 'bill_cnt_late_tot_6bmnth'
                             , 'bill_charge_amt_late_tot_6bmnth', 'bill_overdue_days_late_avg_6bmnth'
                             , 'bill_charge_amt_ontime_tot_6bmnth' , 'bill_overdue_days_ontime_avg_6bmnth', 
                             'bill_cnt_full_tot_6bmnth', 'bill_cnt_over_tot_6bmnth', 'bill_cnt_partial_tot_6bmnth'
                             , 'bill_cnt_unpaid_tot_6bmnth'

                              ]

# COMMAND ----------

for i in ls_test_field_numeric_time:     
    df_result = (df_fs_bill
            .groupBy('reporting_date')
            .agg(
                f.sum(i).alias('sum'),
                f.mean(i).alias('mean'),
                f.percentile_approx(i, 0.25,100).alias('25pct'),
                f.percentile_approx(i, 0.79,100).alias('75pct'),
                f.percentile_approx(i, 0.95,100).alias('95pct'),
                f.percentile_approx(i, 0.99,100).alias('99pct'),
                f.median(i).alias('median'),
                f.stddev(i).alias('stddev'),
                f.min(i).alias('min'),
                f.max(i).alias('max'), 
                f.countDistinct('fs_acct_id'), 
                f.countDistinct('bill_no')
                )
    )
    print(i)
    display(df_result)

# COMMAND ----------

# 21001318.9100000000
display(df_fs_bill
        .filter(f.col('reporting_date') == '2024-03-31')
        .select('fs_acct_id', 'bill_due_amt_tot_6bmnth')
        .distinct()
        .orderBy(f.desc('bill_due_amt_tot_6bmnth'))
        .limit(20)
        # .filter(f.col('bill_due_amt_tot_6bmnth') >=21001310 )
        )
