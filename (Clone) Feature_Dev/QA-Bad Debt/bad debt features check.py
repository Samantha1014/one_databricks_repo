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

# COMMAND ----------

# MAGIC %run "./Function"

# COMMAND ----------

dir_data_fs_support = "/mnt/feature-store-dev/dev_users/dev_el/2024q4_moa_account_risk"
 
df_fs_bill = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_bill_cycle_billing_6"))
df_fs_aod = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_aod"))
df_fs_payment_01 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_cycle_rolling_3"))
df_fs_payment_02 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_cycle_rolling_6"))
df_fs_payment_03 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_method_cycle_rolling_6"))
df_fs_payment_04 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_latest"))
df_fs_credit_score = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_credit_score"))
df_fs_oa = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base')


## finance report 

df_finance_report =( spark
                    .read
                    .option('header', 'true')
                    .csv('/FileStore/mnt/ml-lab/dev_users/dev_sc/FinanceReport_31032024222238.csv')
                    .withColumn('account_ref_no',regexp_replace('account_ref_no', '^0+', ''))
                     #.withColumnRenamed('account_ref_no', 'fs_acct_id')
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### parameter

# COMMAND ----------

vt_param_test_date = '2024-03-31'
vt_param_test_reporting_cycle = 'calendar cycle'

# columns to test 
original_columns = [
    "AOD_Current",
    "AOD_01TO30", "AOD_31TO60", "AOD_61TO90",
    "AOD_91TO120", "AOD_121TO150", "AOD_151TO180", "AOD_181PLUS"
]

rename_dict = {col: f'{col}_ref' for col in original_columns}

print(rename_dict)

# COMMAND ----------

# DBTITLE 1,combine with oa
df_oa_fin = (
    df_fs_oa.select('reporting_date', 'fs_acct_id')
    .filter(f.col('reporting_date') == vt_param_test_date)
    .distinct()
    .join(df_finance_report, f.col('fs_acct_id') == f.col('account_ref_no')
          , 'inner')
    .select('reporting_date', 'fs_acct_id', 'account_ref_no',
            *[f.col(old_name).alias(new_name) for old_name, new_name in rename_dict.items()]
             )
            )     

display(df_oa_fin.count())

# COMMAND ----------

display(df_fs_aod
        .filter(f.col('reporting_date') == vt_param_test_date)
        .filter(f.col('reporting_cycle_type') == f.lit(vt_param_test_reporting_cycle))
        .limit(10)
                    )
        

# COMMAND ----------

display(df_oa_fin.limit(100))

# COMMAND ----------

display(df_fs_aod.limit(10))
display(df_oa_fin)

# COMMAND ----------

display(df_oa_fin
        .filter(f.col('account_ref_no').isNull())
        .count()
        )

# COMMAND ----------

df_aod  = (df_fs_aod
       #  .filter(f.col('aod_01to30_flag') == f.lit('Y'))
        .filter(f.col('reporting_date') == '2024-03-31')
        .filter(f.col('reporting_cycle_type') == 'calendar cycle')
        .select('fs_acct_id', *original_columns )
        .distinct()
        )

# COMMAND ----------

df_joined = (df_aod
        .join(
            df_oa_fin
        # .filter(f.col('AOD_01TO30') >0)
        .distinct(),  ['fs_acct_id'],  'full'
        )
        .withColumn('match_flag', f.when(
            f.col('aod_01to30').isNull(), f.lit('in_report_not_in_fs')
            )
                                   .when(f.col('AOD_01TO30_ref').isNull(), f.lit('in_fs_not_in_report')
                                         )
                                   .when((f.col('aod_01to30').isNotNull() ) & 
                                        (f.col('aod_01to30_ref').isNotNull()), f.lit('in_both') )
                                   .otherwise(f.lit('misc'))
                    )

        )
        

# COMMAND ----------

display(df_joined_30
        .groupBy('match_flag')
        .agg(f.count('*'))
        )

# COMMAND ----------

rename_dict

# COMMAND ----------

df_result = check_similarity(dataframe=df_joined, dict_pairs=rename_dict, value_adj=1, threshold=0.95, excl_zero = True)

# COMMAND ----------


display(
    df_result
    .groupBy('fs_col','benchmark_col')
    .agg(
        f.count('*').alias('count')
        , f.sum('similar_flag').alias('align')
    )
    .withColumn(
        'rate'
        , f.col('align')/f.col('count')
    )
    .withColumn('not_align', f.col('count') - f.col('align'))
)


# COMMAND ----------

display(df_fs_aod.filter(f.col('fs_acct_id')
                         ==1088828
                         )
        .filter(f.col('reporting_date') =='2024-03-31')

)

# COMMAND ----------

display(df_fs_aod.limit(100))

# COMMAND ----------

ls_param_fields = ['aod_current_flag', 
                   'aod_31to60_flag', 
                   'aod_61to90_flag', 
                   'aod_91to120_flag', 
                   'aod_121to150_flag',
                   'aod_151to180_flag'
                   ]

# COMMAND ----------

for i in ls_param_fields: 

    display(df_fs_aod
        .filter(f.col('reporting_cycle_type') =='calendar cycle') 
        .groupBy('reporting_date', 'reporting_cycle_type')
        .pivot(i)
        .agg(f.countDistinct('fs_acct_id'))
        .withColumn('sum', f.col('N') + f.col('Y'))
        .withColumn('pct', f.col('N')/ f.col('sum'))
        .orderBy('reporting_date')
        )
    print(i)
    

# COMMAND ----------

display(df_fs_bill
        .groupBy('reporting_date')
        .pivot( 'bill_payment_timeliness_status')
        .agg(f.countDistinct('fs_acct_id')
             , f.avg('bill_overdue_days')
        )
)

# COMMAND ----------

display(
    df_fs_bill
    .filter(f.col('reporting_date') =='2024-01-31')
    .filter(f.col('reporting_cycle_type') == 'calendar cycle')
    .select('reporting_date', 'fs_acct_id'
            , 'bill_overdue_days_late_avg_6bmnth'
            , 'bill_cnt_late_tot_6bmnth'
            )
    .distinct()
)

# COMMAND ----------

display(
    df_fs_bill
    .filter(f.col('reporting_date') =='2023-09-30')
    .filter(f.col('reporting_cycle_type') == 'calendar cycle')
    
    .groupBy('bill_payment_timeliness_status')
    .agg(
        f.avg('bill_overdue_days')
        , f.countDistinct('fs_acct_id')
    )
)

# COMMAND ----------

display(df_fs_aod
        .filter(f.col('fs_acct_id')
                ==357063499
                 )
        )

# COMMAND ----------

display(df_fs_payment_01.limit(10))

# COMMAND ----------

display(df_fs_payment_02.limit(10))

# COMMAND ----------

display(df_fs_payment_03
         .filter(f.col('reporting_date') =='2023-09-30')
         .filter(f.col('reporting_cycle_type') == 'calendar cycle')
        .select('reporting_date', 'reporting_cycle_type', 
                'fs_acct_id'
                , 'payment_method_main_type_6cycle'
                , 'payment_method_main_cnt_6cycle'
                , 'payment_auto_flag_6cycle'
                , 'payment_auto_pct_6cycle'
                )
        .distinct()
        .groupBy('reporting_date','reporting_cycle_type','payment_auto_flag_6cycle')
        .agg(f.countDistinct('fs_acct_id').alias('count_account'))
        .withColumn('total_acct_sum', f.sum('count_account').over(Window.partitionBy()))
        .withColumn('pct', f.round(f.col('count_account')/f.col('total_acct_sum'),2))
        )

# COMMAND ----------

display(df_fs_payment_04
        .filter(f.col('reporting_date') >= '2024-01-01')
        .limit(100))

# COMMAND ----------

# DBTITLE 1,never payer analysis
display(df_fs_payment_04
        .select('reporting_date', 'fs_acct_id', 'never_pay_flag', 
                'latest_payment_date'
                )
        .distinct()
        .filter(f.col('reporting_date') >= '2023-01-31' )
        .groupBy('reporting_date')
        .pivot('never_pay_flag')
        .agg(f.countDistinct('fs_acct_id'))
        .withColumn('sum of account', f.col('N') + f.col('Y'))
        .withColumn('neverpay_pct', f.round(f.col('Y')/ f.col('sum of account'),2) )
        )

# COMMAND ----------

display(df_fs_credit_score.limit(1000))

# COMMAND ----------

display()
