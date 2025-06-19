# Databricks notebook source
# MAGIC %md
# MAGIC ### Setup

# COMMAND ----------

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

# MAGIC %md
# MAGIC ### Data load

# COMMAND ----------

dir_data_fs_support = "/mnt/feature-store-dev/dev_users/dev_el/2024q4_moa_account_risk"
#df_fs_bill = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_bill_cycle_billing_6"))
# df_fs_aod = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_aod"))
df_fs_aod = spark.read.format('delta').load('dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d999_tmp/qa_aod_july23')
#df_fs_payment_01 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_cycle_rolling_3"))
#df_fs_payment_02 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_cycle_rolling_6"))
#df_fs_payment_03 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_method_cycle_rolling_6"))
#df_fs_payment_04 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_latest"))
#df_fs_credit_score = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_credit_score"))
df_fs_oa = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base')


df_fin_march_24 =( spark
                    .read
                    .option('header', 'true')
                    .csv('/FileStore/mnt/ml-lab/dev_users/dev_sc/FinanceReport_31032024222238.csv')
                    .withColumn('account_ref_no',regexp_replace('account_ref_no', '^0+', ''))
                     #.withColumnRenamed('account_ref_no', 'fs_acct_id')
)

df_fin_july_23 = (spark
                  .read
                  .option('header', 'true')
                  .csv('/FileStore/mnt/ml-lab/dev_users/dev_sc/FinanceReport_31072023020106.csv')
                  .withColumn('account_ref_no',regexp_replace('account_ref_no', '^0+', ''))
                  )

# COMMAND ----------

df_fs_aod= (
    df_fs_aod
    .groupBy('reporting_date', 'reporting_cycle_type','fs_cust_id', 'fs_acct_id')
    .pivot('aod_ind')
    .agg(f.sum('value')
         )
)

# COMMAND ----------

# DBTITLE 1,check report example
display(df_fin_july_23.limit(10))

# COMMAND ----------

# DBTITLE 1,check fs example
display(df_fs_aod.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check Distribution and Trend

# COMMAND ----------

# DBTITLE 1,numeric features
ls_test_field = ['aod_current', 'aod_01to30', 'aod_31to60', 'aod_61to90', 'aod_91to120', 'aod_121to150', 'aod_151to180', 'aod_181plus'
                 ]

# COMMAND ----------

for i in ls_test_field:     
    df_result = (df_fs_aod
            .filter(f.col('reporting_cycle_type') == 'calendar cycle')
            .groupBy('reporting_date') 
            .agg(
                f.sum(i).alias('sum'),
                f.mean(i).alias('mean'),
                f.median(i).alias('median'),
                f.stddev(i).alias('stddev'),
                f.min(i).alias('min'),
                f.max(i).alias('max'), 
                f.countDistinct('fs_acct_id')
                )
    )
    print(i)
    display(df_result)

# COMMAND ----------

# DBTITLE 1,aod current check
# check aod current for max = 
# reporting_date	sum	mean	median	stddev	min	max	count(fs_acct_id)
#  2024-03-31	6116468 57.9010000	801.74343278900	58.88	11714.355610694842	0E-7	199905.0100000	545214


display( df_fs_aod
    .filter(f.col('reporting_date') =='2023-07-31')
    # .filter(f.col('reporting_cycle_type') == '')
    .filter(f.col('aod_current') >= 199900 )    
        )

# massive aod current amount 

# COMMAND ----------

# DBTITLE 1,categorical features
ls_test_cat_field = ['aod_current_flag', 'aod_01to30_flag', 'aod_31to60_flag', 'aod_61to90_flag', 'aod_91to120_flag', 'aod_121to150_flag', 'aod_151to180_flag', 'aod_181plus_flag'
                 ]

# COMMAND ----------

for i in ls_test_cat_field:
    df_result = (df_fs_aod
        .filter(f.col('reporting_cycle_type') == 'calendar cycle')
        .groupBy('reporting_date')
        .pivot(i)
        .agg(f.countDistinct('fs_acct_id')
             )
        )
    
    df_result_2 = (
        df_result
        .withColumn('y_pct', 
                             f.col('Y') / (f.col('Y') + f.col('N'))
                             )
    )
    print(i)
    display(df_result_2)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Validate with Exsiting Report

# COMMAND ----------

# MAGIC %md
# MAGIC ### Report = 2024-03-31

# COMMAND ----------

print('finance report')

display(df_fin_march_24
        .select('account_ref_no')
        .agg(f.countDistinct('account_ref_no')
             , f.count('*') 
             )
        )       
# duplicate in existing report 
print (' ')
print('fs_aod')

display(df_fs_aod
        .filter(f.col('reporting_date') == '2024-03-31')
        .select('fs_acct_id')
        .distinct()
        .count()
        )



# COMMAND ----------

# DBTITLE 1,duplicate in existing report
display(df_fin_march_24
        .distinct()
        .withColumn('cnt', f.count('*')
                                .over(Window.partitionBy('account_ref_no') )
                    )
        .filter(f.col('cnt') >1)
        )


# COMMAND ----------

vt_test_date = '2024-03-31'
vt_test_cycle_type = 'calendar cycle'
ls_param_unit_base_field = ['reporting_date', 'reporting_cycle_type' , 'fs_acct_id']

# COMMAND ----------

print(ls_test_field)

# COMMAND ----------


rename_dict = {col: f'{col}_ref' for col in ls_test_field}

print(rename_dict)

# COMMAND ----------

# DBTITLE 1,convert to int from string
for c in ls_test_field: 
    df_fin_march_24 = df_fin_march_24.withColumn(c, f.col(c).cast('float'))

# COMMAND ----------

# MAGIC %run "./Function"

# COMMAND ----------

display(df_fs_oa
        .filter(f.col('reporting_date') == f.lit(vt_test_date))
        .filter(f.col('reporting_cycle_type') == vt_test_cycle_type)
        .select(ls_param_unit_base_field)
        .distinct()
        .join(df_aod
              .distinct()
              , ['fs_acct_id'], 'left')
        .fillna(0)
        #.count()
        )

# COMMAND ----------

# DBTITLE 1,create test base
# create test reference table from report 
df_test_ref = (df_fs_oa
        .filter(f.col('reporting_date') == f.lit(vt_test_date))
        .filter(f.col('reporting_cycle_type') == vt_test_cycle_type)
        .select(ls_param_unit_base_field)
        .distinct()
        .join(df_fin_march_24
              .distinct()
              , f.col('fs_acct_id') == f.col('account_ref_no'), 'left')
        .withColumn('match_flag', 
                    f.when(f.col('account_ref_no').isNull(), 'not_in_fs_oa')
                     .otherwise('in_fs_oa')
                     )
        .select('reporting_date', 'fs_acct_id','match_flag',
                 *[f.col(old_name).alias(new_name) for old_name, new_name in rename_dict.items()]
                )
        .fillna(0)
        )

# check count for matching rate 
print('source finance report match to feature store unit base')
print('finance report only have the age debt records, if all of the age bucket = 0')
print( 'it will not show up in report')

display(df_test_ref
        .groupBy('match_flag')
        .agg(f.count('*').alias('row_cnt')
             , f.countDistinct('fs_acct_id').alias('acct_cnt')
            )
        .withColumn('sum_acct_cnt', f.sum('acct_cnt')
                                .over(Window.partitionBy())
                    )
        .withColumn('coverage_pct', f.col('acct_cnt') / f.col('sum_acct_cnt') )
        )

# get feature store aod 

# df_aod = (
#     df_fs_aod
#     .filter(f.col('reporting_date') == f.lit(vt_test_date))
#     .filter(f.col('reporting_cycle_type') == vt_test_cycle_type)
#     .select( 'fs_acct_id',
#                 *ls_test_field
#                 )
#     .distinct()
# )

df_aod = (df_fs_oa
        .filter(f.col('reporting_date') == f.lit(vt_test_date))
        .filter(f.col('reporting_cycle_type') == vt_test_cycle_type)
        .select(ls_param_unit_base_field)
        .distinct()
        .join(df_aod
              .distinct()
              , ['fs_acct_id'], 'left')
        .fillna(0)
)

# joined base to prepare for test 
df_joined= (df_aod
        .join(df_test_ref, ['fs_acct_id'], 'inner')
        )


# get similarity test result 

df_result = check_similarity(dataframe=df_joined, dict_pairs=rename_dict, value_adj=1, threshold=0.95, excl_zero = True)


# result analysis 

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

display(df_result)

# COMMAND ----------

display(df_fin_march_24
        .filter(f.col('account_ref_no') == '483651339')
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Report = 2023-07-31

# COMMAND ----------

# DBTITLE 1,check finance report
print('finance report')

display(df_fin_july_23
        .select('account_ref_no')
        .agg(f.countDistinct('account_ref_no')
             , f.count('*') 
             )
        )       
# duplicate in existing report 
print (' ')
print('fs_aod')

display(df_fs_aod
        .filter(f.col('reporting_date') == '2023-07-31')
        .select('fs_acct_id')
        .distinct()
        .count()
        )



# COMMAND ----------

# DBTITLE 1,parameter
vt_test_date = '2023-07-31'
vt_test_cycle_type = 'calendar cycle'
ls_param_unit_base_field = ['reporting_date', 'reporting_cycle_type' , 'fs_acct_id']

# get dictionary 
rename_dict = {col: f'{col}_ref' for col in ls_test_field}

# COMMAND ----------

# DBTITLE 1,convert to float
for c in ls_test_field: 
    df_fin_july_23 = df_fin_july_23.withColumn(c, f.col(c).cast('float'))

# COMMAND ----------

# MAGIC %run "./Function" 

# COMMAND ----------

# DBTITLE 1,create test base
# create test reference table from report 
df_test_ref = (df_fs_oa
        .filter(f.col('reporting_date') == f.lit(vt_test_date))
        .filter(f.col('reporting_cycle_type') == vt_test_cycle_type)
        .select(ls_param_unit_base_field)
        .distinct()
        .join(df_fin_july_23
              .distinct()
              , f.col('fs_acct_id') == f.col('account_ref_no'), 'left')
        .withColumn('match_flag', 
                    f.when(f.col('account_ref_no').isNull(), 'not_in_fs_oa')
                     .otherwise('in_fs_oa')
                     )
        .select('reporting_date', 'fs_acct_id','match_flag',
                 *[f.col(old_name).alias(new_name) for old_name, new_name in rename_dict.items()]
                )
        .fillna(0)
        )

# check count for matching rate 
print('source finance report match to feature store unit base')
print('finance report only have the age debt records, if all of the age bucket = 0')
print( 'it will not show up in report')

display(df_test_ref
        .groupBy('match_flag')
        .agg(f.count('*').alias('row_cnt')
             , f.countDistinct('fs_acct_id').alias('acct_cnt')
            )
        .withColumn('sum_acct_cnt', f.sum('acct_cnt')
                                .over(Window.partitionBy())
                    )
        .withColumn('coverage_pct', f.col('acct_cnt') / f.col('sum_acct_cnt') )
        )

# get feature store aod 
# df_aod = (
#     df_fs_aod
#     .filter(f.col('reporting_date') == f.lit(vt_test_date))
#     .filter(f.col('reporting_cycle_type') == vt_test_cycle_type)
#     .select( 'fs_acct_id',
#                 *ls_test_field
#                 )
#     .distinct()
# )

df_aod = (df_fs_oa
        .filter(f.col('reporting_date') == f.lit(vt_test_date))
        .filter(f.col('reporting_cycle_type') == vt_test_cycle_type)
        .select(ls_param_unit_base_field)
        .distinct()
        .join(df_fs_aod
              .distinct()
              , ['fs_acct_id'], 'left')
        .fillna(0)
)


# joined base to prepare for test 
df_joined= (df_aod
        .join(df_test_ref, ['fs_acct_id'], 'inner')
        )


# get similarity test result 

df_result = check_similarity(dataframe=df_joined, dict_pairs=rename_dict, value_adj=1, threshold=0.95, excl_zero = True)


# result analysis 

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

display(df_result)
