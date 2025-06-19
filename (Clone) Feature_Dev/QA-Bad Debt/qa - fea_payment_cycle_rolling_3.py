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

dir_data_fs_support = "/mnt/feature-store-dev/dev_users/dev_el/2024q4_moa_account_risk"
df_fs_bill = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_bill_cycle_billing_6"))
df_fs_aod = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_aod"))
df_fs_payment_01 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_cycle_rolling_3"))
df_fs_payment_02 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_cycle_rolling_6"))
df_fs_payment_03 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_method_cycle_rolling_6"))
df_fs_payment_04 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_latest"))
df_fs_credit_score = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_credit_score"))
df_fs_oa = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base')

# COMMAND ----------

ls_groupby_field = ['reporting_date', 'reporting_cycle_type']

# COMMAND ----------

display(df_fs_payment_01.limit(10))

# COMMAND ----------

# DBTITLE 1,check payment cnt

df_result = (df_fs_payment_01
        .groupBy(*ls_groupby_field)
        .pivot('payment_cnt_tot_1cycle', values = ['1','2','3','4', '5'])
        .agg(f.countDistinct('fs_acct_id')
             )
        .withColumn('sum', f.col('1') + f.col('2') + f.col('3') + f.col('4') + f.col('5'))
        )

for i in ['1','2','3','4', '5']:
    df_result = df_result.withColumn(f'pct_{i}', f.col(i)/ f.col('sum'))

display(df_result)  

# COMMAND ----------

# DBTITLE 1,check payment cnt 3 cycle
df_result = (df_fs_payment_01
        .groupBy(*ls_groupby_field)
        .pivot('payment_cnt_tot_3cycle')
        .agg(f.countDistinct('fs_acct_id')
             )
       #  .withColumn('sum', f.col('1') + f.col('2') + f.col('3') + f.col('4') + f.col('5'))
        )

# for i in ['1','2','3','4', '5']:
#     df_result = df_result.withColumn(f'pct_{i}', f.col(i)/ f.col('sum'))

display(df_result)  

# COMMAND ----------

# DBTITLE 1,in cycle payment flag
display(df_fs_payment_01
        .groupBy(*ls_groupby_field)
        .pivot('payment_cycle_rolling_flag')
        .agg(f.countDistinct('fs_acct_id'))
        .withColumn('sum', f.col('N') + f.col('Y'))
        .withColumn('Y_pct', f.col('Y')/ f.col('sum'))
        )
# ~ 80% pay within current cycle 

# COMMAND ----------

# DBTITLE 1,fail payment check
display(df_fs_payment_01
        .groupBy(*ls_groupby_field)
        .pivot('payment_fail_cnt_tot_1cycle')
        .agg(f.countDistinct('fs_acct_id')
            )
             
        )

        ## why there is a sudden increase since may 2023 ?! 

# COMMAND ----------

# DBTITLE 1,check example
display(df_fs_payment_01
        .filter(f.col('reporting_date') == '2023-05-31')
        .filter(f.col('payment_fail_cnt_tot_1cycle') == f.lit('1'))
        .limit(100)
        )

# COMMAND ----------

# DBTITLE 1,load payment base
df_payment_base = spark.read.format('delta').load('/mnt/feature-store-dev/dev_users/dev_sc/d100_raw/PAYMENT_BASE')

# COMMAND ----------

# DBTITLE 1,check raw data
display(df_payment_base.filter(f.col('fs_acct_id')
                               =='393128424'
                               )
        )

# COMMAND ----------

# DBTITLE 1,adjustment cnt check
display(df_fs_payment_01
        .groupBy(*ls_groupby_field)
        .pivot('payment_adj_cnt_tot_3cycle')
        .agg(f.countDistinct('fs_acct_id')
             )
        )


# payment_adj_cnt_tot_3cycle
# 1 adjustment happeed to peak at end 2021 early 2022 
# and 2023 may  - june  
