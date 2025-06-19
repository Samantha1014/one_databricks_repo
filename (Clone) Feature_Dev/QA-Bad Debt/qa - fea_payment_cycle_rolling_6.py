# Databricks notebook source
# MAGIC %md
# MAGIC ### summary 
# MAGIC
# MAGIC 1. there is one big adjustment amount - account 652751 in march 2024, amount - 20643236.8900000000 for 6 cycle total 
# MAGIC

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

dbutils.fs.ls("/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/")

# COMMAND ----------

dir_data_fs_support = "/mnt/feature-store-dev/dev_users/dev_el/2024q4_moa_account_risk"
dir_data_fs_support = '/mnt/feature-store-prod-lab'
#df_fs_bill = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_bill_cycle_billing_6"))
#df_fs_aod = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_aod"))
#df_fs_payment_01 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_cycle_rolling_3"))
df_fs_payment_02 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_cycle_rolling_6"))
#df_fs_payment_03 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_method_cycle_rolling_6"))
#df_fs_payment_04 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_latest"))
#df_fs_credit_score = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_credit_score"))
df_fs_oa = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base')

# COMMAND ----------

ls_test_field = ['payment_amt_med_6cycle', ]

# COMMAND ----------

for i in ls_test_field:     
    df_result = (df_fs_payment_02
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

display(df_fs_payment_02.limit(10))

# COMMAND ----------



# COMMAND ----------

ls_test_field = ['payment_amt_tot_6cycle', 'payment_cnt_tot_6cycle', 'payment_fail_cnt_tot_6cycle', 'payment_adj_cnt_tot_6cycle'
                 , 'payment_amt_tot_6cycle',  'payment_adj_amt_tot_6cycle', 
                 ]

# COMMAND ----------

for i in ls_test_field:     
    df_result = (df_fs_payment_02
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

# big adjustment amount 

display(df_fs_payment_02
        .filter(f.col('payment_adj_amt_tot_6cycle') >= 20643230 )
        )
# customer name = New Zealand, Vodafone  in bill image 
# it doesnt pay since 2013, use adjustment as payment 

