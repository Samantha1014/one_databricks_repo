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
#df_fs_bill = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_bill_cycle_billing_6"))
#df_fs_aod = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_aod"))
#df_fs_payment_01 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_cycle_rolling_3"))
#df_fs_payment_02 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_cycle_rolling_6"))
#df_fs_payment_03 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_method_cycle_rolling_6"))
#df_fs_payment_04 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_latest"))
df_fs_credit_score = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_credit_score"))
df_fs_oa = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base')



# COMMAND ----------

display(df_fs_credit_score.limit(19))

# COMMAND ----------



# COMMAND ----------

display(df_fs_credit_score
        .groupBy('cs_source')
        .agg(f.min('cs_submit_date')
             , f.max('cs_submit_date')

        )
)



# COMMAND ----------

ls_test_cat_field = ['cs_flag', 'credit_score_segment', 'cs_source' ]


# COMMAND ----------

for i in ls_test_cat_field:
    df_result = (df_fs_credit_score
        .groupBy('reporting_date')
        .pivot(i)
        .agg(f.countDistinct('fs_acct_id')
             )
        )
    print(i)
    display(df_result)


# COMMAND ----------

ls_test_field = ['credit_score', 'cs_submit_date'
                 ]

# COMMAND ----------

for i in ls_test_field:     
    df_result = (df_fs_credit_score
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
