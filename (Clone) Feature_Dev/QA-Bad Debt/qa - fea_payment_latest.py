# Databricks notebook source
# MAGIC %md
# MAGIC ### Library

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
#df_fs_aod = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_aod"))
#df_fs_payment_01 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_cycle_rolling_3"))
#df_fs_payment_02 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_cycle_rolling_6"))
#df_fs_payment_03 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_method_cycle_rolling_6"))
df_fs_payment_04 = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_payment_latest"))
#df_fs_credit_score = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_credit_score"))
df_fs_oa = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base')


df_fin_march_24 =( spark
                    .read
                    .option('header', 'true')
                    .csv('/FileStore/mnt/ml-lab/dev_users/dev_sc/FinanceReport_31032024222238.csv')
                    .withColumn('account_ref_no',regexp_replace('account_ref_no', '^0+', ''))
                     #.withColumnRenamed('account_ref_no', 'fs_acct_id')
)


# COMMAND ----------

vt_test_date = '2024-03-31'
vt_test_cycle_type = 'calendar cycle'
ls_param_unit_base_field = ['reporting_date', 'reporting_cycle_type' , 'fs_acct_id']


# COMMAND ----------

display(df_fs_payment_04.limit(10))


# COMMAND ----------

# DBTITLE 1,never pay flag
display(df_fs_payment_04
        .groupBy('reporting_date')
        .pivot('never_pay_flag')
        .agg(f.countDistinct('fs_acct_id')
        )
        .withColumn('sum', f.col('Y') + f.col('N'))
        .withColumn('Y_pct', f.col('Y')/ f.col('sum'))
        )

# COMMAND ----------



# COMMAND ----------

df_payment_rpt = (
    df_fs_oa
    .filter(f.col('reporting_date') == f.lit(vt_test_date))
    .filter(f.col('reporting_cycle_type') ==vt_test_cycle_type )
    .select(ls_param_unit_base_field)
    .distinct()
    .join(df_fin_march_24, f.col('fs_acct_id') == f.col('account_ref_no'), 'inner')
    .select(*ls_param_unit_base_field, 'last_payment_date', 'last_payment_amount')
        )

# COMMAND ----------

df_test_01 = (df_fs_payment_04
        .filter(f.col('reporting_date') == f.lit(vt_test_date))
        .filter(f.col('reporting_cycle_type') ==vt_test_cycle_type )
        .select('fs_acct_id', 'reporting_date', 'latest_payment_date', 'latest_payment_amt')
        .distinct()
        .join(df_payment_rpt, ['fs_acct_id'], 'inner')
        .withColumn('amt_diff', f.col('latest_payment_amt') + f.col('last_payment_amount'))
        )

# COMMAND ----------

display(df_test_01)

# COMMAND ----------

display(df_test_01
        .withColumn('fs_payment', f.col('latest_payment_date').isNull())
        .withColumn('rpt_payment', f.col('last_payment_date').isNull())
        .withColumn('check_flag', f.col('fs_payment') == f.col('rpt_payment'))
        .filter(~f.col('check_flag'))
)
