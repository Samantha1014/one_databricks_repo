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

# COMMAND ----------

dir_data_parent = "/mnt/feature-store-prod-lab"
dir_data_parent_shared = os.path.join(dir_data_parent, "")
dir_data_parent_users = os.path.join(dir_data_parent, "")

# COMMAND ----------

dir_data_raw =  os.path.join(dir_data_parent_shared, 'd100_raw')
dir_data_meta = os.path.join(dir_data_parent_users, 'd000_meta')
dir_data_stg = os.path.join(dir_data_parent_users, "d200_staging")
dir_data_int =  os.path.join(dir_data_parent_users, "d200_intermediate")
dir_data_prm =  os.path.join(dir_data_parent_users, "d300_primary/d301_mobile_oa_consumer")
dir_data_fea =  os.path.join(dir_data_parent_users, "d400_feature/d401_mobile_oa_consumer")
dir_data_mvmt = os.path.join(dir_data_parent_users, "d500_movement/d501_mobile_oa_consumer")
dir_data_serv = os.path.join(dir_data_parent_users, "d600_serving")
dir_data_tmp =  os.path.join(dir_data_parent_users, "d999_tmp")

# COMMAND ----------

df_fea_bill = spark.read.format("delta").load(os.path.join(dir_data_fea, "fea_bill_cycle_billing_6"))
df_prm_bill = spark.read.format('delta').load(os.path.join(dir_data_prm, "prm_bill_cycle_billing_6"))

# COMMAND ----------

display(df_prm_bill
        .filter(f.col('total_due') >0 )
        .select('bill_no', 'fs_acct_id', 'bill_due_date', 'bill_close_date')
        .filter(f.col('bill_due_date')>='2023-01-01')
        .filter(f.col('bill_due_date')<='2024-03-31')
        .distinct()
        .withColumn('date_month', f.date_format('bill_due_date', 'yyyy-MM'))
        .filter(f.col('bill_close_date') != '1970-01-01')
        .withColumn('od_days', f.datediff('bill_close_date', 'bill_due_date'))
        .filter(f.col('od_days') >1)
        .groupBy('date_month')
        .agg(f.count('bill_no')
             , f.countDistinct('fs_acct_id')
             )
     )

# COMMAND ----------

display(df_prm_bill
        .filter(f.col('total_due') >0 )
        .select('bill_no', 'fs_acct_id', 'bill_due_date', 'bill_close_date')
        .filter(f.col('bill_due_date')>='2023-01-01')
        .filter(f.col('bill_due_date')<='2024-03-31')
        .distinct()
        .withColumn('date_month', f.date_format('bill_due_date', 'yyyy-MM'))
         #.filter(f.col('bill_close_date') != '1970-01-01')
        .withColumn('od_days', f.datediff('bill_close_date', 'bill_due_date'))
        .withColumn('reporting_od_days', f.when(f.col('bill_close_date') == '1970-01-01'
                                                , f.datediff(f.last_day('bill_due_date') , f.col('bill_due_date'))
                                                )
                    .otherwise(f.col('od_days'))
                    )
        .filter(f.col('reporting_od_days') >1)
        .groupBy('date_month')
        .agg(f.count('bill_no')
             , f.countDistinct('fs_acct_id')
             , f.avg('reporting_od_days')
             , f.median('reporting_od_days')
             )
     )
