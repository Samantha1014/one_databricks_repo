# Databricks notebook source
# MAGIC %md
# MAGIC ### S00001 Library

# COMMAND ----------

import pyspark
import os 
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql.functions import from_unixtime,from_utc_timestamp
from pyspark.sql.functions import date_format, last_day, datediff
from pyspark.sql.functions import regexp_replace, last
from pyspark.sql.functions import lag, lead
from pyspark.sql.functions import col, month, dayofmonth, when, add_months
from pyspark.sql import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ### S00002 Directory

# COMMAND ----------

dir_global_calendar_meta = 'dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d000_meta/d001_global_cycle_calendar'
df_global_calendar_meta = spark.read.format('delta').load(dir_global_calendar_meta)

# COMMAND ----------

dir_oa_int = '/mnt/feature-store-prod-lab/d200_intermediate/d201_mobile_oa_consumer/int_ssc_billing_account/reporting_cycle_type=calendar cycle'
df_oa_int = spark.read.format('delta').load(dir_oa_int)

# COMMAND ----------

dir_bill_base = '/mnt/feature-store-dev/dev_users/dev_sc/d100_raw/BILL_BASE'
df_bill_base = spark.read.format('delta').load(dir_bill_base)

# COMMAND ----------

# MAGIC %md
# MAGIC ### S00003 Development

# COMMAND ----------

# DBTITLE 1,Parameter
vt_param_ssc_reporting_date = '2023-07-31'
vt_param_cycle = 6

# COMMAND ----------

# DBTITLE 1,OA Unit Base
df_oa_int_curr = (
    df_oa_int
    .filter(
        f.col('reporting_date') == vt_param_ssc_reporting_date
            )
    .select(
        'reporting_date'
        , 'fs_acct_id'
        , 'fs_cust_id')
    .distinct()
)

display(df_oa_int_curr.count()) # 539771

# COMMAND ----------

# DBTITLE 1,Inner Join to Narrow Down
# inner join to narrow down to oa consumer 
df_bill_int_01 = (
    df_bill_base
    .join(df_oa_int_curr, ['fs_acct_id'], 'inner')
)

# COMMAND ----------

# DBTITLE 1,Reorder Fields
ls_param_fields = [
'fs_acct_id',
'fs_cust_id',
'reporting_date',
'bill_no',
'rec_created_dttm',
'rec_mod_dttm',
'bill_due_dttm',
'bill_start_dttm',
'bill_end_dttm',
'bill_closed_dttm',
'bill_previous_total',
'bill_total_due',
'bill_adjusted',
'bill_due_amt',
'bill_recvd_amt',
'bill_transferred_amt',
'bill_subords_total',
'bill_current_total',
'bill_writeoff_amt',
'bill_end_month',
'last_bill_cycle_rnk'
]

# COMMAND ----------

df_output_curr = (
    df_bill_int_01
    .filter(col('bill_due_dttm')<= vt_param_ssc_reporting_date)
    .withColumn('last_bill_cycle_rnk', 
                f.row_number()
                .over(Window.partitionBy('fs_acct_id')
                      .orderBy(f.desc('bill_due_dttm'))
                      )
                )
    .filter(col('last_bill_cycle_rnk')<= vt_param_cycle)
    .select(ls_param_fields)
)

# 3,051,818 rows 

# COMMAND ----------

display(df_output_curr.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### S00004 Export to Output

# COMMAND ----------

 # dbutils.fs.rm('dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d200_intermediate/bill_int_scc',True)

# COMMAND ----------

(
    df_output_curr
    .write
    .format("delta")
    .mode("overwrite")
    .partitionBy("bill_end_month")
    .option("overwriteSchema", "true")
    .option("partitionOverwriteMode", "dynamic")
    .save("dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d200_intermediate/bill_int_scc")
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### S00005 QA

# COMMAND ----------

dir_bill_int = 'dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d200_intermediate/bill_int_scc'
df_bill_int = spark.read.format('delta').load(dir_bill_int)

# COMMAND ----------

display(df_bill_int.count()) # 3051852
display(df_bill_int.limit(100))


# COMMAND ----------

# DBTITLE 1,Check Output
display(df_output_curr
        .groupby('fs_acct_id')
        .agg(f.count('*').alias('cnt'))
        .groupBy('cnt')
        .agg(f.countDistinct('fs_acct_id').alias('count'))   
        .withColumn('sum',f.sum('count').over(Window.partitionBy()))
        .withColumn('pct', f.col('count')/f.col('sum'))
)
# ~ around 91% has 6 bills 

# COMMAND ----------

display(df_output_curr
        .groupby('fs_acct_id')
        .agg(f.count('*').alias('cnt'))
        .filter(f.col('cnt')==1)
        .limit(10)
)
