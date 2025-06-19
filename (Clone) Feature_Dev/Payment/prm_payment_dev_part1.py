# Databricks notebook source
# MAGIC %md
# MAGIC ### Payment Part 1
# MAGIC this part contains current cycle and 6 cycles back of 
# MAGIC 1.  payment, adjustment sum total/ avergage/ cnt and
# MAGIC 2.  cnt of fail payment 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### S00001 Library

# COMMAND ----------

import pyspark
from pyspark.sql import functions as f
from pyspark.sql.functions import trunc

# COMMAND ----------

# MAGIC %md
# MAGIC ### S00002 Directory

# COMMAND ----------

dir_payment_int_ssc = 'dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d200_intermediate/payment_int_ssc'
df_payment_int_ssc = spark.read.format('delta').load(dir_payment_int_ssc)

# COMMAND ----------

# MAGIC %run "../Function"

# COMMAND ----------

vt_param_ssc_reporting_date = '2023-07-31'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Latest Payment 

# COMMAND ----------

ls_param_payment_latest_field = ['fs_acct_id','rec_created_dttm', 'item_amount', 'payment_method', 'onenz_auto_pay' ]


# COMMAND ----------

df_payment_prm_latest = (
    df_payment_int_ssc
    .filter(f.col('item_category') == 'payment')
    .withColumn('rnk', f.row_number()
                .over(Window.partitionBy('fs_acct_id')
                      .orderBy(f.desc('rec_created_dttm')))
                )
    .filter(f.col('rnk') ==1)
    .select(ls_param_payment_latest_field) 
    .withColumnRenamed('rec_created_dttm', 'last_payment_date')
    .withColumnRenamed('item_amount', 'last_payment_amount')
    .withColumnRenamed('payment_method', 'last_payment_method') 
    .withColumnRenamed('onenz_auto_pay', 'last_payment_auto')
)  

# COMMAND ----------

# MAGIC %md
# MAGIC ### Current Cycle 

# COMMAND ----------

df_payment_prm_curr = (
    df_payment_int_ssc
    .withColumn('reporting_start_date', 
                f.date_trunc('month', f.lit(vt_param_ssc_reporting_date))
                )
    .filter(f.col('rec_created_dttm') >= f.col('reporting_start_date'))
    .filter(f.col('rec_created_dttm') <= vt_param_ssc_reporting_date)
    .groupBy('fs_acct_id')
    .pivot('item_category')
    .agg(
        f.count('item_id').alias('cnt_curr')
         , f.round(f.sum('item_amount'),2).alias('amount_curr')
         )
    .withColumn('payment_avg_curr', f.col('payment_amount_curr') / f.col('payment_cnt_curr'))
    .drop('fail payment_amount_curr')
    # .filter(f.col('fs_acct_id') == 475907990)
)

# COMMAND ----------

display(df_payment_prm_curr.limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6 Cycle Version 

# COMMAND ----------

vt_param_lookback_cycles = 6

# COMMAND ----------

# 6 cycle sum 
df_payment_prm_cycle = (df_payment_int_ssc
        .groupBy('fs_acct_id')
        .pivot('item_category')
        .agg(
                f.count('item_id').alias('cnt_c')
                ,f.sum('item_amount').alias('amount_c')
        )
       .withColumn('payment_avg_c', f.col('payment_amount_c') / f.col('payment_cnt_c'))
        .drop('fail payment_amount_c')
        # .selectExpr(*[f'`{c}` as `{rename_dict.get(c, c)}`' for c in df_payment_prm_cycle.columns])
)

# COMMAND ----------

rename_dict = {c: c + f'{vt_param_lookback_cycles}' for c in df_payment_prm_cycle.columns if c.endswith ('c') }

# COMMAND ----------

# DBTITLE 1,rename cycle column
df_payment_prm_cycle = (
    df_payment_prm_cycle
     .selectExpr(*[f'`{c}` as `{rename_dict.get(c, c)}`' for c in df_payment_prm_cycle.columns])
)

# COMMAND ----------

display(df_payment_prm_cycle.limit(10))

# COMMAND ----------

# DBTITLE 1,check by joinning
df_test = (df_payment_prm_cycle
        .join(df_payment_prm_curr, ['fs_acct_id'], 'left')
        .join(df_payment_prm_latest, ['fs_acct_id'], 'left')
        )

# COMMAND ----------

display(df_test.limit(10))

# COMMAND ----------

display(df_test
        .agg(f.count('fs_acct_id')
             , f.countDistinct('fs_acct_id')
             )
        )
