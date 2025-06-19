# Databricks notebook source
# MAGIC %md
# MAGIC ### Payment Part 2
# MAGIC this part contains 6 cycles back of 
# MAGIC 1. payment method (most frequent)
# MAGIC 2. auto pay pct
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

display(df_payment_int_ssc.count())

# COMMAND ----------

display(df_payment_int_ssc.limit(10))

# COMMAND ----------

# MAGIC %run "../Function"

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6 cycle version 

# COMMAND ----------

vt_param_lookback_cycles = 6

# COMMAND ----------

# 6 cycle most frequent payment method 
df_prm_pay_method_cycle =  (df_payment_int_ssc
        .filter(f.col('item_category') == 'payment')
        .groupBy('fs_acct_id', 'payment_method')
        .agg(f.count('*').alias('payment_cnt'))
        .withColumn('rnk', 
                    f.row_number().over(
                            Window.partitionBy('fs_acct_id').orderBy(f.desc('payment_cnt'))))
        .filter(f.col('rnk')==1)
        .drop('rnk', 'payment_cnt')
        .withColumnRenamed('payment_method', 'frequent_payment_mehtod_c' + str(vt_param_lookback_cycles))
)

# COMMAND ----------

# 6 cycle one nz auto pay 
df_prm_autopay_cycle = (df_payment_int_ssc
        .filter(f.col('item_category') == 'payment')
        .withColumn('onenz_auto_pay_int', f.when(
            f.col('onenz_auto_pay') =='Y',1)
                    .otherwise(0)
        )
        .groupBy('fs_acct_id')
        .agg(f.sum('onenz_auto_pay_int').alias('total_autopay_cnt')
             , f.count('*').alias('total_cnt')
             )
        .withColumn('auto_pay_pct', 
                    f.round(f.col('total_autopay_cnt')/ f.col('total_cnt'),2)
                     )
        .select('fs_acct_id', 'auto_pay_pct')
        .withColumnRenamed('auto_pay_pct', 'auto_pay_pct_c'+ str(vt_param_lookback_cycles))
)


# COMMAND ----------

# DBTITLE 1,check by joinning
df_test = (df_prm_autopay_cycle
        .join(df_prm_pay_method_cycle, ['fs_acct_id'], 'inner')
        )

# COMMAND ----------

display(df_test)
