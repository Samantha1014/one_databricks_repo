# Databricks notebook source
import pyspark
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window 

# COMMAND ----------

dir_bill = 'dbfs:/mnt/feature-store-dev/dev_users/dev_el/2024q4_moa_account_risk/d200_staging/d299_src/stg_brm_bill_t'
dir_payment = 'dbfs:/mnt/feature-store-dev/dev_users/dev_el/2024q4_moa_account_risk/d200_staging/d299_src/stg_brm_payment_hist'

# COMMAND ----------

df_bill = spark.read.format('delta').load(dir_bill)
df_payment = spark.read.format('delta').load(dir_payment)

# COMMAND ----------

display(df_bill.count())

# COMMAND ----------

display(df_payment.limit(100))

# COMMAND ----------

display(df_payment
        .withColumn('poid_cnt',
                    f.count('item_poid_id0')
                    .over(Window.partitionBy('item_poid_id0'))
                    )
        .filter(f.col('item_poid_type')!='/item/adjustment')
        .filter(f.col('poid_cnt') >= 2
        
        )

# COMMAND ----------

display(df_bill.limit(10))

# COMMAND ----------

display(df_bill
        .agg(f.count('bill_no')
             , f.countDistinct('bill_no')
             )
        )


# COMMAND ----------

display(df_bill
        .groupBy('')
        .agg(f.count())
        )

# COMMAND ----------

display(df_bill
        .withColumn('bill_no_rnk', f.count('bill_no')
                    .over(Window.partitionBy('bill_no'))
                    )
        .filter(f.col('bill_no_rnk') >1)
        )
