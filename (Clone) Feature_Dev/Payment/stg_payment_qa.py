# Databricks notebook source
display(df_payment_base
        .filter(col('poid_type') != '/item/adjustment')
        .groupBy('type', 'subtype', 'pay_type', 'PAYMENT_EVENT_TYPE')
        .agg(f.count('poid_id0'))
        )

# COMMAND ----------

display(df_payment_base
        .filter(col('created_t') >='2021-01-01')
        .count()) # 37M overall 
