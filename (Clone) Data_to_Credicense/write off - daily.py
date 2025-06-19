# Databricks notebook source
# MAGIC %md
# MAGIC ### S01 Setup

# COMMAND ----------

import pyspark
import os
from pyspark.sql import functions as f
from pyspark.sql import Window

# COMMAND ----------

dir_prod_brm = 'dbfs:/mnt/prod_brm/raw/cdc'
dir_prod_siebel = 'dbfs:/mnt/prod_siebel/raw/cdc'

# brm 
df_item_t = spark.read.format('delta').load(os.path.join(dir_prod_brm, 'RAW_PINPAP_ITEM_T'))
df_account_t = spark.read.format('delta').load(os.path.join(dir_prod_brm,'RAW_PINPAP_ACCOUNT_T'))


# siebel 
#df_sbl_org_ext = spark.read.format('delta').load(os.path.join(dir_prod_siebel, 'RAW_SBL_ORG_EXT'))
#df_sbl_contact = spark.read.format('delta').load(os.path.join(dir_prod_siebel, 'RAW_SBL_CONTACT'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### S02 Load Data

# COMMAND ----------

df_sbl_output = spark.sql( """ 
   with cust as(
        select 
            a.row_id as customer_id
            , b.birth_dt
            , b.FST_NAME
            , b.last_name
            , b.mid_name
        from DELTA.`/mnt/prod_siebel/raw/cdc/RAW_SBL_ORG_EXT` as a
        left join DELTA.`/mnt/prod_siebel/raw/cdc/RAW_SBL_CONTACT` as b
            on  a.pr_con_id = b.row_id 
        where a.accnt_type_cd = 'Customer'
            and a._is_latest =1 
            --and a._is_deleted = 0 
            and b._is_latest = 1 
            --and b._is_deleted = 0
        ), 
    bill as (
        select 
            ou_num as account_no
            , row_id as billing_acct_id
            , par_ou_id as customer_id
        from DELTA.`/mnt/prod_siebel/raw/cdc/RAW_SBL_ORG_EXT` as b
        where accnt_type_cd in ('Billing')
           and b._is_latest = 1 
           --and b._is_deleted = 0
        ), 
    output as (
        select 
            bill.*
            , cust.birth_dt
            , cust.FST_NAME
            , cust.last_name
            , cust.mid_name 
        from bill 
        left join cust 
            on bill.customer_id = cust.customer_id
        )
        select  *
        from output ; 
         """
    )
   

# COMMAND ----------

# MAGIC %md
# MAGIC ### S03 Development

# COMMAND ----------

vt_daily_look_back_period =  f.date_sub(f.current_date(),5)  # look back 5 days just in case on monday we havent loaded the history 

# COMMAND ----------

# DBTITLE 1,daily version
### wo aggreated into requsted format 
### ADO Link - https://dev.azure.com/vodafonenz/IT/_workitems/edit/318883
### ## 1st Upload will be for 5 years history based on write off Date followed by daily upload as incremental

df_wo_base_daily = (
    df_item_t.alias('b')
    .filter(f.col('_IS_LATEST') ==1)
    .filter(f.col('_IS_DELETED') ==0)
    .filter(f.col('POID_TYPE') == '/item/writeoff')
    .withColumn('rnk', f.row_number()
                .over(Window.partitionBy('ACCOUNT_OBJ_ID0')
                      .orderBy(f.asc('CREATED_T')
                               , f.asc('ITEM_TOTAL') # negative
                               )
                      )
                )
    .filter(f.col('rnk') ==1)
    .withColumn('create_date', 
                 f.date_format(
                       f.from_utc_timestamp(
                        f.from_unixtime(f.col('CREATED_T')), "Pacific/Auckland"
                                        ) , 'yyyy-MM-dd'
                              )
                )
    .filter(f.col('item_total') <= -100)
    .join(df_account_t.alias('a')
          .select('account_no', 'poid_id0')
          .distinct()      
          , f.col('a.poid_id0') == f.col('b.account_obj_id0'), 
          'inner')
    .select(
             'account_no'
            , 'create_date'
            , 'item_total'
           )
    .filter( f.col('create_date') >= vt_daily_look_back_period)
              )


# COMMAND ----------

df_wo_daily = (
              df_wo_base_daily.alias('a')
                .join(df_sbl_output, ['account_no'], 'left')
                .select(
                      # 'a.account_no'
                       'customer_id'
                      , f.date_format('create_date','dd/MM/yyyy').alias('write_off_date')
                       #, f.round('item_total',2).alias('write_off_amount')
                      , f.date_format('birth_dt', 'dd/MM/yyyy').alias('date_of_birth')
                      , f.col('fst_name').alias('first name')
                      , 'last_name'
                      , 'mid_name'
                      , f.date_format(f.add_months('create_date', 60), 'dd/MM/yyyy').alias('expiry_date')
                      , f.lit('WRITEOFF').alias('case_type')
                      )
              .distinct() # avoid same customer but with two different billing written off at same date 
              )

# COMMAND ----------

# MAGIC %md
# MAGIC ### S04 Export Data

# COMMAND ----------

#display(df_wo_daily.count())

# COMMAND ----------

df_output_02 = (df_wo_daily)

display(df_output_02)

# COMMAND ----------

# MAGIC %md
# MAGIC ### S05 QA

# COMMAND ----------

display(df_output_02.limit(100))

# COMMAND ----------

display(
       df_output_02
       .withColumn('cnt', f.count('*').over(Window.partitionBy('customer_id')))
       .filter(f.col('cnt') >1 )
       )
       # pick the latest wo date if multi billing accts happening for one customer for bulk load 
        
