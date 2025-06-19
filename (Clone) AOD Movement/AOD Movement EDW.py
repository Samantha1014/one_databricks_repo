# Databricks notebook source
# MAGIC %md
# MAGIC ### S01 Setup

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
from pyspark.sql import SparkSession

# COMMAND ----------

df_fs_oa = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base')

# COMMAND ----------

password = dbutils.secrets.get(scope = "auea-kv-sbx-dxdtlprdct01", key = "sfdbrsdskey")
options = {
  "sfUrl": "vodafonenz_prod.australia-east.azure.snowflakecomputing.com", 
  "sfUser": "SVC_LAB_DS_DATABRICKS",
  "pem_private_key": password.replace('\\n', '\n'),
  "sfDatabase": "prod_edw",
  "sfSchema": "raw",
  "sfWarehouse": "LAB_DS_WH_SCALE"
}

# COMMAND ----------

df_aod = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """ 
         with extract_target as(
        select  
           account_ref_no as fs_acct_id 
          , hist_start_dt
          , hist_end_dt
          , aod_current
          , aod_01to30
          , aod_31to60
          , aod_61to90
          , aod_91to120
          , aod_121to150
          , aod_151to180
          , aod_181plus
          , accounts1
          , payment_status
          , dw_source_system_key
          , deceased_flag
          , tm_dispute
          , dw_update_dttm 
        from prod_edw.raw.edw2prd_stageperm_s_inf_aod  
        where 
         _is_latest = 1 and _is_deleted = 0 
        ) 
        select *  from extract_target
         """
    )
    .load()
)

# COMMAND ----------

# DBTITLE 1,check one account
display(df_aod
        .filter(f.col('fs_acct_id') == '460427624')
        )
        # two accountS1 due to bb migration 

# COMMAND ----------

ls_param_fields = ['fs_acct_id', 'fs_cust_id']

# COMMAND ----------

# DBTITLE 1,QA Check count
display(df_fs_oa
        .select(ls_param_fields)
        .filter(f.col('reporting_date') >= '2021-01-01')
        .distinct()
        .join(df_aod
               .filter(f.col('hist_end_dt')>='2021-01-01')
               # .filter(f.col('hist_start_dt')>='2021-01-01')
              , ['fs_acct_id'], 'left')
        .filter(f.col('aod_current').isNull())
        #.count()
)
# 4 records not in fs oa unit base 


# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocessing

# COMMAND ----------

# DBTITLE 1,filter after 2021 and add helper columns
df_aod_f01 = (
        df_fs_oa
        .select(ls_param_fields)
        .filter(f.col('reporting_date')>= '2021-01-01') # after 2021
        .distinct()
        .join(df_aod
              .filter(f.col('hist_end_dt')>='2021-01-01')
              , ['fs_acct_id'], 'left'
              ) # after 2021
        .withColumn('aod_sum',
                     f.col('AOD_CURRENT') + f.col('AOD_01TO30') + f.col('AOD_31TO60')+f.col('AOD_61TO90')
                    + f.col('AOD_91TO120') + f.col('AOD_121TO150') + f.col('AOD_151TO180') + f.col('AOD_181PLUS')
                    )
        .withColumn('aod_period', 
                    f.datediff('hist_end_dt', 'hist_start_dt')
                    )
        .withColumn('payment_status', 
                    f.when( f.col('payment_status') =='X', f.lit('7'))
                    .otherwise(f.col('payment_status'))
                    )
                )

        

# COMMAND ----------

# DBTITLE 1,check distribution and sample
display(df_aod_f01
        .filter(f.col('fs_acct_id') =='431435097')
        )
# this one has a problematic from 0 to 2 movment (status jump)


# distribution of payment status 
display(df_aod_f01
        .groupBy('payment_status')
        .agg(f.count('*'))
        )

# COMMAND ----------

# DBTITLE 1,filter for change records
df_aod_f02 = (df_aod_f01
        .select('fs_acct_id', 'hist_start_dt', 'hist_end_dt', 'aod_period','payment_status')
        .withColumn('previous_payment_status', 
                    f.lag('payment_status', 1)
                    .over(Window.partitionBy('fs_acct_id')
                          .orderBy('hist_start_dt')
                          )
                    )
        .withColumn('next_payment_status', 
                    f.lead('payment_status', 1)
                    .over(Window.partitionBy('fs_acct_id')
                          .orderBy('hist_start_dt')
                          )
                    )
        .withColumn('change_status_flag',  
                    f.when(f.col('previous_payment_status') != f.col('payment_status')
                           , f.lit('Y')
                           )
                    .otherwise(f.lit('N'))
                    )
        .filter(f.col('change_status_flag') == 'Y') # only capture the first occurance of change 
        .withColumn('movement', 
                    f.concat(f.col('previous_payment_status'), 
                             f.lit(' to '), 
                             f.col('payment_status') )
                    )
        .withColumn('direction', f.when(f.col('payment_status') > f.col('previous_payment_status')
                                        , 'in')
                                   .when(f.col('payment_status') < f.col('previous_payment_status')
                                         ,'out'
                                         )
                                   .otherwise('misc')
                    )
        .drop('change_status_flag')
             )
### f.col(431435097)
             

# COMMAND ----------

# DBTITLE 1,check movement distribution
display(df_aod_f02
        .groupBy('payment_status'
                 , 'previous_payment_status')
        .agg(f.count('*'))
        )
        # around 0.2% has bill jump such as 0 to 2 

# COMMAND ----------

# MAGIC %md
# MAGIC ### AOD IN

# COMMAND ----------

# DBTITLE 1,flag problematic recs
df_aod_f03 = (df_aod_f02
        .withColumn('next_action_date', 
                           f.lead('hist_start_dt',1)
                           .over(Window.partitionBy('fs_acct_id').orderBy('hist_start_dt'))
                    )
        .withColumn('next_movement', 
                    f.lead('movement',1)
                    .over(Window.partitionBy('fs_acct_id').orderBy('hist_start_dt'))
                    )
        .withColumn('movement_date_diff', 
                    f.datediff('next_action_date', 'hist_end_dt')
                    )
        .withColumn('problematic_recs',  
                    f.when(( f.col('previous_payment_status') == f.col('next_payment_status'))  
                           # if previous status = next status within 2 days. then probably process payment delay 
                           &(f.col('aod_period')<=2)  # buffer for 2 days entry 
                           &  (f.col('movement_date_diff') ==1) # subsequent change within 1 day 
                    , f.lit('Y'))
                    .otherwise(f.lit('N'))
                    )
)       

# COMMAND ----------

# DBTITLE 1,get aod in
df_aod_in = (
    df_aod_f03
    .filter(f.col('direction') == f.lit('in'))
    .filter(f.col('problematic_recs') == f.lit('N'))
)

# COMMAND ----------

# DBTITLE 1,aod in trend check
display(df_aod_in
        #.filter(f.col('payment_status') == 2)
        .groupBy(f.date_format('hist_start_dt', 'yyyy-MM'))
        .pivot('payment_status')
       .agg(f.countDistinct('fs_acct_id')
             )
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Spot Check AOD in

# COMMAND ----------

display(df_aod_in.filter(f.col('fs_acct_id') ==431435097))
# problematic from 0 to 2 status jump 

# COMMAND ----------

display(df_aod_in.filter(f.col('fs_acct_id') ==438133714))

# COMMAND ----------

display(df_aod_f01.filter(f.col('fs_acct_id') =='438133714'))

# COMMAND ----------

display(df_aod_f02.filter(f.col('fs_acct_id') =='438133714'))

# COMMAND ----------

display(df_aod_f02.filter(f.col('movement') == '5 to 4'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### AOD Out Calculation

# COMMAND ----------

# DBTITLE 1,filter our problematic recs
df_problem_in = (
    df_aod_f03
    .filter(f.col('direction') == 'in')
    .filter(f.col('problematic_recs') == f.lit('Y'))
    .select('fs_acct_id', 'next_action_date')
)


# filter out problematic out records
df_aod_out = (
    df_aod_f03.alias('a')
    .filter(f.col('direction') == 'out')
    .join(df_problem_in.alias('b'), 
          (f.col('a.fs_acct_id') == f.col('b.fs_acct_id')) & 
          (f.col('a.hist_start_dt') == f.col('b.next_action_date'))
           , 'left')
    .filter(f.col('b.fs_acct_id').isNull())
    .select('a.*')   
)

# COMMAND ----------

# DBTITLE 1,check aod  out distribution
display(df_aod_out
        #.filter(f.col('payment_status') == 2)
        .groupBy(f.date_format('hist_start_dt', 'yyyy-MM'))
        .pivot('payment_status')
       .agg(f.countDistinct('fs_acct_id')
             )
        )

# COMMAND ----------

display(df_aod_out.filter(f.col('a.fs_acct_id') == '1000038')) ## dd payment in brm 

# COMMAND ----------

display(df_aod_in.filter(f.col('fs_acct_id') == 1000038)
        ) # payment reversal 

# COMMAND ----------

display(df_problem_in.count())

# COMMAND ----------

# DBTITLE 1,check count
display (
    df_aod_f03.alias('a')
    .filter(f.col('direction') == 'out')
    .join(df_problem_in.alias('b'), 
          (f.col('a.fs_acct_id') == f.col('b.fs_acct_id')) & 
          (f.col('a.hist_start_dt') == f.col('b.next_action_date'))
           , 'inner')
    .count()
         
)

# COMMAND ----------

display(df_problem_in.filter(f.col('fs_acct_id') == 1000038))

# COMMAND ----------

display(df_problem_out.filter(f.col('a.fs_acct_id') == 1000038))
