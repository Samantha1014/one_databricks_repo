# Databricks notebook source
# MAGIC %md
# MAGIC #### s01 set up 

# COMMAND ----------

# DBTITLE 1,library
import pyspark 
import os
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window
from datetime import datetime, timedelta

# COMMAND ----------

# DBTITLE 1,connector
password = dbutils.secrets.get(scope = "auea-kv-sbx-dxdtlprdct01", key = "sfdbrsdskey")
options = {
  "sfUrl": "vodafonenz_prod.australia-east.azure.snowflakecomputing.com", 
  "sfUser": "SVC_LAB_DS_DATABRICKS",
  "pem_private_key": password.replace('\\n', '\n'),
  "sfDatabase": "LAB_ML_STORE",
  "sfSchema": "sandbox",
  "sfWarehouse": "LAB_DS_WH_SCALE"
}

# COMMAND ----------

# DBTITLE 1,credisene application raw
df_app = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """ 
      select 
      customerref as fs_cust_id
      , a.id
      , a.applicationtype
      , a.plantype
      , case when a.plantype = 1 then 'Consumer plan only'
          when a.plantype = 2 then 'Consumer BB New'
          when a.plantype = 3 then 'Existing Plan Only'
          when a.plantype = 4 then 'Consumer IFP new'
          when a.plantype = 5 then 'Existing BB'
          when a.plantype = 6 then 'Existing IFP'
          when a.plantype = 7 then 'SME New'
          when a.plantype = 8 then 'Enterprise New'
          when a.plantype = 9 then 'Business Existing' end as plantype_desc
      , CONVERT_TIMEZONE('UTC', 'Pacific/Auckland', to_timestamp_ltz(a.createdat)) as application_createdat_nzt
      , CONVERT_TIMEZONE('UTC', 'Pacific/Auckland', to_timestamp_ltz(b.createdat)) as decision_createdat_nzt
      , b.decisionband
      , b.decision
      , b.decisiondescription
      , b.insiebel
      , b.creditlimit
      , b.deposit
      from lab_ml_store.sandbox.TEMP_CREDISENSE_API_APPLICATION a 
      left join lab_ml_store.sandbox.temp_credisense_api_connectionstrings b 
      on a.id = b.relapplicationid
      where b.decisionband in ('105', '206', '207', '208', '209', '210') 
    """
    ).load()
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### s02 parameters

# COMMAND ----------

dir_wo_mvnt  = '/mnt/feature-store-prod-lab/d500_movement/d501_mobile_oa_consumer/mvmt_writeoff'
dir_aod_mvnt = '/mnt/ml-lab/dev_users/dev_sc/aod30_mvnt_acct_agg' # monthly level of aod movement 
dir_wo_score = '/mnt/ml-store-prod-lab/classification/d400_model_score/mobile_oa_consumer_srvc_writeoff_pred365d/model_version=version_1'
dir_fea_unitbase = '/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base'


# COMMAND ----------

# MAGIC %md
# MAGIC #### s03 load data

# COMMAND ----------

df_wo_mvnt = spark.read.format('delta').load(dir_wo_mvnt)
df_aod_mvnt = spark.read.format('delta').load(dir_aod_mvnt)
df_wo_score = spark.read.format('delta').load(dir_wo_score)
df_fea_unitbase = spark.read.format('delta').load(dir_fea_unitbase)

# COMMAND ----------

display(
  df_wo_score
        .filter(f.col('fs_cust_id') == '1-102QKONW')
        )

# COMMAND ----------

df_wo_mvnt_stag = (
       df_wo_mvnt
       # .filter(f.col('reporting_cycle_type') == 'rolling cycle')
       .select('fs_acct_id', 'fs_cust_id', 'movement_date', 'writeoff_amt', 'movement_type' )
       .distinct()
        )

# COMMAND ----------

display(
  df_wo_mvnt_stag
  .filter(f.col('movement_date') > f.lit('2024-05-01')) 
  .groupBy('movement_date')
  .agg(f.countDistinct('fs_acct_id'))
)

# COMMAND ----------

df_aod_mvnt_stag = (
       df_aod_mvnt
       .select('fs_acct_id', 'fs_cust_id', 'movement_date', 'movement_type')
       .distinct()
)

# COMMAND ----------

df_app_agg = (
        df_app
        .withColumn('decision_created_month', f.date_format('DECISION_CREATEDAT_NZT', 'yyyyMM'))
        .withColumn('rnk', f.row_number().over(Window.partitionBy('fs_cust_id').orderBy(f.desc('DECISION_CREATEDAT_NZT'))))
        .filter(f.col('rnk') ==1)
        # .groupBy('decision_created_month')
        # .agg(  f.countDistinct('fs_cust_id')    
        #       , f.count('*') 
        #     )
)

# COMMAND ----------

display(
  df_app_agg
  .agg(f.min('DECISION_CREATEDAT_NZT')
       , f.max('DECISION_CREATEDAT_NZT')
       )
)

# COMMAND ----------

display(df_app_agg
        .groupBy('PLANTYPE_DESC')
        .agg(f.countDistinct('fs_cust_id'))
        )

# COMMAND ----------

# MAGIC %md
# MAGIC #### s04 dev

# COMMAND ----------

display(df_app_agg
        .join(df_wo_mvnt_stag, ['fs_cust_id'], 'left')
        .filter( (f.col('movement_date') > f.col('DECISION_CREATEDAT_NZT'))
                  | (f.col('movement_date').isNull())
        )
        .groupBy(f.date_format('DECISION_CREATEDAT_NZT', 'yyyyMM'))
        .agg(  f.countDistinct('fs_cust_id')
             , f.countDistinct('fs_acct_id')
        )
)

# COMMAND ----------

# DBTITLE 1,grater than $50
display(df_app_agg
        .join(df_wo_mvnt_stag, ['fs_cust_id'], 'left')
        .withColumn('decision_created_month', f.date_format('DECISION_CREATEDAT_NZT', 'yyyyMM'))
        #.filter(f.col('decision_created_month') == '202406')
        .filter( (f.col('movement_date') > f.col('DECISION_CREATEDAT_NZT'))
                  | (f.col('movement_date').isNull())
        )
        .filter(f.col('writeoff_amt') > 50)
        .groupBy('decision_created_month'
                # , 'movement_date'
                 , 'PLANTYPE_DESC')
        .agg(f.countDistinct('fs_acct_id'))
        #.filter(f.col('movement_date') == '2024-07-24')
)

# COMMAND ----------

display(df_app_agg
        .join(df_wo_mvnt_stag, ['fs_cust_id'], 'left')
        .withColumn('decision_created_month', f.date_format('DECISION_CREATEDAT_NZT', 'yyyyMM'))
        #.filter(f.col('decision_created_month') == '202406')
        .withColumn('created_6months_date', f.add_months('DECISION_CREATEDAT_NZT', 6))
        .filter( (f.col('movement_date') > f.col('DECISION_CREATEDAT_NZT'))
                  | (f.col('movement_date').isNull())
        )
        .filter( (f.col('writeoff_amt') > 50) | f.col('writeoff_amt').isNull())
        .groupBy(
                f.date_format('DECISION_CREATEDAT_NZT', 'yyyyMM')
                 , f.date_format('created_6months_date', 'yyyyMM')
                 , 'DECISIONBAND' 
                 , 'DECISIONDESCRIPTION'
                 , 'PLANTYPE_DESC'
                 )
        .agg(
            f.countDistinct('fs_cust_id').alias('cust_cnt')
             , f.countDistinct('fs_acct_id').alias('acct_cnt')
            )
        .withColumn('default_rate', f.col('acct_cnt')/f.col('cust_cnt')*100)
      )

# COMMAND ----------

# 2024-07-07
# 2024-07-14
# 2024-07-21
# 2024-07-28
# 2024-07-31

# COMMAND ----------

# per account level to get highest propentisty 
df_wo_score_acct = (
        df_wo_score
        .filter(f.col('reporting_date') >= '2024-07-01')
        .filter(f.col('reporting_date') <= '2025-01-26')
        #.filter(f.col('reporting_date') == '2024-07-07')
        .select('reporting_date'
                , 'reporting_cycle_type'
                , 'fs_acct_id'
                , 'fs_cust_id'
                , 'propensity_top_ntile'
                )
        .withColumn('rnk', f.row_number().over(Window
                                               .partitionBy('fs_acct_id' , 'fs_cust_id', 'reporting_date')
                                               .orderBy(f.desc('propensity_top_ntile'))
                                               )
        )
        .filter(f.col('rnk') ==1)
        .drop('rnk')
)
        

# COMMAND ----------

display(df_wo_score_acct.limit(10))

# COMMAND ----------

display(df_app_agg
        .withColumn('decision_created_month', f.date_format('DECISION_CREATEDAT_NZT', 'yyyyMM'))
        #.filter(f.col('decision_created_month') == '202406')
        .join(df_wo_score_acct, ['fs_cust_id'], 'left')
        
        .withColumn('in_oa_flag',  f.when(f.col('fs_acct_id').isNotNull(), 1).otherwise(0))
        .filter(f.col('in_oa_flag') == 0)
)

        

# COMMAND ----------

display(df_app_agg
        .join(df_wo_score_acct, ['fs_cust_id'], 'left')
        .join(df_wo_mvnt_stag, ['fs_cust_id'], 'left')
        .withColumn('decision_created_month'
                    , f.date_format('DECISION_CREATEDAT_NZT', 'yyyyMM')
                    )
        .withColumn('in_oa_flag'
                    , f.when(f.col('propensity_top_ntile').isNotNull(), 1).otherwise(0)
                    )
        .filter(f.col('decision_created_month') == '202406')
        .filter( (f.col('movement_date') > f.col('DECISION_CREATEDAT_NZT'))
                  | (f.col('movement_date').isNull())
               )
        .filter( (f.col('writeoff_amt') > 50) 
                | f.col('writeoff_amt').isNull()
                )
        .groupBy(f.date_format('DECISION_CREATEDAT_NZT', 'yyyyMM')
                 , 'DECISIONBAND' 
                 , 'DECISIONDESCRIPTION'
                 , 'PLANTYPE_DESC'
                 , 'propensity_top_ntile'
                 )
        .agg(
               f.countDistinct('fs_cust_id').alias('cust_cnt')
             , f.countDistinct(df_wo_mvnt_stag['fs_acct_id']).alias('acct_cnt')
            )
        .withColumn('default_rate', f.col('acct_cnt') / f.col('cust_cnt') *100)
        )

# COMMAND ----------

display(
  df_app_agg
  .filter(f.col('PLANTYPE_DESC') == 'Existing IFP')
  .count()
)


# COMMAND ----------

display(df_wo_score_acct.limit(10))

# COMMAND ----------

display(
  df_app_agg
  .join(
    df_wo_score_acct
    , ['fs_cust_id']
    , 'left'
  )
  .withColumn('date_diff'
              , f.abs(
                  f.date_diff(
                    f.col('DECISION_CREATEDAT_NZT'), f.col('reporting_date')
                  )
                )
  )
  .withColumn(
    'rank'
    , f.row_number()
        .over(Window
        .partitionBy('fs_cust_id')
        .orderBy(f.asc('date_diff'))
    )
  )
  .filter(
    (f.col('rank')==1) 
    | (f.col('reporting_date').isNull())
    )
  # .agg(f.countDistinct('fs_cust_id')
  #      , f.count('*')
  #      )
  # # .count()
)

# COMMAND ----------

display(
  df_app_agg
  .filter(f.col('PLANTYPE_DESC') == 'Existing IFP')
  .count()
)


# COMMAND ----------

# DBTITLE 1,check example
df_output_curr = (
        df_app_agg
        .filter(f.col('PLANTYPE_DESC') == 'Existing IFP')
        .join(
        df_wo_score_acct
        , ['fs_cust_id']
        , 'left'
        )
        .withColumn('date_diff'
                , f.abs(
                        f.date_diff(
                        f.col('DECISION_CREATEDAT_NZT'), f.col('reporting_date')
                        )
                )
        )
        .withColumn(
        'rank'
        , f.row_number()
                .over(Window
                .partitionBy('fs_cust_id')
                .orderBy(f.asc('date_diff'))
        )
        )
        .filter(
        (f.col('rank')==1) 
        | (f.col('reporting_date').isNull())
        )
        .join(
                df_wo_mvnt_stag
              , ['fs_cust_id']
              , 'left'
        )
        .withColumn('decision_created_month'
                    , f.date_format('DECISION_CREATEDAT_NZT', 'yyyyMM')
                    )
        .withColumn('decision_created_6months', f.add_months('DECISION_CREATEDAT_NZT',6))
        .withColumn('in_oa_flag'
                    , f.when(f.col('propensity_top_ntile').isNotNull(), 1).otherwise(0)
                    )
        #.filter(f.col('decision_created_month') == '202406')
        .filter( (f.col('movement_date') > f.col('DECISION_CREATEDAT_NZT'))
                  | (f.col('movement_date').isNull())
               )
        .filter( (f.col('writeoff_amt') > 50) 
                | f.col('writeoff_amt').isNull()
                )
        #.count()
        #.filter(f.col('PLANTYPE_DESC') == 'Existing IFP')
        #.filter(f.col('propensity_top_ntile') == 100)
        )


# COMMAND ----------

display(df_output_curr.count())
