# Databricks notebook source
# MAGIC %md
# MAGIC #### library

# COMMAND ----------

import os
import pyspark
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window
from datetime import datetime, timedelta

# COMMAND ----------

# MAGIC %run "./utility_functions"

# COMMAND ----------

# MAGIC %md
# MAGIC #### directory

# COMMAND ----------

dbutils.fs.ls('/mnt/feature-store-prod-lab/d500_movement/d501_mobile_oa_consumer/mvmt_writeoff/')

# COMMAND ----------

dir_dev_parent = '/mnt/ml-lab/dev_users/dev_sc/99_misc/'
dir_fs_data_parent = "/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer"
dir_mls_parent = "/mnt/ml-store-prod-lab/classification/d400_model_score/mobile_oa_consumer_srvc_writeoff_pred365d/model_version=version_1"

# unit base
dir_fs_unit_base = os.path.join(dir_fs_data_parent, 'fea_unit_base')

# movement 
dir_aod_mvnt = '/mnt/ml-lab/dev_users/dev_sc/aod30_mvnt_acct_agg'
dir_wo_mvnt = '/mnt/feature-store-prod-lab/d500_movement/d501_mobile_oa_consumer/mvmt_writeoff/'


# COMMAND ----------

# MAGIC %md
# MAGIC #### load data

# COMMAND ----------

df_aod30d_score = spark.read.format('delta').load(os.path.join(dir_dev_parent, 'predict_aod30d_propensity_v3'))
df_wo_score = spark.read.format('delta').load(os.path.join(dir_mls_parent, 'reporting_cycle_type=calendar cycle'))
df_fs_unitbase = spark.read.format('delta').load(dir_fs_unit_base)

### cohort features
df_coll_action = spark.read.format('delta').load(os.path.join(dir_dev_parent, 'cohort_seg/fea_coll_action'))
df_consec_latepay = spark.read.format('delta').load(os.path.join(dir_dev_parent, 'cohort_seg/fea_payment_behavior'))
df_prodcut_acq = spark.read.format('delta').load(os.path.join(dir_dev_parent, 'cohort_seg/fea_product_acq'))

## movement 
df_wo_mvnt = spark.read.format('delta').load(dir_wo_mvnt)
df_aod_mvnt = spark.read.format('delta').load(dir_aod_mvnt)

# COMMAND ----------

# MAGIC %md
# MAGIC #### parameters

# COMMAND ----------

ls_prm_key = ['fs_cust_id', 'fs_srvc_id', 'fs_acct_id']
ls_reporting_date_key = ['reporting_date', 'reporting_cycle_type']
ls_score_fields_select = [
'propensity_score_raw',
'propensity_score_cb',
'propensity_score',
'propensity_top_ntile',
'propensity_segment_qt',
'propensity_segment_pbty'
]

# COMMAND ----------

ls_reporting_date =  ['2024-03-31', '2024-02-29', '2024-01-31']
vt_reporting_cycle_type = 'calendar cycle'

# COMMAND ----------

# MAGIC %md
# MAGIC #### transformation

# COMMAND ----------

df_wo_base_curr_00 = (df_wo_score
        .filter(f.col('reporting_date').isin(*ls_reporting_date))
        .filter(f.col('reporting_cycle_type') == vt_reporting_cycle_type)
        .select(*[f.col(field).alias(f"wo_{field}") for field in ls_score_fields_select]
                +ls_prm_key+ls_reporting_date_key 
                )
        )

# COMMAND ----------

df_aod_base_curr_00 = (df_aod30d_score
        .filter(f.col('reporting_date').isin(*ls_reporting_date))
        .filter(f.col('reporting_cycle_type') == vt_reporting_cycle_type)
        .select(*[f.col(field).alias(f"aod_{field}") for field in ls_score_fields_select]
                + ls_prm_key+ls_reporting_date_key 
        )
        )

# COMMAND ----------

# DBTITLE 1,aod movement for next month
df_aod_mvnt_nxt_mnth = (df_aod_mvnt
        .filter(f.col('reporting_date').isin(['2024-02-29' , '2024-03-31', '2024-04-30']))
        .withColumnRenamed('reporting_date', 'target_reporting_date')
        .withColumn('reporting_date', f.last_day(f.add_months('target_reporting_date',-1)))
        .select('reporting_date'
                , 'fs_acct_id'
                , f.col('movement_date').alias('aod_movement_date')
                ,f.col('target_reporting_date').alias('aod_target_reporting_date')
                )
)


# COMMAND ----------

# DBTITLE 1,wo movement for next month
df_wo_mvnt_nxt_mnth = (df_wo_mvnt
        .withColumn('reporting_date', f.last_day('writeoff_effective_date'))
        .filter(f.col('reporting_date').isin(['2024-02-29' , '2024-03-31', '2024-04-30']))
        .withColumnRenamed('reporting_date', 'target_reporting_date')
        .withColumn('reporting_date', f.last_day(f.add_months('target_reporting_date',-1)))
        .select('reporting_date'
                , 'fs_acct_id'
                ,  f.col('movement_date').alias('wo_movement_date')
                ,  f.col('target_reporting_date').alias('wo_target_reporting_date')
                )
)


# COMMAND ----------

# DBTITLE 1,join with scores and features
df_output=(
  df_fs_unitbase
  .filter(f.col('reporting_date').isin(*ls_reporting_date))
  .filter(f.col('reporting_cycle_type') == vt_reporting_cycle_type)
  .select(*ls_reporting_date_key, *ls_prm_key)
  .join(df_coll_action, ls_reporting_date_key + ls_prm_key, 'inner')
  .join(df_consec_latepay, ls_reporting_date_key + ls_prm_key, 'inner')
  .join(df_prodcut_acq, ls_reporting_date_key + ls_prm_key, 'inner')
  .join(df_wo_base_curr_00, ls_reporting_date_key + ls_prm_key, 'inner')
  .join(df_aod_base_curr_00, ls_reporting_date_key + ls_prm_key, 'left')
  .join(df_aod_mvnt_nxt_mnth, ['fs_acct_id', 'reporting_date'], 'left')
  .join(df_wo_mvnt_nxt_mnth, ['fs_acct_id', 'reporting_date'], 'left')
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Test Threshold for L1 Segmentaion 

# COMMAND ----------

display(df_output
        #.filter(f.col('reporting_date') == '2024-01-31')
        .select(*ls_prm_key, *ls_reporting_date_key
                , 'aod_propensity_top_ntile', 'aod_propensity_segment_qt'
                , 'wo_propensity_top_ntile', 'wo_propensity_segment_qt'
                , 'aod_movement_date', 'wo_movement_date'
        )
        .withColumn('actual_aod', f.when(f.col('aod_movement_date').isNotNull(), 1).otherwise(0))
        .withColumn('actual_wo',  f.when(f.col('wo_movement_date').isNotNull(), 1).otherwise(0))      
        .withColumn('aod_segment',  f.when(f.col('aod_propensity_top_ntile').between(98,100), f.lit('98+'))
                                     .when(f.col('aod_propensity_top_ntile').between(90,92), f.lit('90-92'))
                                     .when(f.col('aod_propensity_top_ntile').between(93,94), f.lit('93-94'))
                                     .when(f.col('aod_propensity_top_ntile').between(95,96), f.lit('95-96'))
                                     .when(f.col('aod_propensity_top_ntile').isin('97'), f.lit('97'))
                                     .when(f.col('aod_propensity_top_ntile').between(80,89), f.lit('80-89'))
                                     .when(f.col('aod_propensity_top_ntile').between(0,79), f.lit('0-79') )
                                     .otherwise('in_aod30')
                     )
        .withColumn('wo_segment',  f.when(f.col('wo_propensity_top_ntile').between(98,100), f.lit('98+'))
                                     .when(f.col('wo_propensity_top_ntile').between(90,92), f.lit('90-92'))
                                     .when(f.col('wo_propensity_top_ntile').between(93,94), f.lit('93-94'))
                                     .when(f.col('wo_propensity_top_ntile').between(95,96), f.lit('95-96'))
                                     .when(f.col('wo_propensity_top_ntile').isin('97'), f.lit('97'))
                                     .when(f.col('wo_propensity_top_ntile').between(80,89), f.lit('80-89'))
                                     .when(f.col('wo_propensity_top_ntile').between(0,79), f.lit('0-79'))
                                     .otherwise('X')
                     )
        .groupBy( 'wo_segment', 'aod_segment', 'reporting_date')
        .agg(f.count('*').alias('count')
             , f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_srvc_id')
             , f.sum('actual_aod').alias('actual_aod')
             , f.sum('actual_wo').alias('actual_wo')
        )
)

# COMMAND ----------

# MAGIC %md
# MAGIC Risk Category Classify -Final

# COMMAND ----------

display(df_output
        # .filter(f.col('reporting_date') == '2024-03-31')
        .select(*ls_prm_key, *ls_reporting_date_key
                , 'aod_propensity_top_ntile', 'aod_propensity_segment_qt'
                , 'wo_propensity_top_ntile', 'wo_propensity_segment_qt'
                , 'aod_movement_date', 'wo_movement_date'
        )
        .withColumn('actual_aod', f.when(f.col('aod_movement_date').isNotNull(), 1).otherwise(0))
        .withColumn('actual_wo',  f.when(f.col('wo_movement_date').isNotNull(), 1).otherwise(0))      
        .withColumn('aod_segment',  f.when(f.col('aod_propensity_top_ntile').between(97,100), f.lit('H'))
                                     .when(f.col('aod_propensity_top_ntile').between(80,96), f.lit('M'))
                                     .when(f.col('aod_propensity_top_ntile').between(0,79), f.lit('L') )
                                     .otherwise('in_aod30')
                     )
        .withColumn('wo_segment',  f.when(f.col('wo_propensity_top_ntile').between(97,100), f.lit('H'))
                                     .when(f.col('wo_propensity_top_ntile').between(80,96), f.lit('M'))
                                     .when(f.col('wo_propensity_top_ntile').between(0,79), f.lit('L') )
                                     .otherwise('X')
                     )
        .groupBy( 'wo_segment', 'aod_segment', 'reporting_date')
        #.groupBy('wo_propensity_top_ntile', 'aod_propensity_top_ntile', 'reporting_date')
        .agg(f.count('*').alias('count')
             , f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_srvc_id')
             , f.sum('actual_aod').alias('actual_aod')
             , f.sum('actual_wo').alias('actual_wo')
        )
)

# COMMAND ----------

# DBTITLE 1,test cohorts
df_l1_classify = (df_output
        # .filter(f.col('reporting_date') == '2024-03-31')
        .select(*ls_prm_key, *ls_reporting_date_key
                , 'aod_propensity_top_ntile', 'aod_propensity_segment_qt'
                , 'wo_propensity_top_ntile', 'wo_propensity_segment_qt'
                , 'aod_movement_date', 'wo_movement_date'
        )
        .withColumn('actual_aod', f.when(f.col('aod_movement_date').isNotNull(), 1).otherwise(0))
        .withColumn('actual_wo',  f.when(f.col('wo_movement_date').isNotNull(), 1).otherwise(0))      
        .withColumn('aod_segment',  f.when(f.col('aod_propensity_top_ntile').between(97,100), f.lit('H'))
                                     .when(f.col('aod_propensity_top_ntile').between(80,96), f.lit('M'))
                                     .when(f.col('aod_propensity_top_ntile').between(0,79), f.lit('L') )
                                     .otherwise('in_aod30')
                     )
        .withColumn('wo_segment',  f.when(f.col('wo_propensity_top_ntile').between(97,100), f.lit('H'))
                                     .when(f.col('wo_propensity_top_ntile').between(80,96), f.lit('M'))
                                     .when(f.col('wo_propensity_top_ntile').between(0,79), f.lit('L') )
                                     .otherwise('X')
                     )
        # .groupBy( 'wo_segment', 'aod_segment', 'reporting_date')
        # #.groupBy('wo_propensity_top_ntile', 'aod_propensity_top_ntile', 'reporting_date')
        # .agg(f.count('*').alias('count')
        #      , f.countDistinct('fs_acct_id')
        #      , f.countDistinct('fs_srvc_id')
        #      , f.sum('actual_aod').alias('actual_aod')
        #      , f.sum('actual_wo').alias('actual_wo')
        # )
)

# COMMAND ----------

display(df_l1_classify
        .filter(f.col('aod_segment') == 'L')
        .filter(f.col('wo_segment') == 'H')
        .filter(f.col('reporting_date') == '2024-01-31')
        )

# COMMAND ----------

display(df_aod_mvnt
        .filter(f.col('fs_acct_id') == '344581363')
        )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Test data

# COMMAND ----------

display(df_output
        .groupBy(#'wo_target_reporting_date'
                 , 'aod_target_reporting_date'
                 )
        .agg(f.countDistinct('fs_acct_id'))
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ######data check 

# COMMAND ----------

display(df_wo_mvnt_nxt_mnth
        .groupBy( 'reporting_date', 'target_reporting_date')
        .agg(f.countDistinct('fs_acct_id'))
        )

# COMMAND ----------

display(df_wo_mvnt
        .withColumn('reporting_date', f.last_day('writeoff_effective_date'))
        .filter(f.col('reporting_date').isin(['2024-02-29' , '2024-03-31', '2024-04-30']))
        .groupBy( 'reporting_date')
        .agg(f.countDistinct('fs_acct_id'))
        )

# COMMAND ----------

display(df_aod_mvnt
        #.withColumn('reporting_date', f.last_day('writeoff_effective_date'))
        .filter(f.col('reporting_date').isin(['2024-02-29' , '2024-03-31', '2024-04-30']))
        .groupBy( 'reporting_date')
        .agg(f.countDistinct('fs_acct_id'))
        )
