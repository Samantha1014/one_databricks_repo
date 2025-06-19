# Databricks notebook source
# MAGIC %md
# MAGIC #### libraries

# COMMAND ----------

import os
import pyspark
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window
from datetime import datetime, timedelta

# COMMAND ----------

# MAGIC %md
# MAGIC #### directory

# COMMAND ----------

# MAGIC %run "./utility_functions"

# COMMAND ----------

dir_dev_parent = '/mnt/ml-lab/dev_users/dev_sc/99_misc/cohort_seg'
dir_dev_parent_master = '/mnt/ml-lab/dev_users/dev_sc/99_misc'
dir_fs_data_parent = "/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer"
dir_mls_parent = "/mnt/ml-store-prod-lab/classification/d400_model_score/mobile_oa_consumer_srvc_writeoff_pred365d/model_version=version_1"


# unit base
dir_fs_unit_base = os.path.join(dir_fs_data_parent, 'fea_unit_base')

# payment and billing feature 




# movement 

dir_aod30_mvnt = '/mnt/ml-lab/dev_users/dev_sc/aod30_mvnt_acct_agg'
dir_wo_mvnt = '/mnt/feature-store-prod-lab/d500_movement/d501_mobile_oa_consumer/mvmt_writeoff/'
dir_aod05_mvnt = '/mnt/ml-lab/dev_users/dev_sc/aod_mvnt_agg/aod05to15_mvnt_acct_agg'
dir_aod15_mnvt = '/mnt/ml-lab/dev_users/dev_sc/aod_mvnt_agg/aod15to23_mvnt_acct_agg'
dir_aod_23_mvnt = '/mnt/ml-lab/dev_users/dev_sc/aod_mvnt_agg/aod23to30_mvnt_acct_agg'



# COMMAND ----------

# MAGIC %md
# MAGIC #### load data

# COMMAND ----------

# DBTITLE 1,load database
## features 
df_coll_action = spark.read.format('delta').load(os.path.join(dir_dev_parent, 'fea_coll_action'))
df_consec_latepay = spark.read.format('delta').load(os.path.join(dir_dev_parent, 'fea_payment_behavior_v2'))
df_prodcut_acq_12m = spark.read.format('delta').load(os.path.join(dir_dev_parent, 'fea_product_acq_hist'))
df_late_pay_6mp = spark.read.format('delta').load(os.path.join(dir_dev_parent, 'fea_late_pay_6mp'))

## payment and billing existing features 
df_fea_payment_6ycle = spark.read.format('delta').load(os.path.join(dir_fs_data_parent, 'fea_payment_cycle_rolling_6'))
df_fea_payment_3ycle = spark.read.format('delta').load(os.path.join(dir_fs_data_parent, 'fea_payment_cycle_rolling_3'))
df_fea_bill_6bcycle = spark.read.format('delta').load(os.path.join(dir_fs_data_parent, 'fea_bill_cycle_billing_6')) 

# credit score 

df_credit_score = spark.read.format('delta').load(os.path.join(dir_fs_data_parent, 'fea_credit_score'))


## risk scores 
df_aod30d_score = spark.read.format('delta').load(os.path.join(dir_dev_parent_master, 'predict_aod30d_propensity_v3'))
df_wo_score = spark.read.format('delta').load(os.path.join(dir_mls_parent, 'reporting_cycle_type=calendar cycle'))

# unit base 
df_fs_unitbase = spark.read.format('delta').load(dir_fs_unit_base)

## movement 
df_wo_mvnt = spark.read.format('delta').load(dir_wo_mvnt)
df_aod30_mvnt = spark.read.format('delta').load(dir_aod30_mvnt)
df_aod05_mvnt = spark.read.format('delta').load(dir_aod05_mvnt)
df_aod15_mvnt = spark.read.format('delta').load(dir_aod15_mnvt)
df_aod23_mvnt = spark.read.format('delta').load(dir_aod_23_mvnt)

# COMMAND ----------

display(df_fea_bill_6bcycle.limit(10))

# COMMAND ----------

display(df_fea_payment_6ycle.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC #### parameters

# COMMAND ----------

ls_prm_key = ['fs_cust_id', 'fs_srvc_id', 'fs_acct_id']
ls_aod_join_key = ['fs_acct_id', 'fs_cust_id']
ls_reporting_date_key = ['reporting_date', 'reporting_cycle_type']

# COMMAND ----------

ls_reporting_date =  ['2024-03-31', '2024-02-29', '2024-01-31']
vt_reporting_cycle_type = 'calendar cycle'
vt_lookback_billing_cycle = 6
ls_score_fields_select = [
'propensity_score_raw',
'propensity_score_cb',
'propensity_score',
'propensity_top_ntile',
'propensity_segment_qt',
'propensity_segment_pbty'
]

# COMMAND ----------

# MAGIC %md
# MAGIC #### transformation

# COMMAND ----------

# DBTITLE 1,active bill acct tenure
display(
  df_fs_unitbase
  .filter(f.col('reporting_date').isin(*ls_reporting_date))
  .filter(f.col('reporting_cycle_type') == vt_reporting_cycle_type)
  .select(*ls_reporting_date_key, 'fs_acct_id', 'billing_acct_tenure')
  .distinct()
  .groupBy('reporting_date')
  .agg(f.median('billing_acct_tenure')
       , f.percentile_approx('billing_acct_tenure', 0.9)
       ,  f.percentile_approx('billing_acct_tenure', 0.75)
       ,  f.percentile_approx('billing_acct_tenure', 0.25)
       ,  f.percentile_approx('billing_acct_tenure', 0.15)
       )
)

# COMMAND ----------

# DBTITLE 1,check cnt
display(
  df_late_pay_6mp
  .filter(f.col('reporting_date').isin(*ls_reporting_date))
  .filter(f.col('reporting_cycle_type') == vt_reporting_cycle_type)
  .select(*ls_reporting_date_key, *ls_prm_key)
  .groupBy('reporting_date')
  .agg(f.count('*'))
)

display(
  df_coll_action
  .filter(f.col('reporting_date').isin(*ls_reporting_date))
  .filter(f.col('reporting_cycle_type') == vt_reporting_cycle_type)
  .select(*ls_reporting_date_key, *ls_prm_key)
  .groupBy('reporting_date')
  .agg(f.count('*'))
)

display(
  df_consec_latepay
  .filter(f.col('reporting_date').isin(*ls_reporting_date))
  .filter(f.col('reporting_cycle_type') == vt_reporting_cycle_type)
  .select(*ls_reporting_date_key, *ls_prm_key)
  .groupBy('reporting_date')
  .agg(f.count('*'))
)

display(
  df_prodcut_acq_12m
  .filter(f.col('reporting_date').isin(*ls_reporting_date))
  .filter(f.col('reporting_cycle_type') == vt_reporting_cycle_type)
  .select(*ls_reporting_date_key, *ls_prm_key)
  .groupBy('reporting_date')
  .agg(f.count('*'))
)

display(
  df_fea_payment_6ycle
  .filter(f.col('reporting_date').isin(*ls_reporting_date))
  .filter(f.col('reporting_cycle_type') == vt_reporting_cycle_type)
  .select(*ls_reporting_date_key, *ls_prm_key)
  .groupBy('reporting_date')
  .agg(f.count('*'))
)

display(
  df_fea_bill_6bcycle
  .filter(f.col('reporting_date').isin(*ls_reporting_date))
  .filter(f.col('reporting_cycle_type') == vt_reporting_cycle_type)
  .select(*ls_reporting_date_key, *ls_prm_key)
  .groupBy('reporting_date')
  .agg(f.count('*'))
)



# COMMAND ----------

# MAGIC %md
# MAGIC #### transformation on scores

# COMMAND ----------

df_wo_base_curr_00 = (df_wo_score
        .filter(f.col('reporting_date').isin(*ls_reporting_date))
        .filter(f.col('reporting_cycle_type') == vt_reporting_cycle_type)
        .select(*[f.col(field).alias(f"wo_{field}") for field in ls_score_fields_select]
                +ls_prm_key+ls_reporting_date_key 
                )
        )

df_aod_base_curr_00 = (df_aod30d_score
        .filter(f.col('reporting_date').isin(*ls_reporting_date))
        .filter(f.col('reporting_cycle_type') == vt_reporting_cycle_type)
        .select(*[f.col(field).alias(f"aod_{field}") for field in ls_score_fields_select]
                + ls_prm_key +ls_reporting_date_key 
        )
        )

# COMMAND ----------

# MAGIC %md
# MAGIC #### selection on payment and bill features & credit score 

# COMMAND ----------

# DBTITLE 1,payment features
df_payment_6cycle= (df_fea_payment_6ycle
        .filter(f.col('reporting_date').isin(*ls_reporting_date))
        .filter(f.col('reporting_cycle_type') == 'calendar cycle')
        .select(*ls_prm_key, *ls_reporting_date_key
                , 'payment_cnt_tot_6cycle'
                , 'payment_amt_avg_6cycle'
                , 'payment_interval_days_avg_6cycle'
                )
        )

# COMMAND ----------

# DBTITLE 1,bill features
df_bill_6cycle= (df_fea_bill_6bcycle
        .filter(f.col('reporting_date').isin(ls_reporting_date))
        .filter(f.col('reporting_cycle_type') == 'calendar cycle')
        .select(*ls_prm_key, *ls_reporting_date_key
                , 'bill_payment_timeliness_status'
                , 'bill_overdue_days'
                , 'bill_charge_amt'
                , 'bill_charge_amt_delta_max_6bmnth'
                , 'bill_cnt_late_pct_6bmnth'
                , 'bill_overdue_days_late_avg_6bmnth'
                , 'bill_cnt_miss_pct_6bmnth'
                , 'bill_overdue_days_miss_avg_6bmnth'
                )
        )

# COMMAND ----------

df_fea_creditscore = (df_credit_score
        .filter(f.col('reporting_date').isin(*ls_reporting_date))
        .filter(f.col('reporting_cycle_type') == 'calendar cycle')
        .select(*ls_prm_key, *ls_reporting_date_key,'credit_score', 'credit_score_segment' )
        )

# COMMAND ----------

display(df_fea_creditscore.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC #### check bill charge percentile

# COMMAND ----------

## check percentile value 
display(df_bill_6cycle
        .agg(f.min('bill_charge_amt_delta_max_6bmnth')
             , f.max('bill_charge_amt_delta_max_6bmnth')
             , f.percentile_approx('bill_charge_amt_delta_max_6bmnth', 0.9)
             , f.percentile_approx('bill_charge_amt_delta_max_6bmnth', 0.95)
             , f.percentile_approx('bill_charge_amt_delta_max_6bmnth', 0.80)    
             , f.percentile_approx('bill_charge_amt_delta_max_6bmnth', 0.99)
             , f.percentile_approx('bill_charge_amt_delta_max_6bmnth', 0.75)
             , f.percentile_approx('bill_charge_amt_delta_max_6bmnth', 0.5)
             )
        )

## take 80% percentile as grouping 
display(df_bill_6cycle
         .withColumn('charge_delta_max_cat', f.when(f.col('bill_charge_amt_delta_max_6bmnth').between(0,1.5), f.lit('0-1.5'))
                                               .when( (f.col('bill_charge_amt_delta_max_6bmnth') > 1.5)
                                                       & (f.col('bill_charge_amt_delta_max_6bmnth') <=2.5), f.lit('1.5-2.5'))
                                               .when(f.col('bill_charge_amt_delta_max_6bmnth') >2.5, f.lit('2.5+') )
                     )
         .withColumn('bill_shift_flag', f.when(f.col('bill_charge_amt_delta_max_6bmnth') < 1.5
                                               , f.lit('N')
                                               )
                                        .when(f.col('bill_charge_amt_delta_max_6bmnth') >= 1.5
                                              , f.lit('Y')
                                              )
                     
                     )
         .groupBy('reporting_date','bill_shift_flag' )
         .agg(f.count('*')
              , f.countDistinct('fs_acct_id')
              )
         )

# COMMAND ----------

# DBTITLE 1,Charge Shift Category
df_bill_6cycle =(df_bill_6cycle
         .withColumn('charge_delta_max_cat', f.when(f.col('bill_charge_amt_delta_max_6bmnth') <=1.5, f.lit('L'))
                                               .when( (f.col('bill_charge_amt_delta_max_6bmnth') > 1.5)
                                                       & (f.col('bill_charge_amt_delta_max_6bmnth') <=2.5), f.lit('M')
                                                       )
                                               .when(f.col('bill_charge_amt_delta_max_6bmnth') >2.5, f.lit('H'))
                     )
        .withColumn('bill_shift_flag', f.when(f.col('bill_charge_amt_delta_max_6bmnth') < 1.5
                                               , f.lit('N')
                                               )
                                        .when(f.col('bill_charge_amt_delta_max_6bmnth') >= 1.5
                                              , f.lit('Y')
                                              )
                     
                     )
         #.filter(f.col('bill_payment_timeliness_status')!= 'credit_bill')
         # .filter(f.col('charget_delta_max_cat').isin('2.5+'))
)


# COMMAND ----------

# MAGIC %md
# MAGIC #### get next months movement 

# COMMAND ----------

display(df_aod05_mvnt.limit(10))

# COMMAND ----------

# aod30
df_aod30_mvnt_nxt_mnth = (df_aod30_mvnt
        .filter(f.col('reporting_date').isin(['2024-02-29' , '2024-03-31', '2024-04-30']))
        .withColumnRenamed('reporting_date', 'target_reporting_date')
        .withColumn('reporting_date', f.last_day(f.add_months('target_reporting_date',-1)))
        .select('reporting_date'
                , 'fs_acct_id'
                , f.col('movement_date').alias('aod30_movement_date')
                ,f.col('target_reporting_date').alias('aod30_target_reporting_date')
                )
)

# aod 05
df_aod05_mvnt_nxt_mnth = (df_aod05_mvnt
        .filter(f.col('reporting_date').isin(['2024-02-29' , '2024-03-31', '2024-04-30']))
        .withColumnRenamed('reporting_date', 'target_reporting_date')
        .withColumn('reporting_date', f.last_day(f.add_months('target_reporting_date',-1)))
        .select('reporting_date'
                , 'fs_acct_id'
                , f.col('movement_date').alias('aod05_movement_date')
                ,f.col('target_reporting_date').alias('aod05_target_reporting_date')
                )
)
                          

# aod 15 
df_aod15_mvnt_nxt_mnth = (df_aod15_mvnt
        .filter(f.col('reporting_date').isin(['2024-02-29' , '2024-03-31', '2024-04-30']))
        .withColumnRenamed('reporting_date', 'target_reporting_date')
        .withColumn('reporting_date', f.last_day(f.add_months('target_reporting_date',-1)))
        .select('reporting_date'
                , 'fs_acct_id'
                , f.col('movement_date').alias('aod15_movement_date')
                ,f.col('target_reporting_date').alias('aod15_target_reporting_date')
                )
)
# aod 23 

df_aod23_mvnt_nxt_mnth = (df_aod23_mvnt
        .filter(f.col('reporting_date').isin(['2024-02-29' , '2024-03-31', '2024-04-30']))
        .withColumnRenamed('reporting_date', 'target_reporting_date')
        .withColumn('reporting_date', f.last_day(f.add_months('target_reporting_date',-1)))
        .select('reporting_date'
                , 'fs_acct_id'
                , f.col('movement_date').alias('aod23_movement_date')
                ,f.col('target_reporting_date').alias('aod23_target_reporting_date')
                )
)

# wo
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

# MAGIC %md
# MAGIC #### combine all

# COMMAND ----------

# DBTITLE 1,unit base left join features
df_output = (
  df_fs_unitbase
  .filter(f.col('reporting_date').isin(*ls_reporting_date))
  .filter(f.col('reporting_cycle_type') == vt_reporting_cycle_type)
  .select(*ls_reporting_date_key, *ls_prm_key)
  .join(df_coll_action, ls_reporting_date_key+ls_prm_key, 'inner')
  .join(df_consec_latepay, ls_reporting_date_key+ls_prm_key, 'inner')
  .join(df_prodcut_acq_12m, ls_reporting_date_key+ls_prm_key, 'inner')
  .join(df_late_pay_6mp,ls_reporting_date_key+ls_prm_key, 'inner') 
  .join(df_wo_base_curr_00, ls_reporting_date_key + ls_prm_key, 'inner')
  .join(df_aod_base_curr_00, ls_reporting_date_key+ ls_prm_key , 'left')
  .join(df_aod30_mvnt_nxt_mnth, ['fs_acct_id', 'reporting_date'], 'left')
  .join(df_aod05_mvnt_nxt_mnth, ['fs_acct_id', 'reporting_date'], 'left')
  .join(df_aod15_mvnt_nxt_mnth, ['fs_acct_id', 'reporting_date'], 'left')
  .join(df_aod23_mvnt_nxt_mnth, ['fs_acct_id', 'reporting_date'], 'left')
  .join(df_wo_mvnt_nxt_mnth, ['fs_acct_id', 'reporting_date'], 'left')
  .join(df_bill_6cycle,ls_reporting_date_key + ls_prm_key, 'left' )
  .join(df_payment_6cycle, ls_reporting_date_key + ls_prm_key, 'left')
  .join(df_fea_creditscore, ls_reporting_date_key+ ls_prm_key, 'left')
  .withColumn('actual_aod', f.when(f.col('aod30_movement_date').isNotNull(), 1).otherwise(0))
  .withColumn('actual_aod05', f.when(f.col('aod05_movement_date').isNotNull(), 1).otherwise(0))
  .withColumn('actual_aod15', f.when(f.col('aod15_movement_date').isNotNull(), 1).otherwise(0))
  .withColumn('actual_aod23', f.when(f.col('aod23_movement_date').isNotNull(), 1).otherwise(0))
  .withColumn('actual_wo',  f.when(f.col('wo_movement_date').isNotNull(), 1).otherwise(0))      
  .withColumn('aod_segment',  f.when(f.col('aod_propensity_top_ntile').between(97,100), f.lit('H'))
                                     .when(f.col('aod_propensity_top_ntile').between(80,96), f.lit('M'))
                                     .when(f.col('aod_propensity_top_ntile').between(0,79), f.lit('L') )
                                     .otherwise(f.lit('in_aod30'))
                     ) 
  # L1 segmentation 
  .withColumn('wo_segment',  f.when(f.col('wo_propensity_top_ntile').between(97,100), f.lit('H'))
                                     .when(f.col('wo_propensity_top_ntile').between(80,96), f.lit('M'))
                                     .when(f.col('wo_propensity_top_ntile').between(0,79), f.lit('L') )
                                     .otherwise(f.lit('X'))
                     )
  .withColumn('L1_segment', f.when( (f.col('wo_segment').isin('H')) & (f.col('aod_segment').isin('in_aod30') )
                                      | (f.col('wo_segment').isin('H')) & (f.col('aod_segment').isin('H') )
                                   , f.lit('FS'))
                             .when( (f.col('wo_segment').isin('L') & f.col('aod_segment').isin('L')) 
                                   | (f.col('wo_segment').isin('M') & f.col('aod_segment').isin('L')) 
                                   , f.lit('OT'))
                             .otherwise(f.lit('PFM'))
  )   
)

# COMMAND ----------

display(df_output
        .groupBy('reporting_date')
        .agg(f.count('*').alias('count')
             , f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_srvc_id')
             , f.sum('actual_aod').alias('actual_aod')
             , f.sum('actual_wo').alias('actual_wo')
            #  , f.sum('actual_aod05').alias('actual_aod05')
            #  , f.sum('actual_aod15').alias('actual_aod15')
            #  , f.sum('actual_aod23').alias('actual_aod23')
            )
        )     


# COMMAND ----------

display(df_output
        .filter(f.col('fs_acct_id') == '912722')
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### L1 Segmentation

# COMMAND ----------

display(df_output
        .groupBy( 'wo_segment', 'aod_segment', 'reporting_date')
        .agg(f.count('*').alias('count')
             , f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_srvc_id')
             , f.sum('actual_aod').alias('actual_aod')
             , f.sum('actual_wo').alias('actual_wo')
             , f.sum('actual_aod05').alias('actual_aod05')
             , f.sum('actual_aod15').alias('actual_aod15')
             , f.sum('actual_aod23').alias('actual_aod23')
        )
        )

display(df_output
        .groupBy('L1_segment', 'reporting_date')
        .agg(f.count('*').alias('count')
             , f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_srvc_id')
             , f.sum('actual_aod').alias('actual_aod')
             , f.sum('actual_wo').alias('actual_wo')
             , f.sum('actual_aod05').alias('actual_aod05')
             , f.sum('actual_aod15').alias('actual_aod15')
             , f.sum('actual_aod23').alias('actual_aod23')
        )
        .withColumn('sum', f.sum('count').over(Window.partitionBy('reporting_date')))
        .withColumn('pct', f.col('count')/ f.col('sum'))
        )

# COMMAND ----------

display(df_output.filter(f.col('fs_acct_id')
                         == '477943749'
                         )
        
        )

# COMMAND ----------

# MAGIC %md
# MAGIC #### L2 Segmentation Part 1

# COMMAND ----------

# num_of_active_srvc_cnt:decimal(38,0)
# ifp_prm_dvc_flag:string
# ifp_acct_accs_flag:string
# ifp_acct_accs_cnt:long
# ifp_acct_dvc_cnt:long
# product_holding:string
# recent_acq_6mp:string
# cnt_product_add_in_6mp:integer
# reporting_date_6mp:date
# cnt_late_pay_6bcycle_6mp:long
# avg_overdue_days_6bcycle_6mp:double

df_l2_segment_part1= (df_output
        .filter(f.col('L1_segment') == 'FS')
        .withColumn('L2_Segment_part1', 
                    f.when( ( (f.col('cnt_product_add_in_12mp') >=1 ) & (f.col('bill_shift_flag') == 'Y') )
                              |  ( (f.col('cnt_product_add_in_12mp') < 1 ) & (f.col('bill_shift_flag') == 'Y') )
                              |  ( (f.col('cnt_product_add_in_12mp') >= 1 ) & (f.col('bill_shift_flag') == 'N') )
                                & (   (f.col('cnt_late_pay_6bcycle') > f.col('cnt_late_pay_6bcycle_6mp' ) )
                                        | (  (f.col('cnt_late_pay_6bcycle') >=3 )  
                                        & (f.col('cnt_late_pay_6bcycle_6mp') >=3) )
                                   )
                         , 'Overloaded'
                          )
                    .when(  ((f.col('cnt_product_add_in_12mp') <1) & (f.col('bill_shift_flag') == 'N' ))
                          & (  ( f.col('cnt_late_pay_6bcycle') > f.col('cnt_late_pay_6bcycle_6mp') )
                                 | ( 
                                    (f.col('cnt_late_pay_6bcycle') >3 )  
                                        & (f.col('cnt_late_pay_6bcycle_6mp') >3) 
                                    )  
                             )                                          
                                  , 'Struggling Payer')
                                        #  .when(f.col('cnt_late_pay_6bcycle') <=f.col('cnt_late_pay_6bcycle_6mp')
                                        #     , 'RecentRecovery'
                                        #        )
                    .otherwise('Other')
        )                    
       .select(*ls_reporting_date_key,*ls_prm_key, 'L2_Segment_part1')
        )

# COMMAND ----------

display(df_l2_segment_part1
        .groupBy('reporting_date', 'L2_Segment_part1')
        .agg(f.count('*')
             , f.countDistinct('fs_srvc_id')
             , f.countDistinct('fs_acct_id')
            #  , f.sum('actual_wo')
            #  , f.avg('bill_charge_amt')
             )
        )


# COMMAND ----------

display(df_l2_segment_part1
        .filter(f.col('reporting_date') == '2024-03-31')
        #.filter(f.col('L2_Segment_part1') == 'OverCommit')
        .groupBy('cnt_late_pay_6bcycle_6mp', 'cnt_late_pay_6bcycle', 'cnt_product_add_in_12mp','bill_shift_flag'
                 , 'L2_Segment_part1' 
                 )
        .agg(
             f.countDistinct('fs_acct_id')   
             , f.sum('actual_wo')
             , f.avg('bill_charge_amt')
        )
        )


# COMMAND ----------

display(df_output
        .filter(f.col('L1_segment') == 'FS')
        .groupBy('cnt_late_pay_6bcycle_6mp', 'cnt_product_add_in_12mp', 'reporting_date', 'cnt_late_pay_6bcycle')
        .agg(f.count('*').alias('count')
             , f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_srvc_id')
             , f.sum('actual_aod').alias('actual_aod')
             , f.sum('actual_wo').alias('actual_wo'))
        )

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC #### L2 Segmentation Part 2

# COMMAND ----------


display(df_output
        .filter(f.col('L1_segment') == 'PFM')
        .groupBy( 'reporting_date', 'cnt_late_pay_6bcycle_6mp')
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             , f.sum('actual_aod').alias('actual_aod')
             , f.sum('actual_wo').alias('actual_wo')
             )
)


# COMMAND ----------

## features 
# num_consecutive_late_pay:long
# cnt_late_pay_6bcycle:long
# avg_overdue_days_6bcycle:double
# avg_late_sequence_length:double
# payment_behavior_label:string
# cnt_late_ppay_6bcycle_6mp 

df_l2_segment_part2 = (df_output
        .filter(f.col('L1_segment') == 'PFM')
        .withColumn('pct_pay_after_coll_12m', f.col('cnt_pay_after_coll_12m')/f.col('cnt_in_coll_action_12m'))
        .withColumn('L2_segment_part2'
                , f.when( (f.col('pct_pay_after_coll_12m') > 0.85)
                            &(f.col('AVG_DAYS_TO_PAYMENT_AFTER_COLL') <=1)
                         , 'Intentional Offender'
                         ) # 85% of the time pay wihtin 1 days of collection actions 
                  .when(f.col('CNT_LATE_PAY_6BCYCLE') > 0.5 * vt_lookback_billing_cycle
                        , 'Chronic Late'
                        )
                  .when(f.col('CNT_LATE_PAY_6BCYCLE') < 0.5 * vt_lookback_billing_cycle
                        , 'Sporadic Late'
                        )
                  .when( (f.col('CNT_LATE_PAY_6BCYCLE') == 0.5 * vt_lookback_billing_cycle)
                         & (f.col('cnt_late_pay_6bcycle_6mp') > 0.5 * vt_lookback_billing_cycle )
                         , 'Chronic Late'
                        )
                  .when( (f.col('CNT_LATE_PAY_6BCYCLE') == 0.5 * vt_lookback_billing_cycle)
                         & (f.col('cnt_late_pay_6bcycle_6mp') < 0.5 * vt_lookback_billing_cycle )
                         , 'Sporadic Late')
                  .when( (f.col('CNT_LATE_PAY_6BCYCLE') == 0.5 * vt_lookback_billing_cycle)
                         & (f.col('cnt_late_pay_6bcycle_6mp') == 0.5 * vt_lookback_billing_cycle )
                          & (f.col('num_consecutive_late_pay') >= 0.5 * 0.5* vt_lookback_billing_cycle ) # 1.5
                         , 'Chronic Late') 
                  .when( (f.col('CNT_LATE_PAY_6BCYCLE') == 0.5 * vt_lookback_billing_cycle) 
                         & (f.col('cnt_late_pay_6bcycle_6mp') == 0.5 * vt_lookback_billing_cycle ) # 3
                          & (f.col('num_consecutive_late_pay') < 0.5 * 0.5 * vt_lookback_billing_cycle ) #1.5
                         , 'Sporadic Late')
                  .otherwise('other')
        )
        .select(*ls_prm_key, *ls_reporting_date_key, 'L2_segment_part2')
)

# COMMAND ----------

display(df_l2_segment_part2
        .groupBy('L2_segment_part2', 'reporting_date')
        .agg(f.count('*').alias('count')
             , f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_srvc_id')
            # , f.sum('actual_aod').alias('actual_aod')
            # , f.sum('actual_wo').alias('actual_wo'))
        )
)

# COMMAND ----------

display(df_l2_segment_part2
        .filter(f.col('L2_segment_part2') == 'Sporadic Late')
        .limit(10)
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### combine segmenation

# COMMAND ----------

display(df_output
        .join(df_l2_segment_part1, ls_prm_key + ls_reporting_date_key, 'left')
        .join(df_l2_segment_part2, ls_prm_key + ls_reporting_date_key, 'left')
        .select(*ls_prm_key, *ls_reporting_date_key
                , 'L1_segment', 'L2_segment_part2', 'L2_Segment_part1'
                , 'actual_aod', 'actual_wo'
        )
        .groupBy( 'reporting_date' ,'L1_segment', 'L2_segment_part2', 'L2_Segment_part1')
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_srvc_id')
             , f.sum('actual_aod').alias('actual_aod')
             , f.sum('actual_wo').alias('actual_wo'))
        )

# COMMAND ----------

df_combine_all = (df_output
                    .join(df_l2_segment_part1, ls_prm_key + ls_reporting_date_key, 'left')
                    .join(df_l2_segment_part2, ls_prm_key + ls_reporting_date_key, 'left')
                    .withColumn('L2_combine', f.coalesce('L2_Segment_part1', 'L2_segment_part2', 'L1_segment'))
                )

# COMMAND ----------

display(df_combine_all
        .groupBy( 'reporting_date' ,'L1_segment', 'L2_segment_part2', 'L2_Segment_part1', 'L2_combine')
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_srvc_id')
             , f.sum('actual_aod').alias('actual_aod')
             , f.sum('actual_wo').alias('actual_wo'))
        )

# COMMAND ----------

# MAGIC %md
# MAGIC #### one off export to snowflake 

# COMMAND ----------

#dbutils.fs.rm('/mnt/ml-lab/dev_users/dev_sc/99_misc/cohort_seg/df_combine_all_v4', True)

# COMMAND ----------

export_data(
            df = df_combine_all
            , export_path = '/mnt/ml-lab/dev_users/dev_sc/99_misc/cohort_seg/df_combine_all_v4' 
            # added aod all for next month # new segment 
            , export_format = "delta"
            , export_mode = "append"
            , flag_overwrite_schema = True
            , flag_dynamic_partition = True
            , ls_dynamic_partition = ["reporting_date"]
        )




# COMMAND ----------

# MAGIC %md
# MAGIC ##### snowflake connector

# COMMAND ----------

# ------------ login to snowflake
password = dbutils.secrets.get(scope = "auea-kv-sbx-dxdtlprdct01", key = "sfdbrsdskey")

options = {
  "sfUrl": "vodafonenz_prod.australia-east.azure.snowflakecomputing.com/", 
  "sfUser": "SVC_LAB_DS_DATABRICKS",
  "pem_private_key": password.replace('\\n', '\n'),
  "sfDatabase": "LAB_ML_STORE",
  "sfSchema": "SANDBOX",
  "sfWarehouse": "LAB_DS_WH_SCALE"
}

# COMMAND ----------

(
    df_combine_all
    .write
    .format("snowflake")
    .options(**options)
    .option("dbtable", "lab_ml_store.sandbox.sc_one_off_cohort_seg_combine_all_v4")
    .mode("overwrite")
    .save()
)

# COMMAND ----------

display(df_combine_all
        .groupBy('reporting_date')
        .agg(f.count('*'))
        )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Test output

# COMMAND ----------

df_test = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/99_misc/cohort_seg/df_combine_all_v4')

# COMMAND ----------

display(df_test
        .groupBy('reporting_date')
        .agg(f.count('*'))
        )
