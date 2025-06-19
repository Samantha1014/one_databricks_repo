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
from pyspark.sql import DataFrame, Window
from typing import Optional, List, Dict, Tuple

# COMMAND ----------

# MAGIC %run "./utils_stratified_sampling"

# COMMAND ----------

#cohort segmentaiton 
df_cohort_all = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/99_misc/cohort_seg/df_combine_all')

# payment and billing 
df_payment_latest = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_payment_latest')
df_fea_bill = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_bill_cycle_billing_6')

# unit base 
df_fea_unitbase = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base')

# stag payment and bill 
df_stag_bill = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d200_staging/d299_src/stg_brm_bill_t')
df_stag_payment = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d200_staging/d299_src/stg_brm_payment_hist') 

# COMMAND ----------

ls_joining_key = ['fs_acct_id', 'fs_cust_id', 'reporting_date', 'reporting_cycle_type']
vt_reporting_date = (
    df_fea_unitbase
    .select(
        f.max(
            f.when(
                f.col('reporting_date') <= f.current_timestamp(),
                f.col('reporting_date')
            )
        ).alias('latest_reporting_date') 
    )
    .collect()[0]['latest_reporting_date']
)

# COMMAND ----------

vt_reporting_date 

# COMMAND ----------

df_late_payer = (
        df_cohort_all
        .select('fs_acct_id', 'fs_cust_id', 'reporting_date', 'L2_combine', 'reporting_cycle_type'
                , 'wo_propensity_top_ntile'
                )
        .distinct()
        #.filter(f.col('reporting_cycle_type') == 'calendar cycle')
        .filter(f.col('L2_combine').isin('Chronic Late', 'Sporadic Late'
                                         ,'Intentional Offender'
                                         ,'Struggling Payer'
                                         , 'Overloaded'
                                         )
                )
        .filter(f.col('reporting_date') >= '2024-01-01')
        .withColumn('rnk', f.row_number().over(Window
                                               .partitionBy('fs_acct_id','fs_cust_id' , 'reporting_date')
                                               .orderBy(f.desc('wo_propensity_top_ntile'))
                                              )
                    )
        .filter(f.col('rnk') == 1)
         )

# COMMAND ----------

display(df_late_payer.limit(10))

# COMMAND ----------

df_bill_cycle = (df_fea_bill
        .filter(f.col('reporting_date') >= '2024-01-01')
       # .filter(f.col('reporting_cycle_type') == 'calendar cycle')
        .select('reporting_date', 'reporting_cycle_type','fs_acct_id', 'fs_cust_id',  'bill_cycle_start_date'
                , 'bill_cycle_end_date', 'bill_due_date', 'bill_due_amt', 'bill_no', 'bill_close_date'
                )
        .distinct()
)

# COMMAND ----------

display(df_bill_cycle
        .groupBy('reporting_date')
        .agg(f.countDistinct('fs_acct_id')
            , f.countDistinct('fs_cust_id')
            , f.count('*')
             )
        )

# COMMAND ----------

display(df_late_payer
        .groupBy('reporting_date', 'L2_combine')
        .agg(f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_cust_id')
             , f.count('*')
             )
        )

# COMMAND ----------

display(df_late_payer
       .join(df_bill_cycle, ls_joining_key, 'inner')
       .groupBy('reporting_date')
       .agg(f.countDistinct('fs_acct_id')
            #, f.countDistinct('fs_srvc_id')
            , f.count('*')
            )
)

# COMMAND ----------

# DBTITLE 1,pay before due flag
df_pay_before_due = (df_late_payer
       .join(df_bill_cycle, ls_joining_key, 'inner')
       .distinct()
       .join(df_stag_payment
             .withColumn('rnk', f.row_number().over(Window.partitionBy('item_no').orderBy(f.desc('payment_mod_dttm'))) 
                         )
             .filter(f.col('rnk') == 1)
             .filter(f.col('item_poid_type').isin('/item/payment'))
             .select('fs_acct_id','item_no','payment_create_date','item_total')
             .distinct()
             , ['fs_acct_id']
             , 'left'
       )
       .withColumn('pay_before_due'
                   , f.when( f.col('payment_create_date').between(f.col('bill_cycle_end_date'), f.col('bill_due_date'))
                            , 'Y'
                            )
                      .otherwise('N')
                   )
       .distinct()
       .filter(f.col('pay_before_due') == 'Y')
       .groupBy('bill_no', 'fs_acct_id', 'fs_cust_id',  'reporting_date', 'reporting_cycle_type'
                # , 'bill_cycle_start_date', 'bill_cycle_end_date'
                # , 'bill_due_date'
                # , 'bill_due_amt'
                # , 'bill_close_date'
                , 'pay_before_due'
                )
      .agg(f.sum('item_total').alias('item_total_agg'))
     # .filter(f.col('fs_acct_id') == '1044142')
  )


# COMMAND ----------

display(df_pay_before_due
        .groupBy('reporting_date')
        .agg(f.countDistinct('fs_acct_id')
             , f.count('*')
             )
        )

# COMMAND ----------

# DBTITLE 1,check records & pay full%
display(df_late_payer
        .join(df_bill_cycle, ls_joining_key, 'inner')
        .join(df_pay_before_due
              ,  ['fs_acct_id', 'fs_cust_id' , 'reporting_date', 'reporting_cycle_type', 'bill_no']
              , 'left')
        .withColumn('pct_pay_to_due', 
                    f.when(f.col('pay_before_due') == 'Y'
                           , f.col('item_total_agg')/f.col('bill_due_amt')
                          )
                     .otherwise(0)
                    )
        .withColumn('pay_full', f.when(f.col('pct_pay_to_due')<=-1, 'full' )
                                 .when(  (f.col('pct_pay_to_due') < 0 ) & 
                                         (f.col('pct_pay_to_due') > -1 ) 
                                         , 'partial'
                                       )
                                 .when(f.col('pay_before_due').isNull(), 'no_pay')
                                 .when(f.col('bill_due_amt')<=0, 'credit_bill')
                                 .otherwise('other')
                    )
        .filter(f.col('pay_full').isin('other'))
        .filter(f.col('pay_before_due') == 'Y')
        # .agg(f.min('item_total_agg')
        #      , f.max('item_total_agg')
        # )
        #  .groupBy('pay_full', 'reporting_date', 'L2_combine', 'pay_before_due')
        # .agg(f.countDistinct('fs_acct_id')
        #      , f.count('*')
        #      , f.avg('pct_pay_to_due')
        #      )
        )

# COMMAND ----------

# MAGIC %md
# MAGIC #### s03 dev

# COMMAND ----------

vt_reporting_date = '2025-02-09'

# COMMAND ----------

display(df_late_payer
        .filter(f.col('reporting_date') == vt_reporting_date)
        )

# COMMAND ----------

display(df_fea_unitbase.limit(10))

# COMMAND ----------

# DBTITLE 1,bill overdue days 6bcycle ntile
df_output = (df_late_payer
        .filter(f.col('reporting_date') == vt_reporting_date)
        .join(
            df_fea_bill
            .filter(f.col('reporting_date') == vt_reporting_date)
            .select('fs_acct_id', 'fs_cust_id', 'bill_overdue_days_late_avg_6bmnth', 'bill_overdue_days')
            .distinct()
            , ['fs_acct_id', 'fs_cust_id']
            , 'left'
        )
        # .join(df_fea_unitbase
        #       .filter(f.col('reporting_date') == vt_reporting_date)
        #       .select('fs_acct_id', 'fs_cust_id', 'billing_acct_tenure','num_of_active_srvc_cnt' , 'srvc_tenure')
        #       .distinct()
        #       , ['fs_acct_id', 'fs_cust_id']
        #       , 'left'
        # )
        .withColumn('rank', f.col('bill_overdue_days_late_avg_6bmnth'))
        .withColumn(
                    "bill_overdays_late_6bmnth_ntile"
                    , f.ntile(20).over(
                        Window
                        .orderBy(f.asc("rank"))
                    )
                )
       # .withColumn('rank', f.col('bill_overdue_days_late_avg_6bmnth'))
        .withColumn(
                    "bill_overdue_days_ntile"
                    , f.ntile(20).over(
                        Window
                        .orderBy(f.asc("bill_overdue_days"))
                    )
                )
        .drop('rnk', 'rank')
        .distinct()
        # .groupBy('bill_overdays_late_6bmnth_ntile'
        #         # , 'wo_propensity_top_ntile'
        #          )
        # .agg(f.count('fs_acct_id')       
        )


# COMMAND ----------

display(df_output.limit(10))

# COMMAND ----------

df_output_chronic = (
    df_output
    #.select('fs_acct_id', 'fs_cust_id', '')
    .filter(f.col('L2_combine') == 'Chronic Late')
)

df_output_sporadic = (
    df_output
    .filter(f.col('L2_combine') == 'Sporadic Late')
)

df_output_struggle = (
    df_output
    .filter(f.col('L2_combine') == 'Struggling Payer')
    )

df_output_overload = (
    df_output
    .filter(f.col('L2_combine') == 'Overloaded')
    )

df_output_offender = (
  df_output
  .filter(f.col('L2_combine') == 'Intentional Offender')
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### s04 sample generation 

# COMMAND ----------

# DBTITLE 1,sample creation 1
# Control group creation
vt_param_seed = 20
vt_param_sample_req = 2500
#vt_param_proportion = 0.25
ls_param_strata_fields = [
    "bill_overdays_late_6bmnth_ntile"
    , 'bill_overdue_days_ntile'
  #  , "mmc_top_ntile"
  #  , "dr_top_ntile"
]
#vt_param_priority_field = "wallet_control_flag"
#dict_param_priority_groups = {'Y':2}

df_base_control, df_base_target = generate_sample(
    df= df_output_overload
    , size=vt_param_sample_req
    #, proportion = vt_param_proportion
    , strata=ls_param_strata_fields
    #, priority_field=vt_param_priority_field
    #, priority_groups=dict_param_priority_groups
    , seed=15
)



# COMMAND ----------

# DBTITLE 1,check cnt
display(df_output_overload
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             )
)

display(df_base_control
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             )
)

display(df_base_target
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             )
)

# COMMAND ----------

# DBTITLE 1,check overlap
display(df_base_control
        .join(df_base_target, ['fs_acct_id', 'fs_cust_id'], 'inner')
        )

# COMMAND ----------

evaluate_sample(
    df_base_control
    , df_base_target
    , [ "bill_overdays_late_6bmnth_ntile", 'bill_overdue_days_ntile',  'wo_propensity_top_ntile']
)

# COMMAND ----------

df_overload_control = df_base_control
df_overload_target = df_base_target

# COMMAND ----------

# DBTITLE 1,sample creation 2
# Control group creation
vt_param_seed = 20
vt_param_sample_req = 2500
#vt_param_proportion = 0.25
ls_param_strata_fields = [
    "bill_overdays_late_6bmnth_ntile"
    , "bill_overdue_days_ntile"
  #  , "mmc_top_ntile"
  #  , "dr_top_ntile"
    #, "plan_family"
    #, "wallet_top_ntile"
    #, "tenure_top_ntile", "usg_top_ntile", "wallet_top_ntile"
    #, "plan_family", "ifp_acct_dvc_flag", "plan_discount_flag", "wallet_flag"
    #, "age_group", "network_dvc_brand", "srvc_privacy_flag"
]
#vt_param_priority_field = "wallet_control_flag"
#dict_param_priority_groups = {'Y':2}


df_base_control, df_base_target = generate_sample(
    df= df_output_struggle
    , size=vt_param_sample_req
    #, proportion = vt_param_proportion
    , strata=ls_param_strata_fields
    #, priority_field=vt_param_priority_field
    #, priority_groups=dict_param_priority_groups
    , seed=15
)

# COMMAND ----------


display(df_base_control
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             )
)

display(df_base_target
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             )
)

display(df_base_control
        .join(df_base_target, ['fs_acct_id', 'fs_cust_id'], 'inner')
        )

# COMMAND ----------

df_struggle_control = df_base_control
df_struggle_target = df_base_target

# COMMAND ----------

# DBTITLE 1,sample creation chronic
# Control group creation
vt_param_seed = 20
vt_param_sample_req = 1000
#vt_param_proportion = 0.25
ls_param_strata_fields = [
    "bill_overdays_late_6bmnth_ntile"
   , 'bill_overdue_days_ntile'
   , 'wo_propensity_top_ntile'
  #  , "mmc_top_ntile"
  #  , "dr_top_ntile"
    #, "plan_family"
    #, "wallet_top_ntile"
    #, "tenure_top_ntile", "usg_top_ntile", "wallet_top_ntile"
    #, "plan_family", "ifp_acct_dvc_flag", "plan_discount_flag", "wallet_flag"
    #, "age_group", "network_dvc_brand", "srvc_privacy_flag"
]
#vt_param_priority_field = "wallet_control_flag"
#dict_param_priority_groups = {'Y':2}


df_base_control, df_base_target = generate_sample(
    df= df_output_chronic
    , size=vt_param_sample_req
    #, proportion = vt_param_proportion
    , strata=ls_param_strata_fields
    #, priority_field=vt_param_priority_field
    #, priority_groups=dict_param_priority_groups
    , seed=15
)

# COMMAND ----------

# Get a list or set of IDs to be filtered out from df_base_control
ids_to_filter_out = df_base_control.select("fs_acct_id").distinct().rdd.flatMap(lambda x: x).collect()

# Filter df_sms_output_chronic to exclude records that are in df_base_control
df_base_target1 = df_output_chronic.filter(~df_output_chronic["fs_acct_id"].isin(ids_to_filter_out))

# COMMAND ----------

display(df_base_control
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
        )
)

display(df_base_target1
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             )
)

display(df_base_control
        .join(df_base_target1, ['fs_acct_id', 'fs_cust_id'], 'inner')
        )

# COMMAND ----------


df_chronic_control = df_base_control
df_chronic_target = df_base_target1

# COMMAND ----------

# DBTITLE 1,sample creation sporadic
# Control group creation
vt_param_seed = 20
vt_param_sample_req = 1000
#vt_param_proportion = 0.25
ls_param_strata_fields = [
    "bill_overdays_late_6bmnth_ntile"
    #,  'bill_overdue_days_ntile'
    #, 'wo_propensity_top_ntile'
  #  , "mmc_top_ntile"
  #  , "dr_top_ntile"
    #, "plan_family"
    #, "wallet_top_ntile"
    #, "tenure_top_ntile", "usg_top_ntile", "wallet_top_ntile"
    #, "plan_family", "ifp_acct_dvc_flag", "plan_discount_flag", "wallet_flag"
    #, "age_group", "network_dvc_brand", "srvc_privacy_flag"
]
#vt_param_priority_field = "wallet_control_flag"
#dict_param_priority_groups = {'Y':2}


df_base_control, df_base_target = generate_sample(
    df= df_output_sporadic
    , size=vt_param_sample_req
    #, proportion = vt_param_proportion
    , strata=ls_param_strata_fields
    #, priority_field=vt_param_priority_field
    #, priority_groups=dict_param_priority_groups
    , seed=200
)

#df_base_control.write.mode('overwrite').saveAsTable("df_base_control")
#df_base_target.write.mode('overwrite').saveAsTable("df_base_target")

# COMMAND ----------

display(df_output_sporadic
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             )
)

display(df_base_control
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             )
)

display(df_base_target
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             )
)

df_base_target1 = (df_output_sporadic
        .join(df_base_control, ['fs_acct_id', 'fs_cust_id'], 'anti')
        )

display(df_base_target1
        .join(df_base_control, ['fs_acct_id', 'fs_cust_id'], 'inner')
        )

# COMMAND ----------

evaluate_sample(
    df_base_control
    , df_base_target1
    , [ "bill_overdays_late_6bmnth_ntile", 'bill_overdue_days_ntile',  'wo_propensity_top_ntile']
)

# COMMAND ----------


df_sporadic_control = df_base_control
df_sporadic_target = df_base_target1

# COMMAND ----------

# DBTITLE 1,sample creation offender
# Control group creation
vt_param_seed = 20
vt_param_sample_req = 2000
#vt_param_proportion = 0.25
ls_param_strata_fields = [
    "bill_overdays_late_6bmnth_ntile"
    , 'bill_overdue_days_ntile'
  #  , "mmc_top_ntile"
  #  , "dr_top_ntile"
    #, "plan_family"
    #, "wallet_top_ntile"
    #, "tenure_top_ntile", "usg_top_ntile", "wallet_top_ntile"
    #, "plan_family", "ifp_acct_dvc_flag", "plan_discount_flag", "wallet_flag"
    #, "age_group", "network_dvc_brand", "srvc_privacy_flag"
]
#vt_param_priority_field = "wallet_control_flag"
#dict_param_priority_groups = {'Y':2}


df_base_control, df_base_target = generate_sample(
    df= df_output_offender
    , size=vt_param_sample_req
    #, proportion = vt_param_proportion
    , strata=ls_param_strata_fields
    #, priority_field=vt_param_priority_field
    #, priority_groups=dict_param_priority_groups
    , seed=15
)

# COMMAND ----------

display(df_output_offender
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             )
)

display(df_base_control
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             )
)

display(df_base_target
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             )
)

# df_base_target1 = (df_output_sporadic
#         .join(df_base_control, ['fs_acct_id', 'fs_cust_id'], 'anti')
#         )

display(df_base_target
        .join(df_base_control, ['fs_acct_id', 'fs_cust_id'], 'inner')
        )



# COMMAND ----------

evaluate_sample(
    df_base_control
    , df_base_target
    , [ "bill_overdays_late_6bmnth_ntile", 'bill_overdue_days_ntile',  'wo_propensity_top_ntile']
)

# COMMAND ----------

df_offender_control = df_base_control
df_offender_target = df_base_target

# COMMAND ----------

df_output_all_control = (df_chronic_control
                        .union(df_sporadic_control)
                        .union(df_struggle_control)
                        .union(df_offender_control)
                        .union(df_overload_control)
                        .withColumn('Group', f.lit('Control'))
        )

# COMMAND ----------

df_output_all_target = (df_chronic_target
                        .union(df_sporadic_target)
                        .union(df_struggle_target)
                        .union(df_offender_target)
                        .union(df_overload_target)
                        .withColumn('Group', f.lit('Target'))
        )

# COMMAND ----------

display(df_output_all_control
        .groupBy('reporting_date', 'L2_combine' , 'Group')
        .agg(f.count('*'))
        )

# COMMAND ----------

display(df_output_all_target
        .groupBy('reporting_date', 'L2_combine' , 'Group')
        .agg(f.count('*'))
        )

# COMMAND ----------

# DBTITLE 1,export
(df_output_all_control
 .write
 .mode('overwrite')
 .format('delta')
 .option('header', True)
 .save('/mnt/ml-lab/dev_users/dev_sc/dailer_list/call_audience_control_20241208')
 )

# COMMAND ----------

(df_output_all_target
 .write
 .mode('overwrite')
 .format('delta')
 .option('header', True)
 .save('/mnt/ml-lab/dev_users/dev_sc/dailer_list/call_audience_target_20241208')
 )

# COMMAND ----------

df_output_control = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/dailer_list/call_audience_control_20241208')
df_output_treatment = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/dailer_list/call_audience_target_20241208')

# COMMAND ----------

display(df_output_treatment
        .join(df_output_control, ['fs_acct_id', 'fs_cust_id'], 'inner')
        .count()
)

# COMMAND ----------

display(df_output_control
        .groupBy('reporting_date', 'L2_combine' , 'Group')
        .agg(f.count('*'))
        )

# COMMAND ----------

display(df_output_treatment
        .groupBy('reporting_date', 'L2_combine' , 'Group')
        .agg(f.count('*'))
        )

# COMMAND ----------

# DBTITLE 1,snowflake connector
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
    df_output_control
    .write
    .format("snowflake")
    .options(**options)
    .option("dbtable", "LAB_ML_STORE.SANDBOX.SC_CALL_AUDIENCE_CONTROL_20241208")
    .mode("append")
    .save()
)

# COMMAND ----------


(
    df_output_treatment
    .write
    .format("snowflake")
    .options(**options)
    .option("dbtable", "LAB_ML_STORE.SANDBOX.SC_CALL_AUDIENCE_TREATMENT_20241208")
    .mode("append")
    .save()
)
