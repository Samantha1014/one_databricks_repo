# Databricks notebook source
# MAGIC %md
# MAGIC #### s01 set up

# COMMAND ----------

import pyspark 
import os
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window
from datetime import datetime, timedelta
from pyspark.sql import DataFrame, Window
from typing import Optional, List, Dict, Tuple

# COMMAND ----------

dir_data_dl_brm = '/mnt/prod_brm/raw/cdc'

# COMMAND ----------

ls_joining_key = ['fs_acct_id', 'fs_cust_id', 'reporting_date', 'reporting_cycle_type']
vt_reporting_date = '2024-12-01'

# COMMAND ----------

password = dbutils.secrets.get(scope = "auea-kv-sbx-dxdtlprdct01", key = "sfdbrsdskey")
options = {
  "sfUrl": "vodafonenz_prod.australia-east.azure.snowflakecomputing.com", 
  "sfUser": "SVC_LAB_DS_DATABRICKS",
  "pem_private_key": password.replace('\\n', '\n'),
  "sfDatabase": "prod_pdb_masked",
  "sfSchema": "modelled",
  "sfWarehouse": "LAB_DS_WH_SCALE"
}

# COMMAND ----------

#cohort segmentaiton 
df_cohort_all = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/99_misc/cohort_seg/df_combine_all' )

# raw billing account for DOM date 
df_raw_bill = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d100_raw/d101_dp_c360/d_billing_account')

# raw brm bill_t 
df_raw_billt = spark.read.format('delta').load('/mnt/prod_brm/raw/cdc/RAW_PINPAP_BILL_T')

# feature unit base 
df_fea_unitbase = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base')
# df_fea_bill = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_bill_cycle_billing_6')

# payment and billing 
df_payment_latest = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_payment_latest')
df_fea_bill = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_bill_cycle_billing_6')

# stag payment and bill 
df_stag_bill = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d200_staging/d299_src/stg_brm_bill_t')
df_stag_payment = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d200_staging/d299_src/stg_brm_payment_hist') 

# COMMAND ----------

# DBTITLE 1,check
df_late_payer = (df_cohort_all
        .select('fs_acct_id', 'fs_cust_id', 'reporting_date', 'L2_combine', 'reporting_cycle_type')
        .distinct()
       # .filter(f.col('reporting_cycle_type') == 'calendar cycle')
        .filter(f.col('L2_combine').isin('Chronic Late', 'Sporadic Late', 'Struggling Payer', 'Overloaded', 'Intentional Offender' ))
        .filter(f.col('reporting_date') >= '2024-01-01')
        #.groupBy('reporting_date')
        #.agg(f.countDistinct('fs_acct_id'))
        )

# COMMAND ----------

df_bill_cycle = (df_fea_bill
        .filter(f.col('reporting_date') >= '2024-01-01')
        #.filter(f.col('reporting_cycle_type') == 'calendar cycle')
        .select('reporting_date', 'reporting_cycle_type','fs_acct_id', 'fs_cust_id',  'bill_cycle_start_date'
                , 'bill_cycle_end_date', 'bill_due_date', 'bill_due_amt', 'bill_no', 'bill_close_date'
                )
        .distinct()
)

# COMMAND ----------

display(df_late_payer
       .join(df_bill_cycle, ls_joining_key, 'inner')
       .groupBy('reporting_date')
       .agg(f.countDistinct('fs_acct_id')
          # , f.countDistinct('fs_srvc_id')
            , f.count('*')
            )
)


# COMMAND ----------

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
      #   .withColumn('pay_full', f.when(f.col('pct_pay_to_due')<=-1, 'full' )
      #                            .when(  (f.col('pct_pay_to_due') < 0 ) & 
      #                                    (f.col('pct_pay_to_due') >-1 ) 
      #                                    , 'partial'
      #                                  )
      #                            .otherwise('no_pay')
      #               )
       .withColumn('pay_full', f.when(f.col('pct_pay_to_due')<=-1, 'full' )
                                 .when(  (f.col('pct_pay_to_due') < 0 ) & 
                                         (f.col('pct_pay_to_due') > -1 ) 
                                         , 'partial'
                                       )
                                 .when(f.col('pay_before_due').isNull(), 'no_pay')
                                 .when(f.col('bill_due_amt')<=0, 'credit_bill')
                                 .otherwise('other')
                    )
        .groupBy('pay_full', 'reporting_date', 'L2_combine', 'pay_before_due')
        .agg(f.countDistinct('fs_acct_id')
             , f.count('*')
             , f.avg('pct_pay_to_due')
             )
        )
