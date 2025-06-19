# Databricks notebook source
import os
import pyspark
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window

# COMMAND ----------

dir_fs_data_parent = "/mnt/feature-store-prod-lab"

# COMMAND ----------

# dir_fs_data_meta = os.path.join(dir_fs_data_parent, 'd000_meta')
# dir_fs_data_raw =  os.path.join(dir_fs_data_parent, 'd100_raw')
# dir_fs_data_int =  os.path.join(dir_fs_data_parent, "d200_intermediate")
# dir_fs_data_prm =  os.path.join(dir_fs_data_parent, "d300_primary")
dir_fs_data_fea =  os.path.join(dir_fs_data_parent, "d400_feature")
dir_fs_data_target = os.path.join(dir_fs_data_parent, "d500_movement")
dir_fs_data_serv = os.path.join(dir_fs_data_parent, "d600_serving")

# COMMAND ----------

vt_param_reporting_date = "2024-06-30"
vt_param_reporting_cycle_type = "calendar cycle"

# COMMAND ----------

df_fs_master = spark.read.format("delta").load(os.path.join(dir_fs_data_serv, "serv_mobile_oa_consumer"))
df_fs_ifp_srvc = spark.read.format("delta").load(os.path.join(dir_fs_data_fea, "d401_mobile_oa_consumer/fea_ifp_device_on_service"))
df_fs_ifp_bill = spark.read.format("delta").load(os.path.join(dir_fs_data_fea, "d401_mobile_oa_consumer/fea_ifp_device_on_bill"))
#df_fs_ifp_device = spark.read.format('delta').load(os.path.join(dir_fs_data_fea, 'd401_mobile_oa_consumer/fea_ifp_device_account'))
df_fs_unit_base = spark.read.format('delta').load(os.path.join(dir_fs_data_fea, 'd401_mobile_oa_consumer/fea_unit_base'))
df_fs_deact = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d500_movement/d501_mobile_oa_consumer/mvmt_service_deactivation')
# /mnt/feature-store-prod-lab/d500_movement/d501_mobile_oa_consumer/mvmt_service_deactivation

# meta for fields 
#df_fs_meta = spark.read.format('delta').load(os.path.join(dir_fs_data_meta,'d004_fsr_meta','fsr_field_meta'))

# COMMAND ----------

vt_primary_key = ['fs_cust_id', 'fs_acct_id', 'fs_srvc_id']

# COMMAND ----------

df_deact  = (df_fs_deact
        .select('fs_cust_id'
                , 'fs_acct_id'
                ,  'fs_srvc_id'
                , 'movement_date'
                )
        .filter(f.col('deactivate_reason_std')!= 'transfer')
        .distinct()
        .withColumn('index', f.row_number().over(Window.partitionBy(vt_primary_key).orderBy(f.desc('movement_date'))))
        .filter(f.col('index') == f.lit(1))
        .drop('index')
         ) 

# COMMAND ----------

display(df_fs_master
        .filter(f.col('reporting_cycle_type') == vt_param_reporting_cycle_type)
        .filter(f.col('ifp_prm_dvc_flag') == f.lit('Y'))
        .filter(f.col('srvc_start_date') >= '2019-01-01')
        .select( 'fs_cust_id', 'fs_acct_id', 'fs_srvc_id')
        .distinct()
        .count()
        )

# COMMAND ----------

display(df_deact.count())

# COMMAND ----------

# cohort base 
# never ever have ifp , have ifp on bill, have ifp on service 

# ifp base 
df_prm_ifp_base = (
    df_fs_master
        .filter(f.col('reporting_cycle_type') == vt_param_reporting_cycle_type)
        .filter(f.col('srvc_start_date') >= '2019-01-01')
        .filter(f.col('ifp_prm_dvc_flag') == f.lit('Y'))
        .select(  
                # 'reporting_date'
                  'fs_cust_id'
                , 'fs_acct_id'
                , 'fs_srvc_id' 
                # , 'fs_ifp_prm_dvc_id'
               # , 'fs_ifp_prm_dvc_order_id'
               # , 'srvc_start_date'
               # , 'ifp_prm_dvc_term_start_date'
                #, 'ifp_prm_dvc_term_end_date'
                #, 'ifp_prm_dvc_term'
                )
        .distinct()
)


# plan only base 
df_plan_only = (
    df_fs_master
    .filter(f.col('reporting_cycle_type') == vt_param_reporting_cycle_type) 
    .filter(f.col('srvc_start_date') >= '2019-01-01')
    #.filter(f.col('reporting_date')<='2024-06-30')
    .select('fs_cust_id'
            , 'fs_acct_id'
            , 'fs_srvc_id'
            #, 'fs_ifp_prm_dvc_id' 
            , 'srvc_start_date'
            )
    .withColumn('rnk', f.row_number().over(Window.partitionBy(vt_primary_key).orderBy('srvc_start_date'))) # multi service date, select the earliest one 
    .filter(f.col('rnk') == 1)
    .distinct()
    .join(df_prm_ifp_base, vt_primary_key
        , 'anti'
          )
    .drop('rnk')
)


# COMMAND ----------

display(df_plan_only.count()) # 652,917

# COMMAND ----------

display(df_prm_ifp_base
        .count())

# COMMAND ----------

# DBTITLE 1,snapshot base for plan only and churn
df_plan_only_churn = (df_fs_master
        .filter(f.col('reporting_cycle_type') ==  vt_param_reporting_cycle_type)
        .select('reporting_date'
                , 'fs_cust_id'
                , 'fs_acct_id'
                , 'fs_srvc_id'
                )
        .distinct()
        .join(df_plan_only, vt_primary_key, 'inner')
        .join(df_deact
              .withColumn('deact_month_end', f.last_day(f.col('movement_date')))
              , vt_primary_key, 'left')
        .filter(( f.col('deact_month_end').isNull()) | 
                (f.col('reporting_date') <=f.col('deact_month_end'))
                ) # make sure churn_event is the last rows 
        .withColumn('month_start', 
                    f.date_trunc('month', f.col('reporting_date'))
                    )
        .withColumn('month_end', 
                    f.last_day(f.col('reporting_date'))
                    )
        .withColumn('churn_event', 
                    f.when( (f.col('movement_date') >= f.col('month_start')) & 
                           (f.col('movement_date') <= f.col('month_end')), 1
                    )
                    .otherwise(0)
        )
        .withColumn('active_status', f.when(f.col('churn_event') == 1, 0)
                                     .when(f.col('churn_event') == 0, 1)
                                     .otherwise(-1)
                    )
        .drop('month_start', 'month_end', 'deact_month_end')
        .withColumn('minimum_reporting_date', f.min('reporting_date').over(Window.partitionBy(vt_primary_key)))
        .withColumn('active_months',  
                        f.when(f.last_day(f.col('srvc_start_date')) >  f.col('minimum_reporting_date'), 
                               f.row_number().over(Window.partitionBy(vt_primary_key).orderBy('reporting_date'))
                               ) # for service start later than the earliest reporting date 
                        .when( f.col('srvc_start_date') <= f.col('minimum_reporting_date'),  
                              f.round(f.months_between('reporting_date', 'srvc_start_date'),0) )
                                )
                   )
      
        

# COMMAND ----------

display(df_plan_only_churn
        #.filter(f.col('reporting_cycle_type') == vt_param_reporting_cycle_type)
        .filter(f.col('fs_acct_id') == '478957953')
        .filter(f.col('fs_srvc_id') == '64211228026')
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plan Only Churn 

# COMMAND ----------

display(df_plan_only_churn
        .groupBy('active_months')
        .agg(f.count('*').alias('total_cnt')
             , f.sum('churn_event').alias('sum_churn_event')
             , f.sum('active_status').alias('sum_active_event')
        )
       .withColumn('churn_pct', f.col('sum_churn_event')/ f.col ('total_cnt')*100)
        )

# COMMAND ----------

display(df_plan_only_churn
        .filter(f.col('active_months') == 0)
        )

# COMMAND ----------

display(df_plan_only_churn
        .filter(f.col('fs_acct_id') =='479852733')
        .filter(f.col('fs_srvc_id') == '64273121676')
       # .filter(f.col('reporting_cycle_type') == vt_param_reporting_cycle_type)
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### IFP

# COMMAND ----------

# DBTITLE 1,snapshot base for ifp only
df_ifp_snapshot = (df_fs_master
        .filter(f.col('reporting_cycle_type') ==  vt_param_reporting_cycle_type)
        .select('reporting_date'
                , 'fs_cust_id'
                , 'fs_acct_id'
                , 'fs_srvc_id'
                , 'fs_ifp_prm_dvc_id'
                , 'ifp_prm_dvc_term_start_date'
                , 'ifp_prm_dvc_term_end_date'
                , 'ifp_prm_dvc_term'
               #  , 'ifp_prm_dvc_term_remaining'
                , 'ifp_prm_dvc_flag'
                )
        .distinct()
        .join(df_prm_ifp_base, vt_primary_key, 'inner')
        .join(df_deact
              .withColumn('deact_month_end', f.last_day(f.col('movement_date')))
              , vt_primary_key, 'left')
        .filter(( f.col('deact_month_end').isNull()) | 
                (f.col('reporting_date') <=f.col('deact_month_end'))
                ) # make sure churn_event is the last rows 
        .withColumn('month_start', 
                    f.date_trunc('month', f.col('reporting_date'))
                    )
        .withColumn('month_end', 
                    f.last_day(f.col('reporting_date'))
                    )
        .withColumn('churn_event', 
                    f.when( (f.col('movement_date') >= f.col('month_start')) & 
                           (f.col('movement_date') <= f.col('month_end')), 1
                    )
                    .otherwise(0)
        )
        .withColumn('active_status', f.when(f.col('churn_event') == 1, 0)
                                     .when(f.col('churn_event') == 0, 1)
                                     .otherwise(-1)
                    )
        .drop('month_start', 'month_end', 'deact_month_end')
)
       #  .withColunn( )
        # .withColumn('ifp_tenure', f.col('ifp_prm_dvc_term') - f.col('ifp_prm_dvc_term_remaining') )
     # calculate max reporting date for IFP = Y 
        .withColumn('temp_max_month', 
                f.max(f.when(f.col('ifp_prm_dvc_flag') == 'Y', f.col('reporting_date')))
                        .over(Window.partitionBy(vt_primary_key))
                   )
    # Calculate the term at the max month
       .withColumn('temp_term_at_max_month',
        f.max(
            f.when(
                (f.col('reporting_date') == f.col('temp_max_month')) & (f.col('ifp_prm_dvc_flag') == 'Y'), 
                f.col('ifp_prm_dvc_term')
            )
        ).over(Window.partitionBy(vt_primary_key))
    )
    # create max_month_ifp_flag_Y and max_term_ifp_flag_Y, only showing values when ifp_prm_dvc_flag is 'N' and after the ifp end 
        .withColumn('max_month_ifp_flag_Y',
        f.when( (f.col('ifp_prm_dvc_flag') == 'N') & (f.col('reporting_date') >= f.col('temp_max_month')), f.col('temp_max_month'))
        .otherwise(None)
                  )
        .withColumn('max_term_ifp_flag_Y',
                f.when( (f.col('ifp_prm_dvc_flag') == 'N') & (f.col('reporting_date') >= f.col('temp_max_month')), f.col('temp_term_at_max_month'))
                .otherwise(None)
                        )
    # Drop temporary columns
        #.drop('temp_max_month', 'temp_term_at_max_month')
)
      

# assumption - if early cancel term, and churn within 6 months post the cancellation date, count as churn beyond term 

# COMMAND ----------

display(df_ifp_snapshot
        .filter(f.col('fs_acct_id') == '478958512')
        .filter(f.col('f'))
        )

# COMMAND ----------

display(df_ifp_snapshot 
        .groupBy('temp_term_at_max_month', 'ifp_prm_dvc_flag')
        .agg(f.sum('active_status').alias('sum_active_event')
             , f.sum('churn_event').alias('sum_churn_event')
             , f.count('*').alias('total_cnt')
             )
        .withColumn('churn_pct',  f.col('sum_churn_event') / f.col('total_cnt')*100)
        )

# COMMAND ----------

display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### separate within/ beyond term 

# COMMAND ----------

display(df_ifp_snapshot
        .withColumn('churn_type', 
                    f.when(
                        (f.col('churn_event') == 1) & (f.col('ifp_prm_dvc_flag') == 'Y')
                        , 'churn_within'
                    )
                    .when(
                         (f.col('churn_event') == 1) & (f.col('ifp_prm_dvc_flag') == 'N')
                         , 'churn_beyond'
                    )
                    .otherwise('active')
                    )
         .withColumn('survival_months', 
                    f.when(
                      f.col('churn_type') == f.lit('churn_within')
                        , f.col('ifp_tenure')
                    )
                    .when(
                       f.col('churn_type') == f.lit('churn_beyond')
                         , f.months_between(f.col('reporting_date'), f.col('temp_max_month') )
                    )
                    .when(
                        f.col('churn_type') == f.lit('active'), 
                        f.col('ifp_tenure')
                    )
                    )
         .withColumn('term', 
                      f.when(f.col('churn_type') == 'churn_within', 
                             f.col('ifp_prm_dvc_term')
                             )
                      .when(f.col(''))
                     )
        #.filter(f.col('fs_acct_id') == '490382054')
        #.filter(f.col('fs_srvc_id') == '642102994986')
        #.filter(f.col('churn_type') == 'churn_beyond')
        .groupBy('churn_type', 'survival_months', 'max_term_ifp_flag_Y')
        .agg(f.count('*'))
        #.filter(f.col('fs_acct_id') == '480686425')
        #.filter(f.col('fs_srvc_id') == '6421595461')
)
#	480686425	6421595461
# 479460001	642102986937
# 1-15RQW8CL	490382054	642102994986 24 months churn beyond

# COMMAND ----------

display(df_ifp_snapshot
        # .withColumn('max_ifp_end_date', f.max('ifp_prm_dvc_term_end_date').over(Window.partitionBy(vt_primary_key))
        #             )
        .withColumn('check',
                    f.when( f.col('temp_max_month') < f.col('ifp_prm_dvc_term_end_date'), 
                           'early cancel'
                            )
                    .otherwise('other')
        )
        #.groupBy('check')
        #.agg(f.count('*'))
        #.filter(f.col('fs_acct_id') == '478959432')
        #.filter(f.col('fs_srvc_id') == '64210728726')
        .groupBy('check')
        .agg(f.countDistinct('fs_acct_id', 'fs_cust_id', 'fs_srvc_id'))
)


# COMMAND ----------


  df_ifp_churn_within = (  
    df_ifp_snapshot
    # churn within term 
    .filter(f.col('churn_event') ==1)
    .filter(f.col('ifp_prm_dvc_flag') == 'Y')
    .select('fs_cust_id'
            , 'fs_acct_id'
            , 'fs_srvc_id'
            )
    .distinct()
  )



  df_ifp_churn_beyond = (
      df_ifp_snapshot
      .filter(f.col('churn_event') == 1)
      .filter(f.col('ifp_prm_dvc_flag') == 'N')
      .filter(f.col('max_term_ifp_flag_Y').isNotNull())
      .select('fs_cust_id', 'fs_acct_id', 'fs_srvc_id')
      .distinct()
  )


df_ifp_active = (
  df_ifp_snapshot
  .select('fs_cust_id'
          , 'fs_acct_id'
          , 'fs_srvc_id'
          )
  .distinct()
  .join(df_ifp_churn_beyond, vt_primary_key, 'anti')
  .join(df_ifp_churn_within, vt_primary_key, 'anti')
)

