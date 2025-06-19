# Databricks notebook source
# DBTITLE 1,library
import os
import pyspark
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window

# COMMAND ----------

dir_fs_data_parent = "/mnt/feature-store-prod-lab"

# COMMAND ----------

dir_fs_data_serv = os.path.join(dir_fs_data_parent, "d600_serving")
dir_fs_data_fea =  os.path.join(dir_fs_data_parent, "d400_feature")

# COMMAND ----------

df_fs_master = spark.read.format("delta").load(os.path.join(dir_fs_data_serv, "serv_mobile_oa_consumer"))
df_fs_unit_base = spark.read.format('delta').load(os.path.join(dir_fs_data_fea, 'd401_mobile_oa_consumer/fea_unit_base'))
df_fs_deact = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d500_movement/d501_mobile_oa_consumer/mvmt_service_deactivation')

# COMMAND ----------

display(df_fs_master
         .filter(f.col('reporting_cycle_type') == vt_param_reporting_cycle_type)
         .groupBy('reporting_date')
         .agg(f.count('*'))
         )

# COMMAND ----------

vt_param_reporting_cycle_type = "calendar cycle"
vt_primary_key = ['fs_cust_id', 'fs_acct_id', 'fs_srvc_id']

# COMMAND ----------

display(df_fs_master
        .filter(f.col('reporting_cycle_type') == vt_param_reporting_cycle_type)
        .filter(f.col('fs_acct_id') == '478958512')
        .filter(f.col('fs_srvc_id') == '64225145692')       
)
# ifp_prm_dvc_model vs network device 
# ifp_prm_dvc_discount_flag
# ifp_prm_dvc_discount
# ifp_prm_dvc_pmt_upfront
# ifp_prm_dvc_pmt_monthly
# ifp_prm_dvc_discount_monthly
# ifp_prm_dvc_pmt_net_monthly
# ifp_prm_dvc_rebate_flag
# ifp_prm_dvc_rebate
# ifp_prm_dvc_trade_in_flag
# ifp_prm_dvc_trade_in

# ifp_prm_dvc_pmt_monthly
# ifp_prm_dvc_pmt_net_monthly 

# COMMAND ----------

df_deact  = (df_fs_deact
        .select('fs_cust_id'
                , 'fs_acct_id'
                ,  'fs_srvc_id'
                , 'movement_date'
                , 'deactivate_type'
                )
        .filter(f.col('deactivate_reason_std')!= 'transfer')
        .distinct()
        .withColumn('index', f.row_number().over(Window.partitionBy(vt_primary_key).orderBy(f.desc('movement_date'))))
        .filter(f.col('index') == f.lit(1))
        .drop('index')
         ) 

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
                  'fs_cust_id'
                , 'fs_acct_id'
                , 'fs_srvc_id' 
                )
        .distinct()
)


# plan only base 
df_plan_only = (
    df_fs_master
    .filter(f.col('reporting_cycle_type') == vt_param_reporting_cycle_type) 
    .filter(f.col('srvc_start_date') >= '2019-01-01')
    .select('fs_cust_id'
            , 'fs_acct_id'
            , 'fs_srvc_id'
            , 'srvc_start_date'
            )
    .withColumn('rnk', f.row_number().over(Window.partitionBy(vt_primary_key).orderBy('srvc_start_date'))) # multi service date, select the earliest one 
    .filter(f.col('rnk') == 1)
    .distinct()
    .join(df_prm_ifp_base, vt_primary_key
        , 'anti'
          ) # exclude ifp base 
    .drop('rnk')
)


# COMMAND ----------

# MAGIC %md
# MAGIC Plan Only

# COMMAND ----------

df_plan_only_churn = (df_fs_master
        .filter(f.col('reporting_cycle_type') ==  vt_param_reporting_cycle_type)
        .select('reporting_date'
                , 'fs_cust_id'
                , 'fs_acct_id'
                , 'fs_srvc_id'
                )
        .distinct()
        .join(df_plan_only, vt_primary_key, 'inner') # plan only 
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
        .groupBy('active_months', 'deactivate_type')
        .agg(f.count('*').alias('total_cnt')
             , f.sum('churn_event').alias('sum_churn_event')
             , f.sum('active_status').alias('sum_active_event')
        )
       .withColumn('churn_pct', f.col('sum_churn_event')/ f.col ('total_cnt')*100)
        )

# COMMAND ----------

# MAGIC %md
# MAGIC IFP 

# COMMAND ----------

display(df_ifp_base
        .select('ifp_prm_dvc_model')
        #.filter(f.col('ifp_prm_dvc_model') == '24/04/2024  4:27:17 pm')
        .distinct()
        .orderBy('ifp_prm_dvc_model')
       )
        


# COMMAND ----------



# COMMAND ----------

df_ifp_base = (
        df_fs_master
       .filter(f.col('reporting_cycle_type') == vt_param_reporting_cycle_type)
       .filter( (f.col('ifp_prm_dvc_term_start_date') >= '2019-01-01') | 
               (f.col('ifp_prm_dvc_term_start_date').isNull())
              )
       .join(df_prm_ifp_base, vt_primary_key, 'inner')
       .select( 'fs_acct_id'
               , 'fs_cust_id'
               , 'fs_srvc_id'
               , 'reporting_date'
               , 'fs_ifp_prm_dvc_id'
                , 'ifp_prm_dvc_term_start_date'
                , 'ifp_prm_dvc_term_end_date'
                , 'ifp_prm_dvc_term_remaining'
                ,  'ifp_prm_dvc_term'
                , 'ifp_prm_dvc_flag'
                , 'ifp_prm_dvc_discount_flag' 
                , 'ifp_prm_dvc_discount'
                , 'ifp_prm_dvc_pmt_monthly'
                , 'ifp_prm_dvc_pmt_net_monthly'
                , 'ifp_prm_dvc_discount_monthly'
          # add more attributes to get $1 or $0 dollar deal 
                , 'ifp_prm_dvc_model'
                , 'ifp_prm_dvc_sales_channel_group'
                , 'plan_name_std'
                , 'plan_share_name'
               )
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
    #     .withColumn('one_dollar_2021',
    #     f.when(
    #     (f.col('ifp_prm_dvc_term_start_date') >= '2021-10-21') &
    #     (f.col('ifp_prm_dvc_term_start_date') <= '2021-12-26') &
    #     (
    #         f.lower(f.col('ifp_prm_dvc_model')).rlike("(?=.*a32)(?=.*5g)(?=.*128g)") |
    #         f.lower(f.col('ifp_prm_dvc_model')).rlike("(?=.*find)(?=.*x3)(?=.*lite)(?=.*5g)(?=.*128g)") |
    #         f.lower(f.col('ifp_prm_dvc_model')).rlike("(?=.*s21)(?=.*5g)(?=.*256g)(?=.*128g)")
    #     ),
    #     1
    # ).otherwise(0)
    #     )
    #     .withColumn('one_dollar_2022',
    #     f.when(
    #     (f.col('ifp_prm_dvc_term_start_date') >= '2022-10-20') &
    #     (f.col('ifp_prm_dvc_term_start_date') <= '2022-12-29') &
    #     (
    #         f.lower(f.col('ifp_prm_dvc_model')).rlike("(?=.*a23)") |
    #         f.lower(f.col('ifp_prm_dvc_model')).rlike("(?=.*iphone)(?=.*11)(?=.*64g)") |
    #         f.lower(f.col('ifp_prm_dvc_model')).rlike("(?=.*find)(?=.*x5)(?=.*5g)")
    #     ),
    #     1
    # ).otherwise(0)
    #     )
    #     .withColumn('zero_dollar_2023',
    #     f.when(
    #     (f.col('ifp_prm_dvc_term_start_date') >= '2023-10-19') &
    #     (f.col('ifp_prm_dvc_term_start_date') <= '2023-12-28') &
    #     (
    #         f.lower(f.col('ifp_prm_dvc_model')).rlike("(?=.*reno)(?=.*10)(?=.*5g)") |
    #         f.lower(f.col('ifp_prm_dvc_model')).rlike("(?=.*a54)") |
    #         f.lower(f.col('ifp_prm_dvc_model')).rlike("(?=.*iphone)(?=.*se)(?=.*64g)")
    #     ),
    #     1
    # ).otherwise(0)
    #     )
# .withColumn(
#     'dollar_dvc_flag',
#     f.when(
#         (f.col('one_dollar_2021') == 1) |
#         (f.col('one_dollar_2022') == 1) |
#         (f.col('zero_dollar_2023') == 1),
#         1
#         ).otherwise(0)
#         )
    .withColumn('dollar_dvc_flag_pmt', 
                f.when(
                    (
                    ( (f.col('ifp_prm_dvc_term_start_date')>= '2023-10-19') & 
                  (f.col('ifp_prm_dvc_term_start_date')<= '2023-12-28') ) | 
                 ((f.col('ifp_prm_dvc_term_start_date')>= '2021-10-21') & 
                  (f.col('ifp_prm_dvc_term_start_date')<= '2021-12-26')) |
                 ((f.col('ifp_prm_dvc_term_start_date') >= '2023-10-19') &
                    (f.col('ifp_prm_dvc_term_start_date') <= '2023-12-28')) 
                    ) & (f.col('ifp_prm_dvc_pmt_net_monthly')<=5) & 
                    (f.col('ifp_prm_dvc_term') == 36)
                    , 1 
                ).otherwise(0)
                )
        .drop('month_start', 'month_end', 'deact_month_end')
    )

# COMMAND ----------

display(df_ifp_base
        .filter(f.col('dollar_dvc_flag_pmt') == 1 )
        .filter(f.col('churn_event') == 1 )
        )

# COMMAND ----------

display(df_ifp_base
        .filter(f.col('fs_acct_id') == '480696427')
        .filter(f.col('fs_srvc_id') == '642040701945')
        )

# COMMAND ----------

display(
     df_ifp_base
        .filter(f.col('ifp_prm_dvc_term') == 36)
        .filter(f.col('ifp_prm_dvc_pmt_net_monthly').isin(0,1))
        .filter( 
                ( (f.col('ifp_prm_dvc_term_start_date')>= '2023-10-19') & 
                  (f.col('ifp_prm_dvc_term_start_date')<= '2023-12-28') ) | 
                 ((f.col('ifp_prm_dvc_term_start_date')>= '2021-10-21') & 
                  (f.col('ifp_prm_dvc_term_start_date')<= '2021-12-26')) |
                 ((f.col('ifp_prm_dvc_term_start_date') >= '2023-10-19') &
                    (f.col('ifp_prm_dvc_term_start_date') <= '2023-12-28')) 
               )
        .select('fs_cust_id'
                         , 'fs_srvc_id'
                         , 'fs_acct_id'
                         , 'fs_ifp_prm_dvc_id' 
                         , 'dollar_dvc_flag'
                         , 'ifp_prm_dvc_discount_monthly'
                         , 'ifp_prm_dvc_model'
                         , 'ifp_prm_dvc_term'
                         , 'ifp_prm_dvc_pmt_monthly'
                         , 'ifp_prm_dvc_pmt_net_monthly'
                           )
        .distinct()
        .groupBy('ifp_prm_dvc_model')
        .agg(f.count('fs_ifp_prm_dvc_id')
             , f.sum('dollar_dvc_flag')
             )
        )

# COMMAND ----------

display(df_ifp_base
        .filter(f.col('dollar_dvc_flag_pmt') ==1)
        .select('fs_ifp_prm_dvc_id')
        .distinct()
        .agg(f.count('*'))
)
        # 7 - 2021 ?? 
        # 794 - 2022 
        # 520 -- 2023
        # 5474 vs. 5665

# COMMAND ----------

display(df_ifp_base
        .filter(f.col('one_dollar_2022') == 1)
        .limit(10)
        # .filter(f.col('fs_srvc_id') == '64221039293')
        # .orderBy('reporting_date')
        )

# COMMAND ----------

df_ifp_base_full = (df_ifp_base
        .withColumn('earliest_start_date'
                    , f.min( 
                            f.when(f.col('ifp_prm_dvc_term_start_date').isNotNull()
                                   , f.col('ifp_prm_dvc_term_start_date')
                                   )
                    ).over(Window.partitionBy(vt_primary_key))
                    )
        .filter(f.col('reporting_date') >= f.last_day( f.col('earliest_start_date'))) # remove the rows before ifp start but if ifp start before 2021-7, 
        .withColumn('term_start', 
                f.when(f.col('ifp_prm_dvc_term_start_date').isNotNull(), 
                       f.col('ifp_prm_dvc_term_start_date'))
                    )
        .withColumn('term_length',
                f.when(f.col('ifp_prm_dvc_term').isNotNull() & (f.col('ifp_prm_dvc_flag') == 'Y'),
                       f.col('ifp_prm_dvc_term'))
        ) 
        .withColumn('last_term_start',
                f.last(f.col('term_start'), ignorenulls=True)
                .over(Window.partitionBy(vt_primary_key).orderBy('reporting_date'))
        ) 
        .withColumn('last_term_length',
                f.last(f.col('term_length'), ignorenulls=True)
                .over(Window.partitionBy(vt_primary_key).orderBy('reporting_date'))
        )  
        .withColumn('new_term_flag',
            (f.col('term_start').isNotNull() & 
                ((f.col('last_term_start') != f.lag('last_term_start', 1, None)
                  .over(Window.partitionBy(vt_primary_key).orderBy('reporting_date'))
                  ) 
                 | (f.row_number().over(Window.partitionBy(vt_primary_key).orderBy('reporting_date')) == 1)) # only one term situation 
            )
                .cast('integer')
        )
        .withColumn('term_group',
                    f.sum('new_term_flag')
                    .over(Window.partitionBy(vt_primary_key).orderBy('reporting_date'))
        ) 
        .withColumn('active_months',
                    f.when(f.col('last_term_start').isNotNull(),
                         f.round(f.months_between('reporting_date', 'last_term_start') ,0)
                       # f.row_number()
                        #.over(Window.partitionBy('fs_cust_id', 'fs_acct_id', 'fs_srvc_id', 'term_group').orderBy('reporting_date'))
                        )
        )
        .withColumn('propagated_ifp_prm_dvc_term',
                    f.when(f.col('ifp_prm_dvc_flag') == 'Y' , f.col('ifp_prm_dvc_term'))
                    .otherwise(f.col('last_term_length'))
        )
        #.filter(f.col('fs_acct_id') == '473483425')
        #.filter(f.col('fs_srvc_id') == '642109060284')
        .orderBy('reporting_date')
    )



# COMMAND ----------

# DBTITLE 1,propagate for dollar device to treat early cancel then churn event
df_ifp_base_full_add = (df_ifp_base_full
        .withColumn('last_dollar_dvc_flag_pmt',
                f.last(f.col('dollar_dvc_flag_pmt'), ignorenulls=True)
                .over(Window.partitionBy(vt_primary_key).orderBy('reporting_date'))
        )
        .withColumn('propagated_dollar_dvc_flag_pmt',
                    f.when(f.col('ifp_prm_dvc_flag') == 'Y' , f.col('dollar_dvc_flag_pmt'))
                    .otherwise(f.col('last_dollar_dvc_flag_pmt'))
        )
        #.filter(f.col('fs_acct_id') == '480696427')
        #.filter(f.col('fs_srvc_id') =='642040701945')
        )

# COMMAND ----------

display(df_ifp_base_full
        .filter(f.col('fs_acct_id') == '480696427' )
        .filter(f.col('fs_srvc_id') == '642040701945')
        )

# COMMAND ----------

# DBTITLE 1,36 months & dollar deals
display(df_ifp_base_full_add
        .filter(f.col('propagated_ifp_prm_dvc_term') == 36)
        # .filter(f.col('churn_event') ==1)
        .groupBy('active_months', 'propagated_ifp_prm_dvc_term', 'deactivate_type', 'propagated_dollar_dvc_flag_pmt')
        .agg(f.sum('active_status')
             , f.sum('churn_event')
             , f.countDistinct(f.concat('fs_acct_id', 'fs_cust_id', 'fs_srvc_id'))
             #, f.sum('propagated_dollar_dvc_flag_pmt')
             , f.count('*')
             )
        )

# COMMAND ----------

display(df_ifp_base_full
        .groupBy('active_months', 'propagated_ifp_prm_dvc_term', 'deactivate_type')
        .agg(f.sum('active_status')
             , f.sum('churn_event')
             , f.countDistinct(f.concat('fs_acct_id', 'fs_cust_id', 'fs_srvc_id'))
             , f.count('*')
             )
        )

# COMMAND ----------

display(df_ifp_base_full_add
        #.filter(f.col('churn_event') ==1)
        .filter(f.col('propagated_dollar_dvc_flag_pmt') ==1)
        .agg(f.countDistinct(f.concat('fs_acct_id', 'fs_srvc_id', 'fs_srvc_id')))
        )

# COMMAND ----------

display(df_fs_master
        .filter(f.col('fs_acct_id') == '471695074')
        .filter(f.col('fs_srvc_id') == '64211747610')
        .select( 'reporting_date'
                , 'fs_cust_id'
                , 'fs_srvc_id'
                , 'fs_acct_id'
                , 'fs_ifp_prm_dvc_id'
                , 'ifp_prm_dvc_term_start_date'
                , 'ifp_prm_dvc_term_end_date'
                , 'ifp_prm_dvc_term_remaining'
                , 'ifp_prm_dvc_term'
                , 'ifp_prm_dvc_flag'
                )
        .orderBy('reporting_date')
        )

# COMMAND ----------

# MAGIC %md
# MAGIC 12 months

# COMMAND ----------

display(df_ifp_base_full
        .filter(f.col('propagated_ifp_prm_dvc_term') == 12)
        .groupBy('active_months', 'propagated_ifp_prm_dvc_term')
        .agg(f.sum('active_status')
             , f.sum('churn_event').alias('churn_cnt')
             , f.countDistinct(f.concat('fs_acct_id', 'fs_cust_id', 'fs_srvc_id'))
             , f.count('*').alias('total_cnt')
             )
        .withColumn('churn_pct', f.col('churn_cnt')/ f.col('total_cnt') *100)
        .orderBy('active_months')
        )

# COMMAND ----------

# MAGIC %md
# MAGIC 24 months

# COMMAND ----------

display(df_ifp_base_full
        .filter(f.col('propagated_ifp_prm_dvc_term') == 24)
        .groupBy('active_months', 'propagated_ifp_prm_dvc_term')
        .agg(f.sum('active_status')
             , f.sum('churn_event').alias('churn_cnt')
             , f.countDistinct(f.concat('fs_acct_id', 'fs_cust_id', 'fs_srvc_id'))
             , f.count('*').alias('total_cnt')
             )
        .withColumn('churn_pct', f.col('churn_cnt')/ f.col('total_cnt') *100)
        .orderBy('active_months')
        )

# COMMAND ----------

# MAGIC %md
# MAGIC 36 months 

# COMMAND ----------

display(df_ifp_base_full
        .filter(f.col('propagated_ifp_prm_dvc_term') == 36)
        .groupBy('active_months', 'propagated_ifp_prm_dvc_term')
        .agg(f.sum('active_status')
             , f.sum('churn_event').alias('churn_cnt')
             , f.countDistinct(f.concat('fs_acct_id', 'fs_cust_id', 'fs_srvc_id'))
             , f.count('*').alias('total_cnt')
             )
        .withColumn('churn_pct', f.col('churn_cnt')/ f.col('total_cnt') *100)
        .orderBy('active_months')
        )
