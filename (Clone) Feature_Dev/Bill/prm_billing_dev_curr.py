# Databricks notebook source
# MAGIC %md
# MAGIC ### S00001 Library

# COMMAND ----------

import pyspark
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql.functions import from_unixtime,from_utc_timestamp
from pyspark.sql.functions import date_format, last_day, datediff
from pyspark.sql.functions import regexp_replace, last
from pyspark.sql.functions import lag, lead
from pyspark.sql.functions import col, month, dayofmonth, when, add_months
from pyspark.sql import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ### S00002 Directory

# COMMAND ----------

dir_oa_prm = '/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_unit_base/reporting_cycle_type=calendar cycle'
df_oa_prm = spark.read.format('delta').load(dir_oa_prm)

# COMMAND ----------

dir_bill_int = 'dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d200_intermediate/bill_int_scc'
df_bill_int = spark.read.format('delta').load(dir_bill_int)
display(df_bill_int.count()) # 3,051,818

# COMMAND ----------

# MAGIC %md
# MAGIC ###S00003 Development 

# COMMAND ----------

# DBTITLE 1,parameter
vt_param_ssc_reporting_date = '2023-07-31'
vt_param_cycle = 6

# COMMAND ----------

# DBTITLE 1,unit base
df_oa_prm_curr = (
    df_oa_prm
    .filter(f.col('reporting_date')==vt_param_ssc_reporting_date)
    .select('reporting_date', 'fs_acct_id', 'fs_cust_id')
    .distinct()
)

# COMMAND ----------

# DBTITLE 1,inner join to unit base
df_bill_prm = (
    df_oa_prm_curr.alias('a')
    .join(df_bill_int.alias('b'), ['fs_acct_id'], 'inner')
    .select('b.*')
)


# COMMAND ----------

display(df_bill_prm.limit(100))

# COMMAND ----------

# DBTITLE 1,Add Transform Column
df_bill_prm_curr_01 = (
    df_bill_prm
    .withColumn('bill_period', 
                f.datediff(f.col('bill_end_dttm'), f.col('bill_start_dttm'))
                )
    .withColumn('bill_current_charge', 
                f.col('bill_total_due') - f.col('bill_previous_total')
                )
    .withColumn(
        'next_record_previous_total', 
                lead('bill_previous_total', 1)
                .over(Window.partitionBy('fs_acct_id')
                      .orderBy('bill_end_dttm'))
                )
    .withColumn('calculated_recvd', 
                f.col('bill_total_due') - f.col('next_record_previous_total')
                )
                )
    

# COMMAND ----------

# MAGIC %md
# MAGIC ### Latest Cycle 

# COMMAND ----------

# due date before reporting date for complete bill cycle 
df_bill_prm_curr_02 = (
    df_bill_prm_curr_01
    .filter(f.col('last_bill_cycle_rnk') ==1)
        .withColumn(
                'overdue_days', 
                f.when( # if bill not close or close after reporting date 
                     (f.col('bill_closed_dttm')=='1970-01-01T12:00:00.000+00:00') | 
                      (  (f.col('bill_closed_dttm') > f.col('reporting_date') ) ) ,
                                       datediff(f.col('reporting_date'), f.col('bill_due_dttm'))
                      )
                .otherwise( # bill close before reporting date 
                    datediff(f.col('bill_closed_dttm'), f.col('bill_due_dttm'))
                          )
                    )
        .withColumn('flag_pay_bills', 
                    f.when( (f.col('bill_closed_dttm')=='1970-01-01T12:00:00.000+00:00') | 
                      ( (f.col('bill_closed_dttm') > f.col('reporting_date') ) )
                      , 'overdue'
                    ).otherwise('paid')
                    )
         .withColumn('flag_pay_time', 
                    f.when( (f.col('overdue_days') >0) 
                           & (f.col('flag_pay_bills') == 'overdue'),
                            'miss') # if overdue bill means payment is missed 
                    .when( (f.col('overdue_days') >0) 
                           & (f.col('flag_pay_bills') == 'paid'), 
                            'late')
                    .when(f.col('overdue_days')<0, 'early' )
                    .when(f.col('overdue_days') ==0, 'ontime')
                    )
        .withColumn('flag_pay_status',  
                    f.when(f.col('flag_pay_bills') =='paid', 'full') # if pay already, means bill is closed, then it is paid in full 
                    .when(f.col('calculated_recvd') ==0, 'no_pay') # if calculated receivd is 0, then means no payment made 
                    .when(f.col('next_record_previous_total') >0, 'partial')
                    .when(f.col('next_record_previous_total') ==0, 'full' )
                    .when(f.col('next_record_previous_total') <0, 'over')
                ) 
)


# COMMAND ----------

# DBTITLE 1,Pivot on flag_pay_bills
df_bill_prm_curr_03 = (df_bill_prm_curr_02
        .groupBy('fs_acct_id', 'reporting_date' )
        .pivot('flag_pay_bills', ['overdue'] )
        .agg(
            f.count('bill_no')
             )
        .fillna(0)
        .withColumn('overdue', f.when(
                                f.col('overdue') ==1, f.lit('Y'))
                                .otherwise(f.lit('N'))
                    )
        .withColumnRenamed('overdue', 'overdue_bill_curr')
)

# COMMAND ----------

# DBTITLE 1,Pivot on flag_pay_time
 df_bill_prm_curr_04 = (df_bill_prm_curr_02        
        .groupBy('fs_acct_id', 'reporting_date' )
        .pivot('flag_pay_time' )
        .agg(
            f.count('bill_no').alias('bill_cnt')
        )
        .fillna(0)
        .withColumnRenamed('early', 'early_pay_curr')
        .withColumnRenamed('late', 'late_pay_curr')
        .withColumnRenamed('miss', 'miss_pay_curr')
        .withColumnRenamed('ontime', 'ontime_pay_curr')
        
)
 
pivoted_columns = [col for col in df_bill_prm_curr_04.columns if col not in ['fs_acct_id', 'reporting_date']]

for col_name in pivoted_columns:
    df_bill_prm_curr_04 = df_bill_prm_curr_04.withColumn(col_name, 
                                       f.when(f.col(col_name) == 0, 'N')
                                        .when(f.col(col_name) == 1, 'Y')
                                        .otherwise(f.col(col_name)))


# COMMAND ----------

# DBTITLE 1,Pivot on flag_pay_status
df_bill_prm_curr_05 = (df_bill_prm_curr_02              
        .groupBy('fs_acct_id', 'reporting_date')
        .pivot('flag_pay_status')
        .agg(
                f.count('bill_no').alias('bill_cnt')
        )
        .fillna(0)
        .withColumnRenamed('null', 'unknown_pay_status_curr')
        .withColumnRenamed('full', 'bill_payfull_curr')
        )

pivoted_columns = [col for col in df_bill_prm_curr_05.columns if col not in ['fs_acct_id', 'reporting_date']]

for col_name in pivoted_columns:
    df_bill_prm_curr_05 = df_bill_prm_curr_05.withColumn(col_name, 
                                       f.when(f.col(col_name) == 0, 'N')
                                        .when(f.col(col_name) == 1, 'Y')
                                        .otherwise(f.col(col_name)))

# COMMAND ----------

display(df_bill_prm_curr_05)

# COMMAND ----------

# DBTITLE 1,other current attributes
df_bill_prm_curr_06 = (df_bill_prm_curr_02
        .select('fs_acct_id'
                , 'reporting_date'
                , 'bill_total_due'
                ,'bill_previous_total'
                , 'bill_no'
                , 'bill_due_dttm' 
                , 'bill_closed_dttm'
                , 'overdue_days')
        .withColumn('days_from_last_bill_due', datediff('reporting_date', 'bill_due_dttm'))
        .withColumnRenamed('bill_no', 'bill_no_curr')
        .withColumnRenamed('bill_due_dttm', 'bill_due_dttm_curr')
        .withColumnRenamed('bill_closed_dttm', 'bill_closed_dttm_curr')
        .withColumnRenamed('bill_total_due', 'bill_total_due_curr')
        .withColumnRenamed('bill_previous_total', 'bill_previous_total_curr')
        )

# COMMAND ----------

# DBTITLE 1,bill jump / decrease
 #calcualte the lastest bill charge vs. the average of 6 cycles  
df_bill_prm_curr_07 = (
    df_bill_prm_curr_02
    .withColumn('bill_period', 
                f.datediff(f.col('bill_end_dttm'), f.col('bill_start_dttm'))
                )
    .withColumn('bill_current_charge', 
                f.col('bill_total_due') - f.col('bill_previous_total')
                )
    .withColumn('bill_charge_avg', f.round(
                                    f.avg('bill_current_charge')
                                    .over(Window.partitionBy('fs_acct_id')
                                          ),2
                                         )
                )
    .filter(f.col('last_bill_cycle_rnk') ==1)
    .select('fs_acct_id'
            , 'reporting_date'
            , 'bill_charge_avg'
            , 'bill_current_charge' )
    .withColumn(f'bill_change_pct_vs_{vt_param_cycle}_cycle',
                f.round( (f.col('bill_current_charge') - f.col('bill_charge_avg'))
                        / f.col('bill_charge_avg')
                        ,2)
                )
    .fillna(0, subset=[f'bill_change_pct_vs_{vt_param_cycle}_cycle']) # there is null value when 0 diviede by 0 , 3.9K volume 
    
)

# COMMAND ----------

# DBTITLE 1,Combine Togetger
 display(df_bill_prm_curr_06
         .join(df_bill_prm_curr_03,['fs_acct_id', 'reporting_date'],'inner')
        .join(df_bill_prm_curr_04, ['fs_acct_id', 'reporting_date'],'inner')
        .join(df_bill_prm_curr_05, ['fs_acct_id', 'reporting_date'],'inner')
        .join(df_bill_prm_curr_07,  ['fs_acct_id', 'reporting_date'],'inner')
 )
