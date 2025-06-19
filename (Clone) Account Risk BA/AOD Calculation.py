# Databricks notebook source
# MAGIC %md
# MAGIC ###Discovery Fact
# MAGIC 1. finance report use credit bureau t which sourced from vf_aging bucket t, bureau t has some adjustment vs. vf aging bucket t 
# MAGIC 2. finance report have more records than aging t (difference is that it also include non aging cust )
# MAGIC 3. vf_aging t is type 1 table in pinpap update weekly and monthly basis and only have the lastest current snapshot 
# MAGIC 4. hist data sitting at brm extract folder X:\BRM\Collection\reports  --- for ATB colleciton 
# MAGIC and 
# MAGIC 5. X:\BRM\Collection\reports\Finance for finance report 

# COMMAND ----------

# MAGIC %md
# MAGIC ###S001 - Set up

# COMMAND ----------

# DBTITLE 1,Library
import pyspark
import re
from pyspark import sql 
from pyspark.sql import Window
from pyspark.sql import functions as f
from pyspark.sql import SparkSession 
from pyspark.sql.types import TimestampType
from pyspark.sql.functions import from_unixtime,from_utc_timestamp
from pyspark.sql.functions import row_number 
from pyspark.sql.functions import date_format
from pyspark.sql.functions import lag, col,lit
from pyspark.sql.functions import datediff
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import to_date
from pyspark.sql.functions import abs

# COMMAND ----------

# DBTITLE 1,directory
dir_oa_consumer = 'dbfs:/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base/reporting_cycle_type=calendar cycle'
dir_atb_report = 'dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/Coll_kpi_03032024221357.csv'
dir_finan_report =  'dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/FinanceReport_01032024053404.csv'
dir_finan_report_2023 = 'dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/FinanceReport_01052023102202.csv'

dir_payment_base = 'dbfs:/mnt/ml-lab/dev_users/dev_sc/PAYMENT_BASE'
dir_bill_base = 'dbfs:/mnt/ml-lab/dev_users/dev_sc/BILL_BASE'

df_payment_base = spark.read.format('delta').load(dir_payment_base)
df_bill_base = spark.read.format('delta').load(dir_bill_base)
df_oa_consumer = spark.read.format('delta').load(dir_oa_consumer)
df_atb_rpt = spark.read.csv(dir_atb_report,header = True)
df_finan_report = spark.read.csv(dir_finan_report, header = True)
df_finan_report_2023 = spark.read.csv(dir_finan_report_2023, header = True)


# COMMAND ----------

display(
    df_payment_base.filter(f.col('account_no') =='343173107')
)

# COMMAND ----------

# DBTITLE 1,check dup in bill_base
display(df_bill_base
.select('account_no', 'bill_no')
.withColumn('cnt', f.count('*').over(Window.partitionBy('bill_no','account_no')))
.filter(f.col('cnt')>=2)
.count()
) # 34K duplicate 

# COMMAND ----------

# DBTITLE 1,dedupe in bill_base
df_bill_base = (df_bill_base
        .withColumn('latest', f.row_number().over(Window.partitionBy('account_no','bill_no').orderBy(f.desc('mod_t'))))
        .filter(f.col('latest')==1)     
        )

display(df_bill_base.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### S002 Derive AOD

# COMMAND ----------

vt_cycle_start_date = '2024-03-03'

# COMMAND ----------

# DBTITLE 0,copy
# df_bill_transform = (df_bill_base
#      .filter(f.col('account_no').isin('470956158'))
#     .filter(f.col('INVOICE_OBJ_ID0') != 0 ) # get rid of pending bills 
#     .filter(f.col('bill_no').isNotNull())   # get rid of pending bills 
#     .filter(f.col('due_t')<=vt_cycle_start_date)
#     .filter((f.col('CLOSED_T') >= vt_cycle_start_date) | (f.col('CLOSED_T') == '1970-01-01T12:00:00.000+00:00'))   #filter out not closed bill as current or bill closed after snapshot 
#     .withColumn("previous_total_at_current", lag("previous_total", -1,0).over(
#         Window.partitionBy("account_obj_id0").orderBy("due_t")))
#     .withColumn('rnk_desc', f.row_number().over(Window.partitionBy('account_obj_id0').orderBy(f.desc('End_T'))))
#     .withColumn('rnk_asc',f.row_number().over(Window.partitionBy('account_obj_id0').orderBy('end_t')))  
#     .withColumn('true_receive', 
#                 f.when(f.col('rnk_desc') ==1,f.col('previous_total_at_current'))
#                 .otherwise(f.col('previous_total_at_current')  - f.col('total_due')))    # fix for last row situation 
#     .withColumn('current_bill_due_amount', f.col('total_due') - f.col('previous_total'))
#     .withColumn('age_of_debt_bucket', 
#                 f.when(datediff(lit(vt_cycle_start_date), f.col('due_t'))<30,'Aod_01To30' )
#                 .when(datediff(lit(vt_cycle_start_date), f.col('due_t')) <60 , 'Aod_31To60')
#                 .when(datediff(lit(vt_cycle_start_date), f.col('due_t')) <90 , 'Aod_61To90')
#                 .when(datediff(lit(vt_cycle_start_date), f.col('due_t'))< 120, 'Aod_91To120')
#                 .when(datediff(lit(vt_cycle_start_date), f.col('due_t'))< 150, 'Aod_121To150')
#                 .when(datediff(lit(vt_cycle_start_date), f.col('due_t')) <180, 'Aod_151To180')
#                 .otherwise('Aod_181Plus')
#                 )
#     .withColumn('AOD_Amount', f.when(f.col('rnk_desc') ==1, f.col('total_due') - f.col('previous_total'))
#                 .when(f.col('rnk_asc') ==1, f.col('total_due') + f.sum('true_receive').over(Window.partitionBy(f.col('account_obj_id0'))))
#                 .when( ((f.col('rnk_asc')!=1) & (f.col('rnk_desc')!=1)),f.col('current_bill_due_amount') )
#                 .otherwise(0)
#                 )
#     .withColumn('reporting_date', lit(vt_cycle_start_date))
#     .withColumn('last_bill_close_date', f.row_number().over() )
# )

# display(df_bill_transform)

# .filter(f.col("ACCOUNT_OBJ_ID0") == "1694152720824")
# current utc, need to convet to nzt 

# COMMAND ----------

# DBTITLE 1,Bill Base and Move Previous Total Up
df_bill_test = (df_bill_base
    # .filter(f.col('account_no').isin('470956158'))
    .filter(f.col('INVOICE_OBJ_ID0') != 0 ) # get rid of pending bills 
    .filter(f.col('bill_no').isNotNull())   # get rid of pending bills 
    .withColumn("previous_total_at_current", lag("previous_total", -1,0).over(
        Window.partitionBy("account_obj_id0").orderBy("due_t")))
    .withColumn('rnk_desc', f.row_number().over(Window.partitionBy('account_obj_id0').orderBy(f.desc('End_T')))) # rank of total invoices 
    .withColumn('true_receive', 
                f.when(f.col('rnk_desc') ==1,f.col('previous_total_at_current'))
                .otherwise(f.col('previous_total_at_current')  - f.col('total_due')))    # fix for last row situation 
    .withColumn('current_bill_due_amount', f.col('total_due') - abs(f.col('previous_total')))
)

# .filter(f.col("ACCOUNT_OBJ_ID0") == "1694152720824")
# current utc, need to convet to nzt 


# COMMAND ----------

display(df_bill_test.filter(f.col('account_no') == '349075076'))


# COMMAND ----------

# DBTITLE 1,Overdue Bill Transformation
df_bill_transform = (
    df_bill_test
    .filter(f.col('due_t')<=vt_cycle_start_date) #  only overdue 
    .filter((f.col('CLOSED_T') >= vt_cycle_start_date) | (f.col('CLOSED_T') == '1970-01-01T12:00:00.000+00:00')) # not close or close after snapshot 
    .withColumn('age_of_debt_bucket', 
                f.when(datediff(lit(vt_cycle_start_date), f.col('due_t'))<30,'Aod_01To30' )
                .when(datediff(lit(vt_cycle_start_date), f.col('due_t')) <60 , 'Aod_31To60')
                .when(datediff(lit(vt_cycle_start_date), f.col('due_t')) <90 , 'Aod_61To90')
                .when(datediff(lit(vt_cycle_start_date), f.col('due_t'))< 120, 'Aod_91To120')
                .when(datediff(lit(vt_cycle_start_date), f.col('due_t'))< 150, 'Aod_121To150')
                .when(datediff(lit(vt_cycle_start_date), f.col('due_t')) <180, 'Aod_151To180')
                .otherwise('Aod_181Plus')
                )
    .withColumn('rnk_od_desc', f.row_number().over(Window.partitionBy('account_no').orderBy(f.desc('end_t')))) # rank of overdue invoices 
    .withColumn('AOD_Amount',
                 f.when(f.col('closed_t') =='1970-01-01T12:00:00.000+00:00', f.col('due'))
                .when((f.col('rnk_od_desc') ==1) & (f.col('rnk_desc')>1) , f.col('previous_total_at_current'))
                .when(f.col('closed_t')!= '1970-01-01T12:00:00.000+00:00',f.col('current_bill_due_amount'))
                .otherwise(0))
    .withColumn('reporting_date', lit(vt_cycle_start_date)
    )
)            
    #             f.when(f.col('rnk_od_desc') ==1, f.col('total_due') - f.col('previous_total'))
    #            .when(f.col('rnk_od_asc') ==1, f.col('total_due') + f.sum('true_receive').over(Window.partitionBy(f.col('account_obj_id0'))))
    #            .when( ((f.col('rnk_od_asc')!=1) & (f.col('rnk_desc')!=1)),f.col('current_bill_due_amount') )
    #            .otherwise(0)
    #            )
   # .withColumn('reporting_date', lit(vt_cycle_start_date))
#)

# COMMAND ----------

display(df_bill_transform.filter(f.col('account_no') =='349075076'))


# COMMAND ----------

# DBTITLE 1,Group By and Pivot
# pivot and aggregate from bill transform 

df_aod = (df_bill_transform.groupBy('account_no','reporting_date')
        .pivot('age_of_debt_bucket')
        .agg(f.sum('AOD_Amount'))
        )



# COMMAND ----------

display(df_aod.filter(f.col('account_no') =='349075076')
        .fillna(0)
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### S003 Compare with ATB Report

# COMMAND ----------

display(df_atb_rpt.limit(100))

# COMMAND ----------

display(df_aod.count()) 
# 631,717
# 2,214,970 at 01-05-2023
# 2,566,963 at 03-03-2024 
# 2,522,233 at 01-03-2024 
display(df_aod.limit(100))

# COMMAND ----------


ls_compare_cols = [c for c in df_aod.columns if 'Aod' in c and 'Current' not in c]
print(ls_compare_cols)

ls_compare_atb = [c for c in df_atb_rpt.columns if 'Aod' in c and 'Current' not in c]
#ls_compare_atb

# COMMAND ----------

# DBTITLE 1,Inner Join in Account
vt_param_join_key = 'Account Ref No'
df_join = (
    df_aod
    .fillna(0)
    .withColumnRenamed('account_no','Account Ref No')
    .join(
        df_atb_rpt
        , vt_param_join_key
        , 'inner'
    )
)

display(df_join.limit(3))



# COMMAND ----------

dictionary_compaire = {'Aod_01To30': 'Aod 01To30',
 'Aod_121To150': 'Aod 121To150',
 'Aod_151To180': 'Aod 151To180', 
 'Aod_181Plus': 'Aod 181Plus', 
 'Aod_31To60':  'Aod 31To60',
 'Aod_61To90': 'Aod 61To90' ,
 'Aod_91To120': 'Aod 91To120' }

# COMMAND ----------

# MAGIC %run "./Function"

# COMMAND ----------

# DBTITLE 1,Run Smilarity Function
df_result = check_similarity(dataframe=df_join, dict_pairs=dictionary_compaire, value_adj=1, threshold=0.99, excl_zero = True)

# COMMAND ----------

# DBTITLE 1,Compare Result
#display(df_result.limit(10))

display(
    df_result
    .groupBy('fs_col','benchmark_col')
    .agg(
        f.count('*').alias('count')
        , f.sum('similar_flag').alias('align')
    )
    .withColumn(
        'rate'
        , f.col('align')/f.col('count')
    )
)




# COMMAND ----------

# DBTITLE 1,Get Difference Record
df_diff = (df_join.
        withColumn('diff', f.col('Aod_01To30').cast('double') - f.col('Aod 01To30').cast('double') )
        .filter(f.col('diff') >=10) 
        .filter(f.col('Segment') =='CONSUMER')
        .distinct()
        )

display(df_diff.limit(100))


# COMMAND ----------

display(df_diff.count()) 
