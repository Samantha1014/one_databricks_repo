# Databricks notebook source
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

dir_oa_consumer = 'dbfs:/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base/reporting_cycle_type=calendar cycle'
dir_atb_report = 'dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/Coll_kpi_03032024221357.csv'
dir_finan_report =  'dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/FinanceReport_01032024053404.csv'
dir_finan_report_2023 = 'dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/FinanceReport_01052023102202.csv' # 2023-05-01

dir_payment_base = 'dbfs:/mnt/ml-lab/dev_users/dev_sc/PAYMENT_BASE'
dir_bill_base = 'dbfs:/mnt/ml-lab/dev_users/dev_sc/BILL_BASE'

df_payment_base = spark.read.format('delta').load(dir_payment_base)
df_bill_base = spark.read.format('delta').load(dir_bill_base)
df_oa_consumer = spark.read.format('delta').load(dir_oa_consumer)
df_atb_rpt = spark.read.csv(dir_atb_report,header = True)
df_finan_report = spark.read.csv(dir_finan_report, header = True)
df_finan_report_2023 = spark.read.csv(dir_finan_report_2023, header = True)


# COMMAND ----------

vt_reporting_date = '2024-03-03'

# COMMAND ----------

df_bill_transform = (df_bill_base
        #.filter(f.col('account_no')=='481541422')
        .filter(col('mod_t') <= lit(vt_reporting_date))
        #.filter(col('due_t')< lit(vt_reporting_date))
        .filter((col('closed_t') == '1970-01-01T12:00:00.000+00:00') | (col('closed_t') > lit(vt_reporting_date)))
        .withColumn('od_days', datediff(lit(vt_reporting_date), col('due_t')))
        .withColumn('age_of_debt_bucket', f.when(col('od_days') <0, 'Aod_Current')
                .when(col('od_days') <30,'Aod_01To30' )
                .when(col('od_days') <60 , 'Aod_31To60')
                .when(col('od_days') <90 , 'Aod_61To90')
                .when(col('od_days') <120, 'Aod_91To120')
                .when(col('od_days') < 150, 'Aod_121To150')
                .when(col('od_days') <180, 'Aod_151To180')
                .otherwise('Aod_181Plus')
                ) 
        
)



# COMMAND ----------

df_aod = (df_bill_transform.groupBy('account_no',lit(vt_reporting_date).alias('snapshot_date'))
        .pivot('age_of_debt_bucket')
        .agg(f.sum('due'))
        .fillna(0)
        )

# COMMAND ----------

display(df_aod.count())

# COMMAND ----------

dictionary_compaire = {
  'Aod_Current' : 'Aod Current',
 'Aod_01To30': 'Aod 01To30',
 'Aod_121To150': 'Aod 121To150',
 'Aod_151To180': 'Aod 151To180', 
 'Aod_181Plus': 'Aod 181Plus', 
 'Aod_31To60':  'Aod 31To60',
 'Aod_61To90': 'Aod 61To90' ,
 'Aod_91To120': 'Aod 91To120' }

# COMMAND ----------

# MAGIC %run "./Function"

# COMMAND ----------

dir_atb_report = 'dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/Coll_kpi_03032024221357.csv'  # 2024-03-03
df_atb_rpt = spark.read.csv(dir_atb_report,header = True)
df_atb_rpt= df_atb_rpt.fillna(0)


# COMMAND ----------

df_joined = (df_aod
.join(df_atb_rpt, col('account_no') == col('Account Ref No'), 'inner')
)

# COMMAND ----------

display(df_joined.count())

# COMMAND ----------

df_result = check_similarity(dataframe=df_joined, dict_pairs=dictionary_compaire, value_adj=1, threshold=0.98
                            , excl_zero = True)

# COMMAND ----------

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
    .withColumn('not_align', f.col('count') - f.col('align'))
)


# COMMAND ----------

df_diff = (df_joined
        .select('account_no', 'snapshot_date', 'Aod_Current', 'Aod_01To30', 'Aod_31To60', 'Aod_61To90', 
                'Aod Current', 'Aod 01To30',  'Aod 31To60',    'Aod 61To90'   )
        .withColumn('diff', f.col('Aod_01To30') - f.col('Aod 01To30') )
        .filter(abs(f.col('diff')) >=1 ) 
        .distinct()
        )

display(df_diff.limit(100))

# COMMAND ----------

display(df_bill_transform.filter(col('account_no') =='369869352'))

# COMMAND ----------

dir_bill_t =  '/mnt/prod_brm/raw/cdc/RAW_PINPAP_BILL_T'
df_bill_t = spark.read.format('delta').load(dir_bill_t)

# COMMAND ----------


df_bill_t = (
    df_bill_t
    .filter(col('_is_latest') ==1)
    .filter(col('_is_deleted') ==0)
    .filter(col('end_t') >= 1609412400) 
)
