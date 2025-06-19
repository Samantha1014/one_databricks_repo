# Databricks notebook source
# MAGIC %md
# MAGIC ### S01 Set up

# COMMAND ----------

import pyspark
import os
from pyspark.sql import functions as f

# COMMAND ----------

# MAGIC %md
# MAGIC ### S02 Load Data

# COMMAND ----------

dir_prod_edw = 'dbfs:/mnt/prod_edw/raw/cdc'
dir_prod_siebel = 'dbfs:/mnt/prod_siebel/raw/cdc'

# siebel 
df_sbl_s_orgext = spark.read.format('delta').load(os.path.join(dir_prod_siebel, 'RAW_SBL_ORG_EXT'))

# COMMAND ----------

# DBTITLE 1,get siebel cust_id
df_cust_id = (
    df_sbl_s_orgext
    .filter(f.col('_is_latest') == 1)
    .select(f.col('ou_num').alias('account_no')
            , f.col('par_ou_id').alias('customer_id')
            # , f.col('row_id').alias('billing_acct_id')
           # , 'accnt_type_cd'
            )
    .filter(f.col('accnt_type_cd').isin('Billing'))
            )

# COMMAND ----------

df_aod = spark.sql(
    """
        SELECT 
        Account_Ref_No 
        , CASE WHEN Aod_181Plus > 0 THEN 'X'
                WHEN Aod_151To180 > 0 THEN '6'
                WHEN Aod_121To150 > 0 THEN '5'
                WHEN Aod_91To120 > 0 THEN '4'
                WHEN Aod_61To90 > 0 THEN '3'
                WHEN Aod_31To60 > 0 THEN '2'
                WHEN Aod_01To30  > 0 THEN '1'
                ELSE '0'
        END AS PaymentStatus
        , payment_status as old_payment_status
        , aod_current + Aod_01To30 + Aod_31To60 + Aod_61To90 + Aod_91To120 + Aod_121To150 + Aod_151To180 + Aod_181Plus as sum_aod
        , hist_start_dt
        , hist_end_dt
    FROM delta.`/mnt/prod_edw/raw/cdc/RAW_EDW2PRD_STAGEPERM_S_INF_AOD`
    WHERE _is_latest = 1
        AND _is_deleted = 0
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### S03 Development

# COMMAND ----------

# DBTITLE 1,get look back months
    vt_reporting_date = f.add_months(f.current_date(),-1)

# COMMAND ----------

### https://dev.azure.com/vodafonenz/IT/_workitems/edit/309240
### ADO Ticket 

df_aod_output = (df_aod
        .filter(f.col('PaymentStatus') != '0')
        .filter(f.col('sum_aod') >=20)
        .withColumn('reporting_date', f.last_day(
                                        f.add_months(f.current_date(), -1)
                                                )
                    )
        .filter( (f.col('hist_start_dt') <= f.col('reporting_date')) &
                (f.col('hist_end_dt') >= f.col('reporting_date'))
                )
        .select(
                f.col('account_ref_no').alias('account_no')
                , f.col('PaymentStatus').alias('payment_status')
                , f.date_format('reporting_date', 'yyyyMM').alias('created_month')
        )
        )
        

# COMMAND ----------

# MAGIC %md
# MAGIC ### S04 Output

# COMMAND ----------

df_output = (df_aod_output
             .join(df_cust_id, ['account_no'], 'left')
             .withColumn('account_no', f.lpad(f.col("account_no").cast("string"), 9, '0'))
             )
             

# COMMAND ----------

# MAGIC %md
# MAGIC ### S05 Check

# COMMAND ----------

display(df_output.count())

display(df_output)

# COMMAND ----------

# DBTITLE 1,check QA
display(df_output
        .filter(f.col('customer_id').isNull())
        .select('account_no')
        .distinct()
)

