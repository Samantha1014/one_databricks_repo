# Databricks notebook source
import pyspark
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window 
from pyspark.sql.functions import lag, date_add

# COMMAND ----------

df_aod = spark.read.format('delta').load('dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d999_tmp/test_aod_movement')
df_oa_prm = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_unit_base/reporting_cycle_type=calendar cycle')

# COMMAND ----------

vt_param_ssc_reporting_date = '2024-03-31'

# COMMAND ----------

display(df_aod.count())

# COMMAND ----------

# DBTITLE 1,data quality in fs oa
display(df_aod
        .filter(f.col('create_date').isNull())
)

# total 3654 accts in fs oa but not in aod table 
# check 5 cases -  seems to be account with 
# 1. no product but still bill 
# 2. only tablet plans - data only 
# 3. bills stoped at 2022   
# 4. bill not found - probably new starter and not bill yet 

# COMMAND ----------

# DBTITLE 1,parameter
winspec = Window.partitionBy('fs_acct_id').orderBy('create_date')

ls_col = ['aod_current', 'aod_30', 'aod_60', 'aod_90', 'aod_120', 'aod_150', 'aod_180']

ls_param_select_fields  = [ 
                           'fs_acct_id', 'reporting_date', 'create_date',
                           'aod_current','aod_30', 'aod_60', 'aod_90', 'aod_120', 'aod_150', 'aod_180']

df_aod = (df_aod.select(*ls_param_select_fields))

# get previous aod figures
for col in ls_col: 
    df_aod = df_aod.withColumn((f'{col}_prev'), lag(f.col(col)).over(winspec))


# COMMAND ----------

# DBTITLE 1,check trend
# check trend 
display(df_aod
        .groupBy('create_date')
        .agg(f.count('fs_acct_id')
             , f.countDistinct('fs_acct_id')
             )
        )

# COMMAND ----------

# DBTITLE 1,60+ example
display(df_aod
        .filter(f.col('fs_acct_id')==501074774)
        )


# COMMAND ----------

# DBTITLE 1,movement base
df_aod_movement = (
    df_aod
    .withColumn(
        'movement_flag', 
        (f.col('aod_current') != f.col('aod_current_prev'))
        | (f.col('aod_30') != f.col('aod_30_prev') )
        | (f.col('aod_60') != f.col('aod_60_prev'))
        | (f.col('aod_90') != f.col('aod_90_prev'))
        | (f.col('aod_120') != f.col('aod_120_prev') )
        | (f.col('aod_150') != f.col('aod_150_prev'))
        | (f.col('aod_180') != f.col('aod_180_prev'))
    )
    .filter(f.col('movement_flag'))
    .drop('movement_flag')
   #  .filter(f.col('fs_acct_id') == 1002892)
        )

# COMMAND ----------

display (
    df_aod
    .withColumn(
        'movement_flag', 
        (f.col('aod_current') != f.col('aod_current_prev'))
        | (f.col('aod_30') != f.col('aod_30_prev') )
        | (f.col('aod_60') != f.col('aod_60_prev'))
        | (f.col('aod_90') != f.col('aod_90_prev'))
        | (f.col('aod_120') != f.col('aod_120_prev') )
        | (f.col('aod_150') != f.col('aod_150_prev'))
        | (f.col('aod_180') != f.col('aod_180_prev'))
    )
    # .filter(f.col('movement_flag'))
    .filter(f.col('fs_acct_id') == 1064310)
        )

# COMMAND ----------

# DBTITLE 1,aod 60 example movement
display(df_aod_movement.
        filter(f.col('fs_acct_id') == 501074774)
        )

# COMMAND ----------

# DBTITLE 1,curr to 30 example
display(df_aod_movement
        .filter(
            f.col('fs_acct_id') == 1002892
                )
)

# COMMAND ----------

df_aod_in = (
        df_aod_movement
        .withColumn('curr_to_30', 
                            (f.col('aod_current') < (f.col('aod_current_prev')))
                           & (f.col('aod_30') > f.col('aod_30_prev'))
                           & (f.col('aod_30_prev')==0)
                           & (f.col('aod_60') + f.col('aod_90') + f.col('aod_120')+ f.col('aod_150') + f.col('aod_180') ==0)
                    )
        .withColumn('30_to_60', 
                            (f.col('aod_30') < (f.col('aod_30_prev')))
                           & (f.col('aod_60') > f.col('aod_60_prev'))
                           & (f.col('aod_60_prev')==0)
                           & (f.col('aod_90') + f.col('aod_120')+ f.col('aod_150') + f.col('aod_180') ==0)
                    )
        .withColumn('60_to_90', 
                           (f.col('aod_60') < (f.col('aod_60_prev')))
                           & (f.col('aod_90') > f.col('aod_90_prev'))
                           & (f.col('aod_90_prev')==0)
                           & (f.col('aod_120')+ f.col('aod_150') + f.col('aod_180') ==0)
                    )
        .withColumn('90_to_120', 
                           (f.col('aod_90') < (f.col('aod_90_prev')))
                           & (f.col('aod_120') > f.col('aod_120_prev'))
                           & (f.col('aod_120_prev')==0)
                           & ( f.col('aod_150') + f.col('aod_180') ==0)
                    )
        .withColumn('120_to_150', 
                            (f.col('aod_120') < (f.col('aod_120_prev')))
                           & (f.col('aod_150') > f.col('aod_150_prev'))
                           & (f.col('aod_150_prev')==0)
                           & (f.col('aod_180') ==0)
                    )
        .withColumn('150_to_180', 
                           (f.col('aod_150') < (f.col('aod_150_prev')))
                           & (f.col('aod_180') > f.col('aod_180_prev'))
                           & (f.col('aod_180_prev')==0)
                    )
     .filter(f.col('curr_to_30') 
             | f.col('30_to_60')
             | f.col('60_to_90')
             |f.col('90_to_120')
             |f.col('120_to_150')
             |f.col('150_to_180')
             )
     # .filter(f.col('fs_acct_id') == 501074774)
)

# COMMAND ----------

# DBTITLE 1,curr to 30 output
df_aod_in_30 = (
        df_aod_in
        .filter(f.col('curr_to_30'))  
        .select('fs_acct_id','create_date', *ls_col)
        .withColumn('aod_ind', f.lit('curr_to_30'))
        .withColumnRenamed('create_date', 'event_date')
                    
       #  .limit(10)
        ) 
        
# maybe need to include the previous fields for QA purpose 

# COMMAND ----------

# DBTITLE 1,30 to 60 output
df_aod_in_30_to_60 = (
      df_aod_in
        .join(df_aod_in_30
              .select('fs_acct_id')
              .distinct()
              , ['fs_acct_id'], 'inner')
        .filter(f.col('30_to_60'))  
        .select('fs_acct_id','create_date', *ls_col)
        .withColumn('aod_ind', f.lit('30_to_60'))
        .withColumnRenamed('create_date', 'event_date')
       #  .limit(10)
)

# COMMAND ----------

# DBTITLE 1,union together
df_aod_in_event = (df_aod_in_30
        .union(df_aod_in_30_to_60)
        )

# COMMAND ----------

display(df_aod_in_event.filter(f.col('fs_acct_id') ==357063499
                               )
        )

# COMMAND ----------

display(df_aod_in_30_to_60
       .agg(f.count('fs_acct_id')
            , f.countDistinct('fs_acct_id')
            )
       )

# COMMAND ----------

# MAGIC %md
# MAGIC ### QA

# COMMAND ----------

ls_col_prev = [col +'_prev' for col in ls_col]
print(ls_col_prev)

df_aod_in_30_qa = (
     df_aod_in
        .filter(f.col('curr_to_30'))  
        .select('fs_acct_id','create_date', *ls_col, *ls_col_prev )
)


df_aod_in_30_qa = (
     df_aod_in
        .filter(f.col('curr_to_30'))  
        .select('fs_acct_id','create_date', *ls_col, *ls_col_prev )
)


# COMMAND ----------

display(df_aod_in_30_qa.filter(
    f.col('fs_acct_id')==466512621)
        )

# COMMAND ----------

display(df_aod_in_30
        .agg(f.countDistinct('fs_acct_id'))
        ) # 397467 


# check distribution 
display(df_aod_in_30
        .withColumn('cnt', f.count('*')
                    .over(Window.partitionBy('fs_acct_id'))
                    )
        .groupBy('cnt')
        .agg(f.countDistinct('fs_acct_id').alias('cnt_acct'))
        .withColumn('sum_cnt', f.sum('cnt_acct').over(Window.partitionBy()
                                                      )
                    )
        .withColumn('pct', f.col('cnt_acct')/f.col('sum_cnt'))
)

# COMMAND ----------

# DBTITLE 1,export to excel
display(df_aod_in_30
        .select('fs_acct_id')
        .distinct()
)


# COMMAND ----------

display(
    df_aod_in_30_qa
    .withColumn('cnt', f.count('*').over(Window.partitionBy('fs_acct_id')))
    .filter(f.col('cnt') ==6)
    .limit(100)
)

# COMMAND ----------

display(df_aod
        .filter(f.col('fs_acct_id') == 450024671)
        )

# COMMAND ----------

display(df_aod_in_30
        .agg(
            f.countDistinct('fs_acct_id')
        )
        )
       #  397467/545214 = 72% account in past 180 days entry at least once collection?

       

# COMMAND ----------

# MAGIC %md
# MAGIC ### QA AOD 30 - 60 

# COMMAND ----------

display(df_aod_in_30_to_60
        .agg(f.countDistinct('fs_acct_id')
             )
        )
        # 41670 / 545214 = 7.6% will go to next pharse? 

# COMMAND ----------

display(df_aod_in_30.filter(f.col('fs_acct_id')
                            ==357063499)
        )

# COMMAND ----------

display(df_aod_in_30_to_60.limit(10))

# COMMAND ----------

   display(df_aod_movement
        .withColumn('curr_to_30', 
                        (f.col('aod_current') < (f.col('aod_current_prev')))
                           & (f.col('aod_30') > f.col('aod_30_prev'))
                           & (f.col('aod_30_prev')==0)
                           & (f.col('aod_60') + f.col('aod_90') + f.col('aod_120')+ f.col('aod_150') + f.col('aod_180') ==0)
                    )
        .withColumn('30_to_60', 
                     (f.col('aod_30') < (f.col('aod_30_prev')))
                           & (f.col('aod_60') > f.col('aod_60_prev'))
                           & (f.col('aod_60_prev')==0)
                           & (f.col('aod_90') + f.col('aod_120')+ f.col('aod_150') + f.col('aod_180') ==0)
                    )
        .filter(f.col('fs_acct_id') == 473375424)
   )

# COMMAND ----------

display (
    df_aod
    .withColumn(
        'movement_flag', 
        (f.col('aod_current') != f.col('aod_current_prev'))
        | (f.col('aod_30') != f.col('aod_30_prev') )
        | (f.col('aod_60') != f.col('aod_60_prev'))
        | (f.col('aod_90') != f.col('aod_90_prev'))
        | (f.col('aod_120') != f.col('aod_120_prev') )
        | (f.col('aod_150') != f.col('aod_150_prev'))
        | (f.col('aod_180') != f.col('aod_180_prev'))
    )
    # .filter(f.col('movement_flag'))
    .filter(f.col('fs_acct_id') == 473375424)
        )

# COMMAND ----------

display(df_aod_in_30_to_60.filter(f.col('fs_acct_id')==349131860
        ))

display(df_aod_in_30
        .filter(f.col('fs_acct_id')
                ==349131860
                )
        )

# COMMAND ----------

display(
    df_aod.filter(f.col('fs_acct_id')
                       ==483400881)
)

# COMMAND ----------

display(df_aod_movement
        .filter(f.col('fs_acct_id') ==450024671)
        )

# COMMAND ----------

display(df_aod_in_30.filter(f.col('fs_acct_id') ==450024671)
        )
