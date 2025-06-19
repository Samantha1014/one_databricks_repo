# Databricks notebook source
# MAGIC %md
# MAGIC ## summary 
# MAGIC 1. oa consumer activation since Jan 2023 - 181334 
# MAGIC 2. anti ifp on bill 7 days  - 163159 
# MAGIC 3. anti ifp on serv 7days  - 160239
# MAGIC 4. first activation - new act - 100632
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Library 

# COMMAND ----------

import pyspark
import os
from pyspark import sql 
from pyspark.sql import Window
from pyspark.sql import functions as f
from pyspark.sql import SparkSession  
spark = SparkSession.builder.appName("SQL").getOrCreate()


# COMMAND ----------

dir_pp_unit_base = 'dbfs:/mnt/feature-store-dev/dev_users/dev_el/2024q4_pp_redev/d200_intermediate/d202_mobile_pp/int_intg_service'
dir_oa_consumer = 'dbfs:/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_unit_base/reporting_cycle_type=calendar cycle'

# COMMAND ----------



# COMMAND ----------

df_pp_unit_base  = spark.read.format("delta").load(dir_pp_unit_base)
display(df_pp_unit_base.limit(100))
print(df_pp_unit_base.count()) # 34994323


# COMMAND ----------

display(df_oa_consumer_base.filter(
    f.col('fs_acct_id') == '502216722')
)


# COMMAND ----------

df_oa_consumer_base = spark.read.format("delta").load(dir_oa_consumer)
display(df_oa_consumer_base.limit(100))
print(df_oa_consumer_base.count())

# COMMAND ----------

# get cust_start_date from 2023-01-01 and its latest reporting date record 
spark = SparkSession.builder.appName("SQL").getOrCreate()
df_oa_consumer_base.createOrReplaceTempView("df_oa_consumer")

# COMMAND ----------

# MAGIC %md
# MAGIC ## OA Consumer and PP 

# COMMAND ----------

df_cust_2023 = (
    df_oa_consumer_base
    .select('fs_cust_id','fs_acct_id','fs_srvc_id','first_activation_date')
    .filter(f.col('first_activation_date')>= '2023-01-01')
    .distinct()
)

display(df_cust_2023.count())

# COMMAND ----------

df_pp_unit_base.createOrReplaceTempView("df_pp")
df_pp = spark.sql("""
                       select reporting_date 
                      ,fs_cust_id
                      , fs_acct_id
                      , fs_srvc_id
                      , row_number() over(partition by fs_srvc_id order by reporting_date desc) as rnk 
                       from  df_pp
                      qualify rnk = 1     
                       """)

display(
   spark.sql("""
             select count(1), reporting_date from df_pp
             group by 2
             order by 2
             """)

)
# display(df_pp.limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC ## OA Plan Activation

# COMMAND ----------

# oa plan activation 
dir_oa_plan  = 'dbfs:/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_plan/reporting_cycle_type=calendar cycle'
df_oa_plan = spark.read.format("delta").load(dir_oa_plan)
display(df_oa_plan.limit(100))

# COMMAND ----------

# oa new plan 

df_oa_plan.createOrReplaceTempView("df_oa_plan")
df_oa_plan_2023 = spark.sql("""
                       select reporting_date 
                      ,fs_cust_id
                      , fs_acct_id
                      , fs_srvc_id
                      , plan_start_date
                      , row_number() over(partition by fs_cust_id order by reporting_date desc) as rnk 
                        from  df_oa_plan
                        where plan_start_date >= '2023-01-01'
                      qualify rnk = 1     
                       """)

display(df_oa_plan_2023.limit(100))
display(df_oa_plan_2023.count())  # service level 218760

# COMMAND ----------

# MAGIC %md
# MAGIC ## IFP Data

# COMMAND ----------

# ifp data 
dir_ifp_bill =  'dbfs:/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_ifp_main_on_bill/reporting_cycle_type=calendar cycle' 
dir_ifp_serv = 'dbfs:/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_ifp_main_on_service/reporting_cycle_type=calendar cycle'

df_ifp_bill = spark.read.format("delta").load(dir_ifp_bill)
df_ifp_serv = spark.read.format("delta").load(dir_ifp_serv)



display(df_ifp_bill.limit(100))
display(df_ifp_serv.limit(100))



# COMMAND ----------

# combine ifp on bill and on service with start date >= 2023-01-01 

df_ifp_bill = (
    df_ifp_bill
    .select(
    "reporting_date"
    ,"fs_cust_id"
    ,"fs_acct_id"
    , "fs_acct_src_id"
    ,"ifp_order_date"
    )
    .distinct()
    .withColumn("rnk", f.row_number().over(
        Window.partitionBy("fs_cust_id").orderBy(f.desc("reporting_date"))
    ))
    .filter(f.col("rnk")==1)
    .filter(f.col("ifp_order_date")>="2023-01-01")
)


df_ifp_serv = (
    df_ifp_serv
    .select(
    "reporting_date"
    ,"fs_cust_id"
    ,"fs_acct_id"
    , "fs_srvc_id"
    ,"ifp_order_date"
    )
    .distinct()
    .withColumn("rnk", f.row_number().over(
        Window.partitionBy("fs_cust_id").orderBy(f.desc("reporting_date"))
    ))
    .filter(f.col("rnk")==1)
    .filter(f.col("ifp_order_date")>="2023-01-01")
)
    


# COMMAND ----------

# MAGIC %md
# MAGIC ## Anti Join to Exclude IFP

# COMMAND ----------

# anti join for oa 2023 - ifp on bill  

df_anti_bill = df_cust_2023.alias("a").join(
    df_ifp_bill.alias("b"), 
    (f.col("a.fs_acct_id")==f.col("b.fs_acct_id")) &
    ( f.col("b.ifp_order_date") >= f.col("a.first_activation_date")) & 
    (f.col("b.ifp_order_date") <= f.date_add(f.col("a.first_activation_date"),14))
    , "anti"
)


display(df_anti_bill.count())  # 163159 for 7 days and 162471 for 14 days 
display(df_anti_bill.limit(100))

# COMMAND ----------

# anti join ifp on serv 
df_anti_ifp = df_anti_bill.alias("a").join(
    df_ifp_serv.alias("b"), 
    (f.col("a.fs_srvc_id")==f.col("b.fs_srvc_id")) &
    (f.col("a.first_activation_date") <= f.col("b.ifp_order_date")) & 
    (f.col("b.ifp_order_date") >= f.date_add(f.col("a.first_activation_date"),14))
    , "anti"
)

display(df_anti_ifp.count())  # 160239  for 7 days vs. 159531 for 14 days 

# new act 
display(
df_anti_ifp
.select(
    "fs_cust_id"
    ,"first_activation_date")
.distinct()
.groupBy("first_activation_date")
.agg(f.count("*"))
)




# COMMAND ----------

display(df_anti_ifp.limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Activation Type

# COMMAND ----------

# if frist activatin date is between 2023 means new activation
# if msisdn in oa_plan 2023's service start date > than pp's reporting date, then its a pre to post 
# anything else, it is a resign 

# COMMAND ----------

df_new_act = (
        df_anti_ifp
        .filter(
            (f.col("first_activation_date") >= f.col("cust_start_date")) & 
            (f.col("first_activation_date")<= f.date_add(f.col("cust_start_date"),14))
                )
)

display(df_new_act.count())

# COMMAND ----------

df_oa_first_report = (
df_oa_consumer_base
.select("fs_cust_id", "fs_acct_id", "fs_srvc_id", "reporting_date")
.filter(f.col("first_activation_date")>='2023-01-01')
.distinct() 
.withColumn("rnk", f.row_number().over(
    Window.partitionBy("fs_cust_id","fs_acct_id","fs_srvc_id").orderBy(f.asc("reporting_date"))

))
.filter(f.col("rnk")==1)
)


display(
df_oa_first_report.alias("a").join(
  df_pp.alias("b"),
  (f.col("a.fs_srvc_id") == f.col("b.fs_srvc_id")) & 
  f.col("b.reporting_date").between(f.col("a.reporting_date"),
                                     f.date_add(f.col("a.reporting_date"),60))
  ,"inner"
)
.select("b.*")
.count()
)




# COMMAND ----------


display(
df_anti_ifp.alias("a").join(
  df_pp.alias("b"),
  (f.col("a.fs_srvc_id") == f.col("b.fs_srvc_id")) 
  ,"inner"
)
.select("b.*")
.count()
)

# COMMAND ----------

# pre to post , 60 days period 



df_p2p = (
df_anti_ifp.alias("a").join(
  df_pp.alias("b"),
  (f.col("a.fs_srvc_id") == f.col("b.fs_srvc_id")) & 
  f.col("b.reporting_date").between(f.col("a.first_activation_date"),
                                     f.date_add(f.col("a.first_activation_date"),180))
  ,"inner"
)
.select("b.*")
)


# group by month get account no 
display(df_p2p
        .select ("fs_acct_id","b.reporting_date")
        .distinct()
        .groupBy("reporting_date")
        .agg(f.count("*"))
        .orderBy(f.col("b.reporting_date"))
)


# check inner join for new act and p2p 
# display(
# df_new_act.alias("a").join(df_p2p.alias("b"), 
#                (f.col("a.fs_srvc_id") == f.col("b.fs_srvc_id"))
#                ,"inner"
#                 )
# ) # 2,321 
# print(df_p2p.count()) # 3133

# COMMAND ----------



# COMMAND ----------

# new act anti join p2p 

df_new_act_f = \
df_new_act.alias("a").join(df_p2p.alias("b"), 
               (f.col("a.fs_srvc_id") == f.col("b.fs_srvc_id"))
               ,"anti"
                )

display(df_new_act_f.count()) # pre new act 58614  post new act 56321 

# COMMAND ----------

# MAGIC %md
# MAGIC ## OA Base 

# COMMAND ----------

display(df_oa_consumer_base.select(
    "fs_cust_id"
    ,"fs_acct_id"
    , "reporting_date"
).distinct() 
.groupBy("reporting_date")
.agg(f.count('*'))
.filter(f.col("reporting_date")>='2023-01-01')
.orderBy(f.col("reporting_date"))
)


# 1,533,0879
# 15302328 
df_result = \
(df_oa_consumer_base
        .select(
            "fs_cust_id"
            ,"fs_acct_id"
            ,"reporting_date" 
        ).distinct()
        .alias("a").join( df_ifp_bill.alias("b"), 
              (f.col("a.fs_acct_id") == f.col("b.fs_acct_id")) & 
               (f.col("a.reporting_date")== f.col("b.reporting_date"))
               ,"anti")
        .alias("c").join(df_ifp_serv.alias("d"), 
              (f.col("c.fs_acct_id") == f.col("d.fs_acct_id")) & 
               (f.col("c.reporting_date")== f.col("d.reporting_date"))
               ,"anti")
    )


# COMMAND ----------

display(df_result.select(
    "fs_cust_id"
    ,"fs_acct_id"
    , "reporting_date"
).distinct() 
.groupBy("reporting_date")
.agg(f.count('*'))
.filter(f.col("reporting_date")>='2023-01-01')
.orderBy(f.col("reporting_date"))
)


# COMMAND ----------


