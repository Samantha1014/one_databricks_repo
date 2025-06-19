# Databricks notebook source
### libraries
import pyspark
import os

import re
import numpy as np

from pyspark import sql 
from pyspark.sql import Window
from pyspark.sql import functions as f
from pyspark.sql.types import IntegerType, StructType
from pyspark.sql.functions import monotonically_increasing_id, row_number

from itertools import islice, cycle

# COMMAND ----------

dir_data_parent = "/mnt/feature-store-prod-lab"
dir_data_parent_shared = os.path.join(dir_data_parent, "")
dir_data_parent_users = os.path.join(dir_data_parent, "")

# COMMAND ----------

dir_data_raw =  os.path.join(dir_data_parent_shared, 'd100_raw')
dir_data_meta = os.path.join(dir_data_parent_users, 'd000_meta')
dir_data_stg = os.path.join(dir_data_parent_users, "d200_staging")
dir_data_int =  os.path.join(dir_data_parent_users, "d200_intermediate")
dir_data_prm =  os.path.join(dir_data_parent_users, "d300_primary/d301_mobile_oa_consumer")
dir_data_fea =  os.path.join(dir_data_parent_users, "d400_feature/d401_mobile_oa_consumer")
dir_data_mvmt = os.path.join(dir_data_parent_users, "d500_movement/d501_mobile_oa_consumer")
dir_data_serv = os.path.join(dir_data_parent_users, "d600_serving")
dir_data_tmp =  os.path.join(dir_data_parent_users, "d999_tmp")

# COMMAND ----------

df_fea_bill = spark.read.format("delta").load(os.path.join(dir_data_fea, "fea_bill_cycle_billing_6"))
dir_data_fs_support = "/mnt/feature-store-dev/dev_users/dev_el/2024q4_moa_account_risk"
df_uat_fea_bill = spark.read.format("delta").load(os.path.join(dir_data_fs_support, "d400_feature/d401_mobile_oa_consumer/fea_bill_cycle_billing_6"))

# COMMAND ----------

dbutils.fs.ls('dbfs:/mnt/feature-store-prod-lab/d200_staging/d299_src/stg_brm_bill_t')

# COMMAND ----------

df_stg_bill = spark.read.format('delta').load(os.path.join(dir_data_stg,'d299_src/stg_brm_bill_t'))

# COMMAND ----------

df_test = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d500_movement/d501_mobile_oa_consumer/mvmt_writeoff/')

# COMMAND ----------

display(df_test.limit(100))

# COMMAND ----------

display(df_fea_bill.limit(10))

# COMMAND ----------

display(df_fea_bill
        .filter(f.col('reporting_cycle_type') == 'calendar cycle')
        .groupBy('reporting_date')
        .pivot('bill_payment_timeliness_status')
        .agg(f.countDistinct('fs_acct_id')
             )
        #.withColumn('sum', f.col('null') + f.col('credit_bill') + f.col('early') 
        #            + f.col('late') + f.col('miss') +f.col('ontime')
        #)
)

# COMMAND ----------

display(  
        df_uat_fea_bill
        .filter(f.col('reporting_cycle_type') == 'calendar cycle')
        .groupBy('reporting_date')
        .pivot('bill_payment_timeliness_status')
        .agg(f.countDistinct('fs_acct_id'))
         .withColumn('sum', f.col('null') + f.col('credit_bill') + f.col('early') 
                    + f.col('late') + f.col('miss') +f.col('ontime')
        )
)



# COMMAND ----------

df_uat_fea_bill_test  = (
    df_uat_fea_bill
    .filter(f.col('reporting_cycle_type') == 'calendar cycle')
    .filter(f.col('bill_payment_timeliness_status') =='late')
    .filter(f.col('reporting_date') == '2024-03-31')
)


df_fea_bill_test  = (
    df_fea_bill
    .filter(f.col('reporting_cycle_type') == 'calendar cycle')
    .filter(f.col('bill_payment_timeliness_status') =='late')
    .filter(f.col('reporting_date') == '2024-03-31')
)

# COMMAND ----------

display(df_fea_bill
        .filter(f.col('fs_acct_id') 
                == '410064051')
        .filter(f.col('reporting_date') == '2024-03-31')
        .filter(f.col('reporting_cycle_type') =='calendar cycle')
        )

# COMMAND ----------

display(df_fea_payment)

# COMMAND ----------

display(df_fea_payment)

# COMMAND ----------

display(df_uat_fea_bill
        .filter(f.col('fs_acct_id') 
                == '410064051')
        .filter(f.col('reporting_date') == '2024-03-31')
        .filter(f.col('reporting_cycle_type') =='calendar cycle')
)

# COMMAND ----------

display(df_uat_fea_bill_test
        .select('fs_acct_id', 'reporting_date', 'bill_payment_timeliness_status', 'bill_overdue_days')
        .distinct()
        .join(df_fea_bill_test, ['fs_acct_id'], 'anti')
        )
    

# COMMAND ----------

display(df_uat_fea_bill_test
        .select('fs_acct_id', 'reporting_date', 'bill_payment_timeliness_status', 'bill_overdue_days')
        .distinct()
        .join(df_fea_bill_test, ['fs_acct_id'], 'anti')
        .groupBy('bill_overdue_days')
        .agg(f.countDistinct('fs_acct_id').alias('cnt'))
        .withColumn('total_sum', f.sum('cnt').over(Window.partitionBy()))
        .withColumn('pct', f.col('cnt')/ f.col('total_sum'))
        )


# COMMAND ----------

df_uat_fea_bill_test  = (
    df_uat_fea_bill
    .filter(f.col('reporting_cycle_type') == 'calendar cycle')
    .filter(f.col('bill_payment_timeliness_status') =='late')
    .filter(f.col('reporting_date') == '2023-11-30')
)


df_fea_bill_test  = (
    df_fea_bill
    .filter(f.col('reporting_cycle_type') == 'calendar cycle')
    .filter(f.col('bill_payment_timeliness_status') =='late')
    .filter(f.col('reporting_date') == '2023-11-30')
)

# COMMAND ----------

display(df_uat_fea_bill_test
        .select('fs_acct_id', 'reporting_date', 'bill_payment_timeliness_status', 'bill_overdue_days', 'bill_no')
        .distinct()
        .join(df_fea_bill_test, ['fs_acct_id'], 'anti')
        #.groupBy('bill_overdue_days')
        #.agg(f.count('fs_acct_id'))
        )
    

# COMMAND ----------

df_fea_payment = spark.read.format('delta').load('dbfs:/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_payment_method_cycle_rolling_6')

# COMMAND ----------

display(df_uat_fea_bill
        .filter(f.col('fs_acct_id') 
                == '370053984')
        .filter(f.col('reporting_date') == '2023-11-30')
        .filter(f.col('reporting_cycle_type') =='calendar cycle')
)



display(df_fea_bill
        .filter(f.col('fs_acct_id') 
                == '370053984')
        .filter(f.col('reporting_date') == '2023-11-30')
        .filter(f.col('reporting_cycle_type') =='calendar cycle')
        )

# COMMAND ----------

for i in ['2023-10-31', '2023-11-30', '2023-12-31' , '2024-01-31', '2024-02-29','2024-03-31' ]: 
    print(i)
    display(df_fea_bill
            .filter(f.col('reporting_date') == f.lit(i))
            .filter(f.col('reporting_cycle_type') =='calendar cycle')
            .filter(f.col('bill_payment_timeliness_status') == 'late')
            .join(
                df_fea_payment
                .filter(f.col('reporting_date') == f.lit(i))
                .filter(f.col('reporting_cycle_type') =='calendar cycle')
                .filter(f.col('payment_auto_flag_6cycle') ==f.lit('Y'))
                , ['fs_acct_id']
                , 'inner'
            )
            .select('fs_acct_id', 'payment_auto_flag_6cycle', 'bill_overdue_days')
            .distinct()
            .agg(f.countDistinct('fs_acct_id')
                 , f.avg('bill_overdue_days')
                    )
    )

# COMMAND ----------

display(df_fea_bill
        .filter(f.col('reporting_date') == '2023-11-30')
        .filter(f.col('reporting_cycle_type') =='calendar cycle')
        .filter(f.col('bill_payment_timeliness_status') == 'late')
        .groupBy('bill_overdue_days')
        .agg(f.countDistinct('fs_acct_id').alias('cnt'))
        .withColumn('sum', f.sum('cnt').over(Window.partitionBy())
                    )
        .withColumn('pct', f.col('cnt') / f.col('sum'))
        )

display(df_fea_bill
        .filter(f.col('reporting_date') == '2023-12-31')
        .filter(f.col('reporting_cycle_type') =='calendar cycle')
        .filter(f.col('bill_payment_timeliness_status') == 'late')
        .groupBy('bill_overdue_days')
        .agg(f.countDistinct('fs_acct_id').alias('cnt'))
        .withColumn('sum', f.sum('cnt').over(Window.partitionBy()))
        .withColumn('pct', f.col('cnt') / f.col('sum'))
        )

# COMMAND ----------



# COMMAND ----------

display(df_fea_payment.limit(10))

# COMMAND ----------

display(  
        df_fea_bill
        .filter(f.col('reporting_cycle_type') =='calendar cycle')
           .join(df_fea_payment
                 .select('fs_acct_id', 'reporting_date', 'reporting_cycle_type', 'payment_auto_flag_6cycle')
                 .distinct()
                .filter(f.col('reporting_cycle_type') =='calendar cycle')
                .filter(f.col('payment_auto_flag_6cycle') ==f.lit('Y'))
                , ['fs_acct_id', 'reporting_date']
                , 'left')
        .withColumn('bill_payment_timeliness_status_2', 
                    f.when(
                        f.col('payment_auto_flag_6cycle') == f.lit('Y'),
                        'ontime'
                         )
                    .when( (f.col('bill_payment_timeliness_status') == 'late') 
                          & (f.col('bill_overdue_days')<=3), f.lit('grace period')
                         )      
                    .otherwise(f.col('bill_payment_timeliness_status'))
                    )
        .groupBy('reporting_date')
        .pivot('bill_payment_timeliness_status_2')
        .agg(f.countDistinct('fs_acct_id'))
)

# COMMAND ----------

# DBTITLE 1,# grace period check
display(  
        df_fea_bill
        .filter(f.col('reporting_cycle_type') == 'calendar cycle')
        .filter(f.col('bill_payment_timeliness_status') == 'late') 
        .filter(f.col('bill_overdue_days') >=3)
        .groupBy('reporting_date')
        .agg(f.countDistinct('fs_acct_id'))
)

# COMMAND ----------


