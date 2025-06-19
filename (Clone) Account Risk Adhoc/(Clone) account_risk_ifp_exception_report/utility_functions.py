# Databricks notebook source
from pyspark import sql
from pyspark.sql import SparkSession
import pandas as pd

# COMMAND ----------

def lower_col_names(
    df: sql.DataFrame
) -> sql.DataFrame:

    df_out = (
        df
        .toDF(*[c.lower() for c in df.columns])
    )

    return df_out

# COMMAND ----------

def add_missing_cols(
    df: sql.DataFrame
    , ls_fields: list
    , vt_assign_value=None
) -> sql.DataFrame:
    
    df_out = df
    for column in [column for column in ls_fields if column not in df.columns]:
        df_out = df_out.withColumn(column, f.lit(vt_assign_value))
        
    return df_out

# COMMAND ----------

def export_data(
    df: sql.DataFrame
    , export_path: str
    , export_format: str
    , export_mode: str
    , flag_overwrite_schema: bool
    , flag_dynamic_partition: bool
    , ls_dynamic_partition: list = None
):
    export_obj = (
        df
        .write
        .format(export_format)
        .mode(export_mode)
    )

    if flag_overwrite_schema:
        export_obj = (
            export_obj
            .option("overwriteSchema", "true")
        )

    if flag_dynamic_partition:
        export_obj = (
            export_obj
            .partitionBy(ls_dynamic_partition)
            .option("partitionOverwriteMode", "dynamic")
        )

    (
        export_obj
        .save(export_path)
    )

# COMMAND ----------

def transform_ifp_srvc(
    df_input: sql.DataFrame
    , vt_param_ssc_start_date: str
    , vt_param_ssc_end_date: str
):
    
    df_base_ifp_curr = (
        df_input
        .withColumn(
            "ifp_term_end_date"
            , f.add_months(f.col("ifp_term_start_date"), f.col("ifp_term"))
        )
        .withColumn(
            "ifp_term_end_date"
            , f.date_add(f.col("ifp_term_end_date"), f.lit(-1))
        )
        # within ifp term
        .filter(f.col("ifp_term_start_date") <= vt_param_ssc_end_date)
        .filter(f.col("ifp_term_end_date") >= vt_param_ssc_start_date)

        # event date is not in the future
        .filter(f.col("event_date") <= vt_param_ssc_end_date)

        # exclud duplicated conn_id + siebel_order
        .withColumn(
            "index"
            , f.row_number().over(
                Window
                .partitionBy("conn_id", "event", "siebel_order_num")
                .orderBy(f.desc("event_date"))
            )
        )
        .filter(f.col("index") == 1)
        
        # change of conn_id due to segment changed?
        .withColumn(
            "index"
            , f.row_number().over(
                Window
                .partitionBy("fs_cust_id", 'fs_acct_id', 'fs_srvc_id', "event", "siebel_order_num")
                .orderBy(f.desc("event_date"))
            )
        )
        .filter(f.col("index") == 1)

        # ifp with rrp > 0
        .filter(f.col("ifp_rrp") > 0)
    )

    # activation events
    df_base_ifp_activate_curr = (
        df_base_ifp_curr
        .filter(f.col("event").isin(["New Interest Free Contract"]))
        .withColumn(
            "ifp_id"
            , f.monotonically_increasing_id()
        )
        .select(
            "event_date"
            , "ifp_id"
            , "conn_id"
            , "fs_cust_id"
            , "fs_acct_id"
            , "fs_srvc_id"
            , "fs_ifp_order_id"
            , f.col("siebel_order_num").alias("ifp_order_num")
            , f.col("merchandise_id").alias("ifp_imei")
            , "ifp_model"
            , "ifp_rrp"
            , "ifp_value"
            , "ifp_rebate"
            , "ifp_pmt_upfront"
            , "ifp_discount"
            , "ifp_term"
            , "ifp_term_start_date"
            , "ifp_term_end_date"
            , "ifp_pmt_monthly"
            , "ifp_discount_monthly"
            , "ifp_sales_channel_group"
            , "ifp_sales_channel"
            , "ifp_sales_channel_branch"
            , "ifp_sales_agent"
        )
    )

    # early cancel events
    df_base_ifp_cancel_curr = (
        df_base_ifp_curr
        .filter(f.col("event").isin("Cancel Before Term End"))
        .withColumn(
            "ifp_early_cancel_id"
            , f.monotonically_increasing_id()
        )
        .select(
            f.col("event_date").alias("ifp_early_cancel_date")
            , "ifp_early_cancel_id"
            , f.lit("Y").alias("ifp_early_cancel_flag")
            , "conn_id"
            , "fs_cust_id"
            , "fs_acct_id"
            , "fs_srvc_id"
            , f.col("siebel_order_num").alias("ifp_order_num")
            , f.col("merchandise_id").alias("ifp_imei_early_cancel")
            , f.col("ifp_model").alias("ifp_model_early_cancel")
        )
    )

    # early cancel match by siebel order id
    df_proc_ifp_01_curr = (
        df_base_ifp_activate_curr
        # exclude order_num = unknown for order num match
        .filter(f.col("ifp_order_num") != 'Unknown')
        .join(
            df_base_ifp_cancel_curr
            , ["conn_id", "fs_srvc_id", "fs_acct_id", "fs_cust_id", "ifp_order_num"]
            , "inner"
        )
        # keep the early cancel event if it is after the IFP terms starts
        .filter(f.col("ifp_early_cancel_date") >= f.col("event_date"))
        # keep the earliest cancel event only
        .withColumn(
            "index"
            , f.row_number().over(
                Window
                .partitionBy("ifp_id")
                .orderBy("ifp_early_cancel_date")
            )
        )
        .filter(f.col("index") == 1)
        # cleaning for problematic early cancel date
        .withColumn(
            "ifp_early_cancel_date"
            , f.when(
                f.col("ifp_term_start_date") > f.col("ifp_early_cancel_date")
                , f.col("ifp_term_start_date")
            ).otherwise(
                f.col("ifp_early_cancel_date")
            )
        )
        .withColumn("ifp_early_cancel_match_type", f.lit("order number"))
    )

    # exclude processed ifp activate events
    df_base_ifp_activate_remain_curr = (
        df_base_ifp_activate_curr
        .join(
            df_proc_ifp_01_curr
            .select("conn_id", "ifp_order_num")
            .distinct()
            , ["conn_id", "ifp_order_num"]
            , "leftanti"
        )
    )

    # exclude processed ifp early cancel events
    df_base_ifp_cancel_remain_curr = (
        df_base_ifp_cancel_curr
        .join(
            df_proc_ifp_01_curr
            .select("conn_id", "ifp_order_num")
            .distinct()
            , ["conn_id", "ifp_order_num"]
            , "leftanti"
        )
    )

    # early cancel match by ifp model
    df_proc_ifp_02_curr = (
        df_base_ifp_activate_remain_curr
        .join(
            df_base_ifp_cancel_remain_curr
            .drop("ifp_order_num")
            .withColumn("ifp_model", f.col("ifp_model_early_cancel"))
            , ["conn_id", "fs_srvc_id", "fs_acct_id", "fs_cust_id", "ifp_model"]
            , "inner"
        )
        # keep the early cancel event if it is after the IFP terms starts
        .filter(f.col("ifp_early_cancel_date") >= f.col("event_date"))
        # keep the earliest cancel event only
        .withColumn(
            "index"
            , f.row_number().over(
                Window
                .partitionBy("ifp_id")
                .orderBy("ifp_early_cancel_date")
            )
        )
        .filter(f.col("index") == 1)
        # cleaning for problematic early cancel date
        .withColumn(
            "ifp_early_cancel_date"
            , f.when(
                f.col("ifp_term_start_date") > f.col("ifp_early_cancel_date")
                , f.col("ifp_term_start_date")
            ).otherwise(
                f.col("ifp_early_cancel_date")
            )
        )
        .withColumn("ifp_early_cancel_match_type", f.lit("ifp model"))
    )
    
    # ifp without any early cancelled
    df_proc_ifp_03_curr = (
        df_base_ifp_activate_curr
        .join(
            df_proc_ifp_01_curr
            .select("conn_id", "ifp_order_num")
            .distinct()
            , ["conn_id", "ifp_order_num"]
            , "leftanti"
        )
        .join(
            df_proc_ifp_02_curr
            .select("conn_id", "ifp_order_num")
            .distinct()
            , ["conn_id", "ifp_order_num"]
            , "leftanti"
        )
    )
    
    # merge all ifp events
    ls_param_proc_ifp_fields_merged = (
        np.unique(
            df_proc_ifp_01_curr.columns
            + df_proc_ifp_02_curr.columns
            + df_proc_ifp_03_curr.columns
        )
        .tolist()
    )
    
    df_proc_ifp_01_curr = add_missing_cols(
        df_proc_ifp_01_curr
        , ls_param_proc_ifp_fields_merged
    )
    
    df_proc_ifp_02_curr = add_missing_cols(
        df_proc_ifp_02_curr
        , ls_param_proc_ifp_fields_merged
    )
    
    df_proc_ifp_03_curr = add_missing_cols(
        df_proc_ifp_03_curr
        , ls_param_proc_ifp_fields_merged
    )
    
    df_proc_ifp_curr = (
        df_proc_ifp_01_curr
        .unionByName(df_proc_ifp_02_curr)
        .unionByName(df_proc_ifp_03_curr)
        .fillna(value='N', subset=["ifp_early_cancel_flag"])
        .withColumn(
            "ifp_end_date"
            , f.when(
                f.col("ifp_early_cancel_flag") == 'Y'
                , f.col("ifp_early_cancel_date")
            ).otherwise(f.col("ifp_term_end_date"))
        )
        .filter(f.col("ifp_term_start_date") <= vt_param_ssc_end_date)
        .filter(f.col("ifp_end_date") >= vt_param_ssc_start_date)
    )
    
    # define basic event type
    # NEW/CANCEL_0/CANCEL_1/FINISH
    df_proc_ifp_curr = (
        df_proc_ifp_curr
        .withColumn(
            "ifp_activate_flag"
            , f.when(
                f.col("ifp_term_start_date")
                .between(vt_param_ssc_start_date, vt_param_ssc_end_date)
                , f.lit("Y")
            ).otherwise("N")
        )
        .withColumn(
            "ifp_finish_flag"
            , f.when(
                (
                    f.col("ifp_term_end_date")
                    .between(
                        vt_param_ssc_start_date
                        , vt_param_ssc_end_date
                    )
                )
                & (
                    f.col("ifp_early_cancel_flag") != 'Y'
                )
                , f.lit("Y")
            ).otherwise("N")
        )
        .withColumn(
            "ifp_event_type"
            , f.when(
                (f.col("ifp_activate_flag") == 'Y')
                & (f.col("ifp_early_cancel_flag") == 'Y')
                , f.lit("cancel at the first month")
            )
            .when(
                f.col("ifp_activate_flag") == 'Y'
                , f.lit("new")
            )
            .when(
                f.col("ifp_finish_flag") == 'Y'
                , f.lit("finish")
            )
            .when(
                f.col("ifp_early_cancel_flag") == 'Y'
                , f.lit("early cancel")
            )
            .otherwise(
                f.lit("remain")
            )
        )
    )
    
    # define transfer event type
    # TRANSFER_IN/TRANSFER_OUT
    df_proc_ifp_curr = (
        df_proc_ifp_curr
        
        # combined ID
        .withColumn(
            "check_id"
            , f.concat_ws("_", f.col("fs_cust_id"), f.col("fs_acct_id"), f.col("fs_srvc_id"))
        )
        
        # index by IFP order
        .withColumn(
            "check_index"
            , f.row_number().over(
                Window
                .partitionBy("ifp_order_num")
                .orderBy(f.desc("event_date"))
            )
        )
        
        # max event date by IFP order
        .withColumn(
            "check_event_date"
            , f.max("event_date").over(
                Window
                .partitionBy("ifp_order_num")
            )
        )
        
        # event end date based on the next event by IFP order
        .withColumn(
            "check_event_end_date"
            , f.lag("event_date", 1).over(
                Window
                .partitionBy("ifp_order_num")
                .orderBy(f.desc("event_date"))
            )
        )
        
        # # events by IFP order
        .withColumn(
            "check_cnt"
            , f.size(
                f.collect_set("check_id").over(
                    Window
                    .partitionBy("ifp_order_num")
                )
            )
        )
        
        # exclude flag
        .withColumn(
            "check_cond"
            , f.when(
                f.col("ifp_order_num") == "Unknown"
                , f.lit("Y")
            )
            .when(
                f.col("check_cnt") == 1
                , f.lit("Y")
            )
            .when(
                (f.col("check_event_date").between(vt_param_ssc_start_date, vt_param_ssc_end_date)) 
                & (f.col("check_event_end_date").between(vt_param_ssc_start_date, vt_param_ssc_end_date))
                , f.lit("Y")
            )
            .when(
                f.col("check_index") == 1
                , f.lit("Y")
            )
            .otherwise(f.lit("N"))
        )
        
        # transfer event type
        .withColumn(
            "ifp_transfer_flag"
            , f.when(
                (f.col("ifp_order_num") != "Unknown")
                & (f.col("check_cnt") > 1)
                & (f.col("ifp_event_type") == "remain")
                & (f.col("check_event_date").between(vt_param_ssc_start_date, vt_param_ssc_end_date))
                , f.lit("Y")
            ).otherwise(f.lit("N"))
        )
        
        # .withColumn(
        #     "ifp_event_type"
        #     , f.when(
        #         (f.col("ifp_order_num") != "Unknown")
        #         & (f.col("check_cnt") > 1)
        #         & (f.col("ifp_event_type") == "remain")
        #         & (f.col("check_event_date").between(vt_param_ssc_start_date, vt_param_ssc_end_date))
        #         , f.lit("TRANSFER")
        #     ).otherwise(f.col("ifp_event_type"))
        # )
        
        
        # transfer in/out
        .withColumn(
            "ifp_event_type"
            , f.when(
                # (f.col("ifp_event_type") == "TRANSFER")
                # & (f.col("check_index") == 1)
                (f.col("ifp_transfer_flag") == "Y")
                & (f.col("event_date").between(vt_param_ssc_start_date, vt_param_ssc_end_date))
                , f.lit("transfer in")
            )
            .when(
                # (f.col("ifp_event_type") == "TRANSFER")
                # & (f.col("check_index") != 1)
                (f.col("ifp_transfer_flag") == "Y")
                & (f.col("check_event_end_date").between(vt_param_ssc_start_date, vt_param_ssc_end_date))
                , f.lit("transfer out")
            )
            .otherwise(f.col("ifp_event_type"))
        )
        .filter(f.col("check_cond") == 'Y')
    )
    
    df_output_curr = (
        df_proc_ifp_curr
        #.withColumn("reporting_date", f.lit(vt_param_ssc_reporting_date))
        #.withColumn("reporting_cycle_type", f.lit(vt_param_ssc_reporting_cycle_type))
        .withColumn(
            "fs_ifp_id"
            , f.concat_ws(
                "_"
                , f.col("fs_cust_id"), f.col("fs_acct_id")
                , f.col("fs_srvc_id"), f.col("fs_ifp_order_id")
            )
        )
        .select(
            #"reporting_date"
            #, "reporting_cycle_type"
            'fs_cust_id'
            , "fs_acct_id"
            , "fs_srvc_id"
            , "conn_id"
            , "fs_ifp_id"
            , "fs_ifp_order_id"
            
            , f.lit("on service").alias("ifp_level")
            , f.lit("device").alias("ifp_type")
            
            , f.col("event_date").alias("ifp_event_date")
            , "ifp_order_num"
            , f.col("ifp_imei").alias("ifp_merchandise_id")
            , "ifp_model"
            
            , "ifp_term"
            , "ifp_term_start_date"
            , "ifp_term_end_date"
            
            , "ifp_rrp"
            , "ifp_value"
            , "ifp_pmt_upfront"
            , "ifp_discount"
            , f.lit(0).alias("ifp_trade_in")
            , "ifp_rebate"
            , "ifp_pmt_monthly"
            , "ifp_discount_monthly"
            
            , "ifp_sales_channel_group"
            , "ifp_sales_channel"
            , "ifp_sales_channel_branch"
            , "ifp_sales_agent"
            
            , "ifp_event_type"
            , "ifp_activate_flag"
            , "ifp_finish_flag"
            , "ifp_transfer_flag"
            , "ifp_early_cancel_flag"
            , "ifp_early_cancel_date"
            , "ifp_early_cancel_match_type"
            
            , "ifp_end_date"
            , f.current_date().alias("data_update_date")
            , f.current_timestamp().alias("data_update_dttm")
        )
    )

    return df_output_curr

# COMMAND ----------

def transform_ifp_bill(
    df_input: sql.DataFrame
    , vt_param_ssc_start_date: str
    , vt_param_ssc_end_date: str
):
    df_base_ifp_curr = (
        df_input
        # within ifp term
        .filter(f.col("ifp_term_start_date") <= vt_param_ssc_end_date)
        .filter(f.col("ifp_term_end_date") >= vt_param_ssc_start_date)
        # event date is not in the future
        .filter(f.col("event_date") <= vt_param_ssc_end_date)
        .withColumn(
            "index"
            , f.row_number().over(
                Window
                .partitionBy("fs_acct_src_id", "fs_ifp_id", "fs_ifp_order_id")
                .orderBy(f.desc("event_date"))
            )
        )
        .filter(f.col("index") == 1)
    )

    df_base_ifp_activate_curr = (
        df_base_ifp_curr
        .filter(f.col("event") == 'Connect')
        # take the first connect event as the activation event
        .withColumn(
            "index"
            , f.row_number().over(
                Window
                .partitionBy("fs_acct_src_id", "fs_ifp_id")
                .orderBy("event_date")
            )
        )
        .filter(f.col("index") == 1)
        .select(
            "event_date"
            , "fs_acct_id"
            , "fs_acct_src_id"
            , "fs_ifp_id"
            , "fs_ifp_order_id"
            , "ifp_type"
            , f.col("siebel_order_num").alias("ifp_order_num")
            , "merchandise_id"
            , "ifp_model"
            , "ifp_model_desc"
            , "ifp_rrp"
            , "ifp_value"
            , "ifp_rebate"
            , "ifp_pmt_upfront"
            , "ifp_discount"
            , "ifp_trade_in"
            , "ifp_term"
            , "ifp_term_start_date"
            , "ifp_term_end_date"
            , "ifp_pmt_monthly"
            , "ifp_discount_monthly"
            , "ifp_sales_channel_group"
            , "ifp_sales_channel"
            , "ifp_sales_channel_branch"
            , "ifp_sales_agent_name_first"
            , "ifp_sales_agent_name_last"
        )
    )
    
    # cancel events
    df_base_ifp_cancel_curr = (
        df_base_ifp_curr
        .filter(f.col("event") == "Disconnect")
        # .join(
        #     df_base_ifp_activate_curr
        #     .select('fs_acct_src_id', "fs_ifp_id", "fs_ifp_order_id")
        #     , ['fs_acct_src_id', "fs_ifp_id", "fs_ifp_order_id"]
        #     , "leftanti"
        # )
        # take the last cancel event as the effect cancel event
        .withColumn(
            "index"
            , f.row_number().over(
                Window
                .partitionBy("fs_acct_src_id", "fs_ifp_id")
                .orderBy(f.desc("event_date"))
            )
        )
        .filter(f.col("index") == 1)
        .select(
            f.col("event_date").alias("ifp_early_cancel_date")
            , "fs_acct_id"
            , "fs_acct_src_id"
            , "fs_ifp_id"
            , f.col("ifp_cancel_reason").alias("ifp_early_cancel_reason")
            , f.col("fs_ifp_order_id").alias("ifp_early_cancel_order_id")
            , f.col("merchandise_id").alias("merchandise_id_early_cancel")
            , f.col("ifp_model").alias("ifp_model_early_cancel")
            , f.lit('Y').alias("ifp_early_cancel_flag")
        )
    )
    
    # reactivate events
    df_base_ifp_reactivate_curr = (
        df_base_ifp_curr
        .filter(f.col("event") != "Disconnect")
        .join(
            df_base_ifp_activate_curr
            .select("fs_acct_id", 'fs_acct_src_id', "fs_ifp_id", "fs_ifp_order_id")
            , ["fs_acct_id", 'fs_acct_src_id', "fs_ifp_id", "fs_ifp_order_id"]
            , "leftanti"
        )
        # take the last modify event as the effect modify event
        .withColumn(
            "index"
            , f.row_number().over(
                Window
                .partitionBy("fs_acct_src_id", "fs_ifp_id")
                .orderBy(f.desc("event_date"))
            )
        )
        .filter(f.col("index") == 1)
        .select(
            f.col("event_date").alias("ifp_reactivate_date")
            , "fs_acct_id"
            , "fs_acct_src_id"
            , "fs_ifp_id"
            , f.col("fs_ifp_order_id").alias("ifp_reactivate_order_id")
            , f.col("event_sub_type").alias("ifp_reactivate_event")
            , f.col("merchandise_id").alias("merchandise_id_reactivate")
            , f.col("ifp_model").alias("ifp_model_reactivate")
            , f.lit('Y').alias("ifp_reactivate_flag")
        )
    )
    
    df_proc_ifp_00_curr = (
        df_base_ifp_activate_curr
        .join(df_base_ifp_cancel_curr, ["fs_acct_id", "fs_acct_src_id", "fs_ifp_id"], "inner")
        .join(df_base_ifp_reactivate_curr, ["fs_acct_id", "fs_acct_src_id", "fs_ifp_id"], "left")
        .fillna(value='N', subset=['ifp_early_cancel_flag', "ifp_reactivate_flag"])
    )
    
    ls_param_ifp_norm_fields = df_base_ifp_activate_curr.columns
    
    # early cancel events
    df_proc_ifp_01_curr = (
        df_proc_ifp_00_curr
        .filter(f.col("ifp_early_cancel_date") >= f.col("event_date"))
        .filter(
            (f.col("ifp_early_cancel_date") > f.col("ifp_reactivate_date"))
            | (f.col("ifp_reactivate_flag") == 'N')
        )
        .select(
            *ls_param_ifp_norm_fields
            , "ifp_early_cancel_flag"
            , "ifp_early_cancel_order_id"
            , "ifp_early_cancel_date"
            , "ifp_early_cancel_reason"
        )
    )
    
    # reinstate events
    df_proc_ifp_02_curr = (
        df_proc_ifp_00_curr
        .filter(f.col("ifp_early_cancel_date") >= f.col("event_date"))
        .filter(f.col("ifp_early_cancel_date") <= f.col("ifp_reactivate_date"))
        .select(
            *ls_param_ifp_norm_fields
            , f.lit("N").alias("ifp_early_cancel_flag")
            , "ifp_early_cancel_order_id"
            , "ifp_early_cancel_date"
            , "ifp_early_cancel_reason"
            , "ifp_reactivate_flag"
            , "ifp_reactivate_order_id"
            , 'ifp_reactivate_event'
            , "ifp_reactivate_date"
        )
    )
    
    # ifp without any early cancelled
    df_proc_ifp_03_curr = (
        df_base_ifp_activate_curr
        .join(
            df_proc_ifp_01_curr
            .select("fs_acct_src_id", "fs_ifp_id")
            .distinct()
            , ["fs_acct_src_id", "fs_ifp_id"]
            , "leftanti"
        )
        .join(
            df_proc_ifp_02_curr
            .select("fs_acct_src_id", "fs_ifp_id")
            .distinct()
            , ["fs_acct_src_id", "fs_ifp_id"]
            , "leftanti"
        )
    )
    
    # merge all ifp events
    ls_param_proc_ifp_fields_merged = (
        np.unique(
            df_proc_ifp_01_curr.columns
            + df_proc_ifp_02_curr.columns
            + df_proc_ifp_03_curr.columns
        )
        .tolist()
    )
    
    df_proc_ifp_01_curr = add_missing_cols(
        df_proc_ifp_01_curr
        , ls_param_proc_ifp_fields_merged
    )
    
    df_proc_ifp_02_curr = add_missing_cols(
        df_proc_ifp_02_curr
        , ls_param_proc_ifp_fields_merged
    )
    
    df_proc_ifp_03_curr = add_missing_cols(
        df_proc_ifp_03_curr
        , ls_param_proc_ifp_fields_merged
    )
    
    df_proc_ifp_curr = (
        df_proc_ifp_01_curr
        .unionByName(df_proc_ifp_02_curr)
        .unionByName(df_proc_ifp_03_curr)
        .fillna(value='N', subset=["ifp_early_cancel_flag", "ifp_reactivate_flag"])
        .withColumn(
            "ifp_end_date"
            , f.when(
                f.col("ifp_early_cancel_flag") == 'Y'
                , f.col("ifp_early_cancel_date")
            ).otherwise(f.col("ifp_term_end_date"))
        )
        .filter(f.col("ifp_term_start_date") <= vt_param_ssc_end_date)
        .filter(f.col("ifp_end_date") >= vt_param_ssc_start_date)
    )
    
    
    # define basic event type
    # NEW/CANCEL_0/CANCEL_1/FINISH
    df_proc_ifp_curr = (
        df_proc_ifp_curr
        .withColumn(
            "ifp_activate_flag"
            , f.when(
                f.col("ifp_term_start_date")
                .between(vt_param_ssc_start_date, vt_param_ssc_end_date)
                , f.lit("Y")
            ).otherwise("N")
        )
        .withColumn(
            "ifp_finish_flag"
            , f.when(
                (
                    f.col("ifp_term_end_date")
                    .between(
                        vt_param_ssc_start_date
                        , vt_param_ssc_end_date
                    )
                )
                & (
                    f.col("ifp_early_cancel_flag") != 'Y'
                )
                , f.lit("Y")
            ).otherwise("N")
        )
        .withColumn(
            "ifp_reactivate_flag2"
            , f.when(
                (f.col("ifp_reactivate_flag") == 'Y')
                & (
                    f.col("ifp_reactivate_date")
                    .between(vt_param_ssc_start_date, vt_param_ssc_end_date)
                )
                & (
                    ~f.col("ifp_early_cancel_date")
                    .between(vt_param_ssc_start_date, vt_param_ssc_end_date)
                )
                , f.lit('Y')
            )
            .otherwise(f.lit('N'))
        )
        .withColumn(
            "ifp_event_type"
            , f.when(
                (f.col("ifp_activate_flag") == 'Y')
                & (f.col("ifp_early_cancel_flag") == 'Y')
                , f.lit("cancel at the first month")
            )
            .when(
                f.col("ifp_activate_flag") == 'Y'
                , f.lit("new")
            )
            .when(
                f.col("ifp_finish_flag") == 'Y'
                , f.lit("finish")
            )
            .when(
                f.col("ifp_early_cancel_flag") == 'Y'
                , f.lit("early cancel")
            )
            .when(
                f.col("ifp_reactivate_flag2") == 'Y'
                , f.lit("reactivate")
            )
            .otherwise(
                f.lit("remain")
            )
        )
    )
    
    
    # define transfer event type
    # TRANSFER_IN/TRANSFER_OUT
    df_proc_ifp_curr = (
        df_proc_ifp_curr
        # index by IFP order
        .withColumn(
            "check_index"
            , f.row_number().over(
                Window
                .partitionBy("fs_ifp_id")
                .orderBy(f.desc("event_date"))
            )
        )
        
        # max event date by IFP order
        .withColumn(
            "check_event_date"
            , f.max("event_date").over(
                Window
                .partitionBy("fs_ifp_id")
            )
        )
        
        # event end date based on the next event by IFP order
        .withColumn(
            "check_event_end_date"
            , f.lag("event_date", 1).over(
                Window
                .partitionBy("fs_ifp_id")
                .orderBy(f.desc("event_date"))
            )
        )
        
        # # events by IFP order
        .withColumn(
            "check_cnt"
            , f.size(
                f.collect_set("fs_acct_src_id").over(
                    Window
                    .partitionBy("fs_ifp_id")
                )
            )
        )

        # exclude flag
        .withColumn(
            "check_cond"
            , f.when(
                f.col("check_cnt") == 1
                , f.lit("Y")
            )
            .when(
                (f.col("check_event_date").between(vt_param_ssc_start_date, vt_param_ssc_end_date)) 
                & (f.col("check_event_end_date").between(vt_param_ssc_start_date, vt_param_ssc_end_date))
                , f.lit("Y")
            )
            .when(
                f.col("check_index") == 1
                , f.lit("Y")
            )
            .otherwise(f.lit("N"))
        )
        
        # transfer event type
        .withColumn(
            "ifp_transfer_flag"
            , f.when(
                (f.col("check_cnt") > 1)
                & (f.col("ifp_event_type") == "remain")
                & (f.col("check_event_date").between(vt_param_ssc_start_date, vt_param_ssc_end_date))
                , f.lit("Y")
            ).otherwise(f.lit("N"))
        )

        # transfer in/out
        .withColumn(
            "ifp_event_type"
            , f.when(
                
                (f.col("ifp_transfer_flag") == "Y")
                & (f.col("event_date").between(vt_param_ssc_start_date, vt_param_ssc_end_date))
                , f.lit("transfer in")
            )
            .when(
                
                (f.col("ifp_transfer_flag") == "Y")
                & (f.col("check_event_end_date").between(vt_param_ssc_start_date, vt_param_ssc_end_date))
                , f.lit("transfer out")
            )
            .otherwise(f.col("ifp_event_type"))
        )
        .filter(f.col("check_cond") == 'Y')
    )
    
    df_output_curr = (
        df_proc_ifp_curr
        #.withColumn("reporting_date", f.lit(vt_param_ssc_reporting_date))
        #.withColumn("reporting_cycle_type", f.lit(vt_param_ssc_reporting_cycle_type))
        .withColumn(
            "ifp_type"
            , f.when(
                f.col("ifp_type") == "Ifp-Device"
                , "device"
            ).otherwise(f.lit("accessory"))
        )
        .withColumn("ifp_level", f.lit("on bill"))
        .withColumn("ifp_sales_agent", f.concat(f.col("ifp_sales_agent_name_first"), f.lit(" "), f.col("ifp_sales_agent_name_last")))
        .withColumn(
            "ifp_term_start_date"
            , f.col("ifp_term_start_date").cast("date")
        )
        .withColumn(
            "ifp_term_end_date"
            , f.col("ifp_term_end_date").cast("date")
        )
        .withColumn(
            "ifp_end_date"
            , f.col("ifp_end_date").cast("date")
        )
        .select(
            #"reporting_date"
            #, "reporting_cycle_type"
            "fs_acct_id"
            , "fs_acct_src_id"
            , "fs_ifp_id"
            , "fs_ifp_order_id"
            , "ifp_level"
            , "ifp_type"
            
            , f.col("event_date").alias("ifp_event_date")
            , "ifp_order_num"
            , f.col("merchandise_id").alias("ifp_merchandise_id")
            
            , "ifp_model"
            , "ifp_model_desc"
            
            , "ifp_term"
            , "ifp_term_start_date"
            , "ifp_term_end_date"
            
            , "ifp_rrp"
            , "ifp_value"
            , "ifp_pmt_upfront"
            , "ifp_discount"
            , "ifp_trade_in"
            , "ifp_rebate"
            , "ifp_pmt_monthly"
            , "ifp_discount_monthly"
            
            , "ifp_sales_channel_group"
            , "ifp_sales_channel"
            , "ifp_sales_channel_branch"
            , "ifp_sales_agent"
            
            , "ifp_event_type"
            , "ifp_activate_flag"
            , "ifp_finish_flag"
            , "ifp_transfer_flag"
            
            , "ifp_early_cancel_flag"
            , "ifp_early_cancel_order_id"
            , "ifp_early_cancel_date"
            , "ifp_early_cancel_reason"
            
            , "ifp_reactivate_flag"
            , "ifp_reactivate_order_id"
            , 'ifp_reactivate_event'
            , "ifp_reactivate_date"
            
            , "ifp_end_date"
            , f.current_date().alias("data_update_date")
            , f.current_timestamp().alias("data_update_dttm")
        )
    )

    return df_output_curr

