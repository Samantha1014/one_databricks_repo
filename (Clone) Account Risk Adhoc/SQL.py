# Databricks notebook source
from jinja2 import Template

# COMMAND ----------

# MAGIC %md
# MAGIC #ifp on service

# COMMAND ----------

vt_param_sql_ifp_srvc_base_core = """
   with ifp_on_service as (
        SELECT
            EVENT_DATE
            , DATE_TRUNC("MONTH", EVENT_DATE) AS EVENT_MONTH
            , EVENT
            , CUSTOMER_SIEBEL_ROW_ID AS cust_src_id
            , ACCT_NUM
            , PRIM_ACCS_NUM
            , CONN_ID
            , SBSCNORDER as siebel_order_num
            , MERCHANDISE_ID
            , OA_IF_DEVICE_TYPE as ifp_type
            , OA_IF_DEVICE as ifp_model
            , DEVICE_RRP as ifp_rrp
            , HOB_IND
            , OA_IF_START_DT as ifp_term_start_date
            , OA_IF_END_DT as ifp_term_end_date
            , OA_IF_TERM_TTL as ifp_term
            , UPFRONT_PAYMENT as ifp_pmt_upfront
            , OA_IF_MDP_REV as ifp_pmt_monthly
            , OA_IF_DEVICE_AMOUNT as ifp_value
            , OA_IF_DISCOUNT_VAL_MNTH as ifp_discount_monthly
            , OA_IF_DISCOUNT_VAL as ifp_discount
            , OA_IF_REBATE_VAL as ifp_rebate
            , SALES_CHANNEL_GROUP as ifp_sales_channel_group
            , SALES_CHANNEL as ifp_sales_channel
            , SALES_CHANNEL_BRANCH as ifp_sales_channel_branch
            , SALES_PERS_NAME as ifp_sales_agent
        FROM PROD_AWS_PROD_MASKED.STAGE_PERM.DS_EDW_IFP_EVENT_SUMMARY_ML
        WHERE OA_IF_START_DT BETWEEN ''{{param_start_date}}'' AND ''{{param_end_date}}''
        and event in ('New Interest Free Contract')
        and ifp_type in ('Device')
        and hob_ind in ('Activation')
    ), 
     credit_score AS (
        select 
        CUSTOMER_MARKET_SEGMENT
        ,current_credit_check_score
        , first_credit_check_score
        , first_credit_check_current_status
        , first_credit_check_submit_date
        , current_credit_check_submit_date
        ,customer_alt_source_id
        , account_no
        , account_name
        , account_write_off_amount
        , account_write_off_month
	from prod_account_risk.modelled.f_account_risk_monthly_snapshot
	qualify row_number() over(partition by account_no order by d_snapshot_date_key desc) = 1
    )
   SELECT * FROM ifp_on_service LEFT JOIN credit_score ON ACCT_NUM = ACCOUNT_NO 

"""
vt_param_sql_ifp_srvc_base_core = Template(vt_param_sql_ifp_srvc_base_core)

# COMMAND ----------


