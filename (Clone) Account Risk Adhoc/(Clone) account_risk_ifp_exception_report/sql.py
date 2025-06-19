# Databricks notebook source
from jinja2 import Template

# COMMAND ----------

# MAGIC %md ## active base for consumer & soho

# COMMAND ----------

vt_param_sql_active_base_core = """
    select
        ---top 100 
        tab_conn.conn_id
        , tab_conn.cust_party_id
        , tab_cust.siebel_row_id as cust_src_id
        , tab_cust.cust_full_name
        , tab_cust.given_name
        , tab_cust.family_name
        , tab_conn.curr_mkt_seg_name
        , tab_cust.curr_service_seg_name
        , tab_cust.party_type_name
        , tab_cust.party_status
        , tab_cust.party_start_dt
        , tab_conn.prim_acct_id
        , tab_acct.acct_num
        , tab_acct.siebel_acct_id as acct_src_id
        , tab_acct.acct_ref
        , tab_acct.acct_cntct_person
        , tab_acct.acct_actv_dt
        , tab_conn.prim_accs_id
        , tab_conn.prim_accs_num
        , tab_conn.conn_full_name
        , tab_conn.first_actvn_dt
        , tab_conn.dctv_dt
        , tab_conn.dss_insert_dttm
        , tab_conn.dss_update_dttm
        , current_date() as data_extract_date
        , row_number() over(partition by tab_conn.prim_accs_num order by tab_conn.dctv_dt desc, tab_conn.first_actvn_dt) as row_index
        
    from prod_aws_prod_masked.stage_perm.ds_edw_connection tab_conn

    --- rate plan to exclude FWA/home phone
    left join prod_aws_prod_masked.stage_perm.ds_edw_prod_rate_plan tab_rateplan
        on trim(tab_conn.usg_plan_key) = trim(tab_rateplan.prod_key)

    --- account related info
    left join prod_aws_prod_masked.stage_perm.ds_edw_acct tab_acct
        on tab_conn.prim_acct_id = tab_acct.acct_id
        and tab_conn.cust_party_id = tab_acct.cust_party_id

    --- customer related info
    left join prod_aws_prod_masked.stage_perm.ds_edw_party_customer tab_cust
        on tab_conn.cust_party_id = tab_cust.party_id

    where 1 = 1
        and trim(tab_conn.curr_fin_conn_stat_name) = 'Active'
        and trim(tab_conn.curr_src_conn_stat_name) = 'Active'
        and trim(tab_conn.prim_acct_type_cd) = 'OA'
        and trim(tab_conn.curr_mkt_seg_name) in ('Consumer', 'SOHO')
        and tab_conn.conn_cnt_grp = 'Y'
        and tab_conn.rev_cnt_grp = 'Y'
        and (
            tab_conn.first_actvn_dt <= '{{param_end_date}}'
            and tab_conn.dctv_dt >= '{{param_start_date}}'
        )
        and trim(lower(tab_rateplan.prod_grp_desc_lvl2)) not in ('home phone plus', 'broadband wireless')

"""

vt_param_sql_active_base_core = Template(vt_param_sql_active_base_core)

# COMMAND ----------

vt_param_sql_active_base = """
    with extract_target as (
            {{param_sql_active_base_core}}
    )
    select *
    from extract_target
    where row_index = 1
"""

vt_param_sql_active_base = Template(vt_param_sql_active_base)

# COMMAND ----------

# MAGIC %md ## ifp on service

# COMMAND ----------

vt_param_sql_ifp_srvc = """
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
    )

    , active_base_core as (
        {{param_sql_active_base_core}}
    )

    , active_base as (
        select *
        from active_base_core
        where row_index = 1
    )

    select
        tab_ifp.*
    from ifp_on_service tab_ifp
    inner join active_base tab_conn
        on tab_ifp.conn_id = tab_conn.conn_id
    WHERE ifp_term_start_date <= '{{param_end_date}}'
        AND ifp_term_end_date >= '{{param_start_date}}'

"""

vt_param_sql_ifp_srvc = Template(vt_param_sql_ifp_srvc)

# COMMAND ----------

# MAGIC %md ## ifp on bill

# COMMAND ----------

vt_param_sql_ifp_bill = """

WITH EDW_SALES_CHANNEL_TMP_00 AS (
    SELECT
        CHNL_GRP_NAME
        , CHNL_CHAIN
        , CHNL_LOC
        , COUNT(*) AS CNT
        , MAX(DSS_UPDATE_DTTM) AS DSS_UPDATE_DTTM_MAX
    FROM PROD_AWS_PROD_MASKED.STAGE_PERM.DS_EDW_SALES_CHANNEL
    GROUP BY CHNL_GRP_NAME, CHNL_CHAIN, CHNL_LOC
)

, EDW_SALES_CHANNEL_TMP_01 AS (
    SELECT
        *
        , ROW_NUMBER() OVER(PARTITION BY LOWER(CHNL_CHAIN), LOWER(CHNL_LOC) ORDER BY DSS_UPDATE_DTTM_MAX DESC, CNT DESC) AS INDEX
    FROM EDW_SALES_CHANNEL_TMP_00
)

, EDW_SALES_CHANNEL_TARGET AS (
    SELECT
        CHNL_GRP_NAME
        , CHNL_CHAIN
        , CHNL_LOC
    FROM EDW_SALES_CHANNEL_TMP_01
    WHERE INDEX = 1
    ORDER BY CHNL_CHAIN, CHNL_LOC, INDEX
)

, D_ACCT_FULL AS (
    SELECT
        BILLING_ACCOUNT_NUMBER
        , BILLING_ACCOUNT_SOURCE_ID
        , ROW_NUMBER() OVER(PARTITION BY BILLING_ACCOUNT_SOURCE_ID ORDER BY RECORD_END_DATE_TIME DESC, RECORD_START_DATE_TIME DESC) AS ROW_INDEX
    FROM PROD_PDB_MASKED.MODELLED.D_BILLING_ACCOUNT
)

, D_ACCT AS (
    SELECT
        BILLING_ACCOUNT_NUMBER
        , BILLING_ACCOUNT_SOURCE_ID
    FROM D_ACCT_FULL
    WHERE ROW_INDEX = 1
)

, active_base_core as (
    {{param_sql_active_base_core}}
)

, active_base as (
    select *
    from active_base_core
    where row_index = 1
)

, EXTRACT_TARGET AS (
    SELECT
        DISTINCT
        ---TOP 100
          TAB_A.F_IFP_SALES_TRANSACTION_KEY as f_ifp_txn_id
        , TAB_A.D_IFP_SALES_TRANSACTION_DATE_KEY
        , TAB_A.D_HARDWARE_KEY 
        , TAB_A.D_BILLING_ACCOUNT_KEY
        , TAB_A.IFP_SALES_TRANSACTION_ID as ifp_txn_id
        , TAB_A.IFP_SALES_TRANSACTION_ID_TABLE_NAME
        , TAB_A.MODEL_NAME as ifp_model
        , TAB_A.MODEL_DESCRIPTION as ifp_model_desc
        ---, SHA2(TAB_A.BILLING_ACCOUNT_SOURCE_ID || 's_org_ext') AS BILLING_ACCOUNT_SOURCE_ID_SHA2
        ---, TAB_A.BILLING_ACCOUNT_SOURCE_ID_SHA2
        , TAB_A.BILLING_ACCOUNT_SOURCE_ID as acct_src_id
        ---, TAB_A.BILLING_ACCOUNT_SOURCE_ID_TABLE_NAME
        , TAB_C.BILLING_ACCOUNT_NUMBER as acct_num
        , TAB_A.SOURCE_SYSTEM_CODE
        , TAB_A.TRANSACTION_TYPE as event_sub_type
        , TAB_A.LINKED_SERVICE_ID
        , TAB_A.MERCHANDISE_ID
        , TAB_A.IFP_TYPE
        , TAB_A.IFP_RECOMMENDED_RETAIL_PRICE as ifp_rrp
        , TAB_A.IFP_TOTAL_VALUE as ifp_value
        , TAB_A.IFP_TRADE_IN_VALUE as ifp_trade_in
        , TAB_A.IFP_UPFRONT_PAYMENT_AMOUNT as ifp_pmt_upfront
        , TAB_A.IFP_MONTHLY_PAYMENT_VALUE as ifp_pmt_monthly
        , TAB_A.IFP_TERM_MONTHS as ifp_term
        , TAB_A.IFP_DISCOUNT_AMOUNT as ifp_discount
        , TAB_A.IFP_MONTHLY_DISCOUNT_AMOUNT as ifp_discount_monthly
        , TAB_A.IFP_DISCOUNT_TERM_MONTHS
        , TAB_A.IFP_REBATE_AMOUNT as ifp_rebate
        , TAB_A.IFP_START_DATE as ifp_term_start_date
        , TAB_A.IFP_END_DATE as ifp_term_end_date
        , TAB_A.ASSET_ROW_ID as siebel_asset_row_id
        , TAB_A.ORDER_NUMBER as siebel_order_num
        
        ---, TAB_A.ORDER_DATE AS ORDER_DTTM
        ---, TO_DATE(TAB_A.ORDER_DATE) AS ORDER_DATE
        ---, TO_DATE(date_trunc('MONTH', TAB_A.ORDER_DATE)) AS ORDER_MONTH
        ---, TAB_A.ORDER_SUB_TYPE

        , TAB_A.ORDER_DATE AS event_dttm
        , TO_DATE(TAB_A.ORDER_DATE) AS event_date
        , TO_DATE(date_trunc('MONTH', TAB_A.ORDER_DATE)) AS event_month
        , TAB_A.ORDER_SUB_TYPE as event


        , TAB_A.SALES_BRANCH as ifp_sales_channel_branch
        , TAB_A.SALES_CHANNEL as ifp_sales_channel
        , TAB_B.CHNL_GRP_NAME as ifp_sales_channel_group
        , tab_a.salesperson_first_name as ifp_sales_agent_name_first
        , tab_a.salesperson_last_name as ifp_sales_agent_name_last
        , TAB_A.CANCEL_REASON as ifp_cancel_reason
        , TAB_A.CURRENT_RECORD_IND
        , TAB_A.IFP_TRADE_IN_REFERENCE
        , TAB_A.IFP_TRADE_IN_DESC
        ---, TAB_B.CHNL_GRP_NAME AS SALES_CHANNEL_GROUP  
    FROM PROD_PDB_MASKED.MODELLED.F_IFP_SALES_TRANSACTION TAB_A

    LEFT JOIN EDW_SALES_CHANNEL_TARGET TAB_B
        ON LOWER(TAB_A.SALES_CHANNEL) = LOWER(TAB_B.CHNL_CHAIN)
        AND LOWER(TAB_A.SALES_BRANCH) = LOWER(TAB_B.CHNL_LOC)

    LEFT JOIN D_ACCT TAB_C
        ON TAB_A.BILLING_ACCOUNT_SOURCE_ID = TAB_C.BILLING_ACCOUNT_SOURCE_ID

    INNER JOIN active_base TAB_D
        ON TAB_C.BILLING_ACCOUNT_NUMBER = TAB_D.ACCT_NUM

)
SELECT 
    *
FROM EXTRACT_TARGET
WHERE ifp_term_start_date <= '{{param_end_date}}'
    AND ifp_term_end_date >= '{{param_start_date}}'

"""

vt_param_sql_ifp_bill = Template(vt_param_sql_ifp_bill)

# COMMAND ----------

# MAGIC %md ## output upload & update

# COMMAND ----------

vt_param_sql_upload_remove = """
    DELETE FROM {{param_table}}
    WHERE REPORTING_DATE = '{{param_date}}'
"""

vt_param_sql_upload_remove = Template(vt_param_sql_upload_remove)

# COMMAND ----------

vt_param_sql_upload_update_01 = """
    UPDATE {{param_table}}
    SET ACTIVE_FLAG = 'N'
"""

vt_param_sql_upload_update_01 = Template(vt_param_sql_upload_update_01)

# COMMAND ----------

vt_param_sql_upload_update_02 = """
    UPDATE {{param_table}}
    SET ACTIVE_FLAG = 'Y'
    WHERE REPORTING_DATE = (
        select max(REPORTING_DATE)
        from {{param_table}}
    )
"""

vt_param_sql_upload_update_02 = Template(vt_param_sql_upload_update_02)
