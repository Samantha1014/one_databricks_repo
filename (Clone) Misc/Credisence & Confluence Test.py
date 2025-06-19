# Databricks notebook source
# Databricks notebook source
import requests
import json
from datetime import datetime, timezone


authUrl = f"https://onenz-api.credisense.io/User/authenticate"
username ="onenz-api-prod@credisense.io"
password = "DA75D99631F1"


# COMMAND ----------

def get_bearer_token(url, username, password):
    url = url
    data = {
        "username": username,
        "password": password
    }
    response = requests.post(url, json=data
                             , verify = False
                             )
    if response.status_code == 200:
        return response.json().get('key')
    else:
        print("Failed to authenticate:", response.text)
        return None

# COMMAND ----------

bearer_token = get_bearer_token(authUrl,username, password)

# COMMAND ----------

print(bearer_token)

# COMMAND ----------

import requests
import json
from datetime import datetime, timezone

# COMMAND ----------

creditSencenseURL = 'onenz-api.credisense.io'

# COMMAND ----------

username = 'onenz-api-prod@credisense.io'
password = 'DA75D99631F1'

# COMMAND ----------

# Function to authenticate and retrieve the bearer token
def get_bearer_token(url, username, password):
    url = url
    data = {
        "username": username,
        "password": password
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json().get('key')
    else:
        print("Failed to authenticate:", response.text)
        return None

# Function to make authenticated request using the bearer token
def make_authenticated_request(bearer_token, method, url, data=None):
    headers = {
        "Authorization": f"Bearer {bearer_token}"
    }
    if method.upper() == 'POST':
        response = requests.post(url, headers=headers, json=data)
    elif method.upper() == 'GET':
        response = requests.get(url, headers=headers)
    elif method.upper() == 'DELETE':
        response = requests.delete(url, headers=headers)
    if response.status_code == 200 or response.status_code == 201:
        #print("Authenticated request successful")
        if not response.content:
            return None
        else:
            return response.json()
       
    else:
        print(response)
        print("Authenticated request failed:", response.text)
 

# COMMAND ----------

# authUrl = f"https://{creditSencenseURL}/User/authenticate"
 
# recordCount = 1
# spark.sql('drop table if exists creditsences.FraudRiskDatabase_ReadbackFull')
# bearer_token = get_bearer_token(authUrl,username, password)
 
# i=1
# while recordCount >0:
#     if bearer_token:
#         geturl = f"https://{creditSencenseURL}/Record/FraudRiskDatabase?page={i}&pageSize=10000&updatedAt.gte={lastRunTime}"
#         lsResponds= make_authenticated_request(bearer_token,'GET', geturl)
#         if len(lsResponds) >0:
#             dfResult = spark.read.json(spark.sparkContext.parallelize([lsResponds]))
#             dfResultSelect = dfResult.select("data.updatedAt" , "data.expirydate","data.customerid","data.casetype","data.createdAt","data.createdBy","data.caseid","data.updatedBy","id","type")
#             dfResultSelect.createOrReplaceTempView("temptable")
#             sqlContext.sql("CREATE TABLE IF NOT EXISTS creditsences.FraudRiskDatabase_ReadbackFull as select * from temptable where 1 =2 ")
#             dfResultSelect.write.format("delta").mode("append").saveAsTable("creditsences.FraudRiskDatabase_ReadbackFull")
#         i+=1
#     recordCount = len(lsResponds)
 
 

# COMMAND ----------

authUrl
#username
#password

# COMMAND ----------

# MAGIC %sh
# MAGIC echo "https://onenz.atlassian.net/wiki/rest/api/content"

# COMMAND ----------



# COMMAND ----------

# MAGIC %sh
# MAGIC curl -I -k  https://onenz.atlassian.net/wiki/rest/api/content

# COMMAND ----------

# MAGIC %sh
# MAGIC curl -X POST -k https://onenz.atlassian.net/wiki/rest/api/content

# COMMAND ----------

# MAGIC %sh
# MAGIC curl -I -k https://onenz-api.credisense.io/User/authenticate

# COMMAND ----------

# MAGIC %sh
# MAGIC curl -X POST -k https://onenz-api.credisense.io/User/authenticate
