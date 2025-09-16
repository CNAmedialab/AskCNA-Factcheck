from elasticsearch import Elasticsearch
import json
from typing import List, Dict, Any, Optional, Union
import os
from dotenv import load_dotenv

load_dotenv()

### Elasticsearch
es = Elasticsearch(os.getenv("es_host"), basic_auth=(os.getenv("es_username"), os.getenv("es_password")), request_timeout=3600)

##### 原生 query 搜尋，可自定義 query
### query 搜尋 (自定義 query)
def es_search_queryJSON(es, index, query):
    response = es.search(index=index, body=query)
    #print(f"Found {response['hits']['total']['value']} documents")
    return response['hits']['hits']


##### 欄位字串搜尋
### 使用 match 單一搜尋
def es_search_string_match(es, index, field_name, search_string, recall_size=10):
    query = { "size": recall_size, "query": { "match": { field_name: search_string } } }
    response = es.search(index=index, body=query)
    #print(f"Found {response['hits']['total']['value']} documents")
    return response['hits']['hits']


### 使用 term 單一搜尋
def es_search_string_term(es, index, field_name, search_string, recall_size=10):
    query = { "size": recall_size, "query": { "term": { field_name: search_string } } }
    response = es.search(index=index, body=query)
    #print(f"Found {response['hits']['total']['value']} documents")
    return response['hits']['hits']





##### 顯示日期
### 使用特定日期搜尋
def es_search_certain_date(es, index, date_column_name, date, size=1000):
    query = {"query":{"bool":{"must":[{"range":{date_column_name:{"gte":date,"lte":date}}}],"must_not":[],"should":[]}},"from":0,"size":size,"sort":[],"aggs":{}}
    response = es.search(index=index, body=query)
    #print(f"Found {response['hits']['total']['value']} documents")
    return response['hits']['hits']


### 使用日期範圍搜尋
def es_search_date_range(es, index, date_column_name, start_date, end_date): # 前後皆含
    query = {"query":{"bool":{"must":[{"range":{date_column_name:{"gte":start_date,"lte":end_date}}}],"must_not":[],"should":[]}},"from":0,"size":1000,"sort":[],"aggs":{}}
    response = es.search(index=index, body=query)
    #print(f"Found {response['hits']['total']['value']} documents")
    return response['hits']['hits']





##### Vector Search
### 純粹向量搜尋
def es_vector_search(es, index, embedding_column_name, input_embedding, recall_size=10):
    query = {
        "size": recall_size,
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "exists": {
                                    "field": embedding_column_name
                                }
                            }
                        ]
                    }
                },
                "script": {
                    "source": f"if (doc['{embedding_column_name}'].size() > 0) {{ return cosineSimilarity(params.query_vector, '{embedding_column_name}') + 1.0; }} else {{ return 0.0; }}",
                    "params": {"query_vector": input_embedding}
                }
            }
        }
    }
    response = es.search(index=index, body=query)
    return response['hits']['hits']


### 智能向量搜尋 - 支持可選日期篩選（基於PID）
def es_smart_vector_search(es, index, embedding_column_name, input_embedding, pid_column_name=None, start_date=None, end_date=None, recall_size=10):
    """
    智能向量搜尋函數，根據輸入參數自動選擇合適的搜尋方式
    :param pid_column_name: PID欄位名稱（可選）
    :param start_date: 開始日期，格式：YYYYMMDD（可選）
    :param end_date: 結束日期，格式：YYYYMMDD（可選）
    
    邏輯：
    - 如果沒有提供日期參數，使用純向量搜尋
    - 如果只提供一個日期，將其作為開始和結束日期（搜尋特定日期）
    - 如果提供兩個日期，進行日期範圍搜尋（基於PID前綴）
    """
    # 沒有提供日期參數，使用純向量搜尋
    if not pid_column_name or (start_date is None and end_date is None):
        return es_vector_search(es, index, embedding_column_name, input_embedding, recall_size)
    
    # 處理日期邏輯
    if start_date is not None and end_date is None:
        # 只提供開始日期，設為特定日期搜尋
        end_date = start_date
    elif start_date is None and end_date is not None:
        # 只提供結束日期，設為特定日期搜尋
        start_date = end_date
    
    # 使用日期範圍向量搜尋（基於PID前綴）
    return es_vector_search_with_date_range(es, index, embedding_column_name, input_embedding, pid_column_name, start_date, end_date, recall_size)


### 向量搜尋結合日期範圍篩選（基於PID前綴）
def es_vector_search_with_date_range(es, index, embedding_column_name, input_embedding, pid_column_name, start_date, end_date, recall_size=10):
    """
    使用PID前綴進行日期範圍篩選的向量搜尋
    :param pid_column_name: PID欄位名稱
    :param start_date: 開始日期，格式：YYYYMMDD
    :param end_date: 結束日期，格式：YYYYMMDD
    
    PID格式：YYYYMMDDNNNN（前8位是日期，後4位是編號）
    """
    # 將日期轉換為PID前綴範圍
    start_pid_prefix = f"{start_date}0002"  # 該日期的最小PID
    end_pid_prefix = f"{end_date}9999"      # 該日期的最大PID
    
    query = {
        "size": recall_size,
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "range": {
                                    pid_column_name: {
                                        "gte": start_pid_prefix,
                                        "lte": end_pid_prefix
                                    }
                                }
                            },
                            {
                                "exists": {
                                    "field": embedding_column_name
                                }
                            }
                        ]
                    }
                },
                "script": {
                    "source": f"if (doc['{embedding_column_name}'].size() > 0) {{ return cosineSimilarity(params.query_vector, '{embedding_column_name}') + 1.0; }} else {{ return 0.0; }}",
                    "params": {"query_vector": input_embedding}
                }
            }
        }
    }
    response = es.search(index=index, body=query)
    return response['hits']['hits']


### 加入1個query條件篩選。
def es_vector_search_with_queryString(es, index, embedding_column_name, input_embedding, query_column_name, filter_query, recall_size=10):
    query = { "size": recall_size,  "query": { "bool": { "must": [ { "term": { query_column_name: filter_query } }, { "script_score": { "query": {"match_all": {}}, "script": { "source": f"cosineSimilarity(params.query_vector, '{embedding_column_name}') + 1.0", "params": { "query_vector": input_embedding } } } } ] } } }
    response = es.search(index=index, body=query)
    return response['hits']['hits']


### 加入多個query條件篩選。
def es_advanced_vector_search(
    es,
    index: str,
    embedding_column_name: str,
    input_embedding,
    filters: List[Dict[str, Any]],
    recall_size: int = 10
) -> List[Dict[str, Any]]:
    """
    執行向量搜尋，並結合多種過濾條件，支持同一欄位多個match_phrase query (OR 條件)
    :param filters: 過濾條件列表，每個條件是一個字典，格式如下：
                    {
                        "type": "term"/"match"/"match_phrase"/"range",
                        "field": "欄位名稱",
                        "value": 過濾值 或 [過濾值1, 過濾值2, ...]
                    }
    :param recall_size: 返回的結果數量
    :return: 匹配的文檔列表
    """
    must_conditions = []
    should_conditions = []

    # 處理所有過濾條件
    for filter_condition in filters:
        filter_type = filter_condition["type"]
        field = filter_condition["field"]
        value = filter_condition["value"]

        if filter_type == "term":
            if isinstance(value, list):
                must_conditions.append({"terms": {field: value}})
            else:
                must_conditions.append({"term": {field: value}})
        elif filter_type == "match" or filter_type == "match_phrase":
            # 如果是多個值，則應將它們加到 should_conditions 中，作為 OR 條件
            if isinstance(value, list):
                for v in value:
                    should_conditions.append({filter_type: {field: v}})
            else:
                should_conditions.append({filter_type: {field: value}})
        elif filter_type == "range":
            must_conditions.append({"range": {field: value}})
        else:
            must_conditions.append({filter_type: {field: value}})
            # raise ValueError(f"不支持的過濾類型: {filter_type}")

    # 構建完整的查詢
    query = {
        "size": recall_size,
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "must": must_conditions,
                        "should": should_conditions,
                        "minimum_should_match": 1 if should_conditions else 0  # 至少匹配一個 should 條件
                    }
                },
                "script": {
                    "source": f"cosineSimilarity(params.query_vector, '{embedding_column_name}') + 1.0",
                    "params": {"query_vector": input_embedding}
                }
            }
        }
    }

    # 執行搜尋
    response = es.search(index=index, body=query)
    return response['hits']['hits']
#  filters = [
#     {"type": "term", "field": "image_type.raw", "value": "其他照片"},
#     {"type": "match", "field": "exp", "value": "降雨"}#,
#     {"type": "range", "field": "price", "value": {"gte": 100, "lte": 500}}
# ]
# results = es_advanced_vector_search(es, "lab_photo_search", "embedding_desc", input_embedding, filters, recall_size=20)


### query加權搜尋
def es_keyword_weighted_search(
    es,
    index: str,
    embedding_column_name: str,
    input_embedding: List[float],
    keyword_fields: Optional[List[Dict[str, Any]]] = None,
    filters: Optional[List[Dict[str, Any]]] = None,
    recall_size: int = 10
) -> List[Dict[str, Any]]:
    """
    執行向量搜尋，可選擇性地結合關鍵字加權和多種過濾條件
    :param keyword_fields: 可選的關鍵字欄位列表，每個項目是一個字典，格式如下：
                           {
                               "field": "欄位名稱",
                               "keywords": ["關鍵字1", "關鍵字2", ...],
                               "weight": 加權值 (可選，默認為1.0)
                           }
    :param filters: 可選的過濾條件列表，每個條件是一個字典，格式如下：
                    {
                        "type": "term"/"match"/"range",
                        "field": "欄位名稱",
                        "value": 過濾值
                    }
    :param recall_size: 返回的結果數量
    :return: 匹配的文檔列表
    """
    must_conditions = []
    should_conditions = []

    # 處理關鍵字加權（如果提供）
    if keyword_fields:
        for field_info in keyword_fields:
            field = field_info["field"]
            keywords = field_info["keywords"]
            weight = field_info.get("weight", 1.0)
            
            for keyword in keywords:
                should_conditions.append({
                    "match": {
                        field: {
                            "query": keyword,
                            "boost": weight
                        }
                    }
                })

    # 處理所有過濾條件（如果提供）
    if filters:
        for filter_condition in filters:
            filter_type = filter_condition["type"]
            field = filter_condition["field"]
            value = filter_condition["value"]

            if filter_type == "term":
                must_conditions.append({"term": {field: value}})
            elif filter_type == "match":
                must_conditions.append({"match": {field: value}})
            elif filter_type == "range":
                must_conditions.append({"range": {field: value}})
            else:
                raise ValueError(f"不支持的過濾類型: {filter_type}")

    # 添加向量搜尋條件
    must_conditions.append({
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": f"cosineSimilarity(params.query_vector, '{embedding_column_name}') + 1.0",
                "params": {"query_vector": input_embedding}
            }
        }
    })

    # 構建完整的查詢
    query = {
        "size": recall_size,
        "query": {
            "bool": {
                "must": must_conditions
            }
        }
    }

    # 如果有 should 條件，添加到查詢中
    if should_conditions:
        query["query"]["bool"]["should"] = should_conditions

    # 將查詢轉換為JSON字符串，然後再解析回Python對象
    # 這可以解決一些JSON序列化的問題
    query_json = json.dumps(query)
    query = json.loads(query_json)

    # 執行搜尋
    try:
        response = es.search(index=index, body=query)
        return response['hits']['hits']
    except Exception as e:
        #print(f"搜索出錯: {str(e)}")
        return []





##### 顯示資料
### 顯示搜尋結果，可自定義結果數量。
def es_search_extend_data(es_reponse_hits_hits, show_data=10):
    count = 0
    for hit in es_reponse_hits_hits:
        #print(f"Score: {hit['_score']}")
        #print(f"Document ID: {hit['_id']}")
        #print(f"Document source: {hit['_source']}")
        #print("---")
        count += 1
        if count >= show_data:
            break
    return


### 顯示搜尋結果，可自定義顯示欄位和結果數量。
def es_search_extend_data_spec(es_response_hits_hits, fields_to_show=None, show_data=10):
    for count, hit in enumerate(es_response_hits_hits, 1): # 計數從 1 開始
        #print(f"Score: {hit['_score']}")
        #print(f"文件 ID: {hit['_id']}")
        
        """if fields_to_show:
            for field in fields_to_show:
                if field in hit['_source']:
                    print(f"{field}: {hit['_source'][field]}")
        else:
            print(f"文件內容: {hit['_source']}")
        
        print("---")"""
        
        if count >= show_data:
            break
    return


