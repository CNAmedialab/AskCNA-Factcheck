import time
import re
import requests
from requests import post
from openai import OpenAI
from es_SearchLib import es_vector_search, es
import os
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

### OpenAI Embedding
def text_embeddings_3(text):
    #搭配aisuite openai升級，修改寫法
    # client = openai.OpenAI()
    client = OpenAI()
    t = client.embeddings.create(model="text-embedding-3-large", input=text)
    return t.data[0].embedding

### 查核點api
def get_check_points(text, media_name=None):
    print(f"[Info] 查核點評估...")
    input = {
        "text": text,
        "media_name": media_name
    }
    url = "https://get-check-points-1007110536706.asia-east1.run.app"
    
    start_time = time.time()
    response = requests.post(url, json=input, timeout=3600)

    if response.status_code == 200:
        response_json = response.json()
        check_points = []
        if response_json.get("Result") == "Y" and "ResultData" in response_json:
            check_points = response_json["ResultData"].get("check_points", [])

            if check_points:
                print(f"[Info] 成功取得查核點")
                print(f">>>{check_points}")
                end_time = time.time()
                print(f"[Info] 查核點API耗時: {end_time - start_time:.2f} 秒")
                return {
                    "Result": "Y",
                    "ResultData": {"check_points": check_points},
                    "Message": "API成功回傳結果"
                }
            else:
                print(f"[Info] 查核點為空")
                end_time = time.time()
                print(f"[Info] 查核點API耗時: {end_time - start_time:.2f} 秒")
                return {
                    "Result": "N",
                    "ResultData": {"check_points": None},
                    "Message": "查核點為空"
                }
    else:
        print(f"[Error] 取得查核點時發生錯誤：{response.status_code}")
        end_time = time.time()
        print(f"[Info] 查核點API耗時: {end_time - start_time:.2f} 秒")
        return {
            "Result": "N",
            "ResultData": {"check_points": None},
            "Message": "API回傳None"
        }


### Openai 判斷es結果跟text的相關性
def es_relation(text, summary):
    client = OpenAI()

    class Relation(BaseModel):
        relation: bool

    response = client.responses.parse(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": "判斷參考資料與要做事時查核的文本的相關性。優先考慮事件、時間、人物、地點的相關性。回傳布林值：相關=true，不相關=false。"},
            {"role": "user", "content": f"參考資料：{summary}\n要做事時查核的文本：{text}\n請回答兩者的相關性。"},
        ],
        text_format=Relation,
    )

    answer = response.output_parsed.relation
    return answer

## 用es搜社稿跟查核中心報告
def es_resources(text): 
    # embedding input
    text_embedding = text_embeddings_3(text)

    # es search CNA
    start_time = time.time()
    print("[Info] 正在搜尋中央社社稿")
    cna_res = es_vector_search(es, index="lab_mainsite_search", embedding_column_name="embeddings",
                              input_embedding=text_embedding, recall_size=10)

    cna_news = []
    if cna_res:
        try:
            print(f"[Info] 找到 {len(cna_res)} 筆社稿資料")
            for item in cna_res:
                source = item['_source']
                data = ({
                        "data_type": "CNA",
                        "title": source.get('h1', ''),
                        "date": source.get('dt', '').replace('/', '-'),
                        "article": source.get('article', ''),
                        "summary": source.get('whatHappen200', ''),
                        "url": f"https://www.cna.com.tw/news/aall/{source.get('pid', '')}.aspx"
                })

                # 相關性檢查
                if es_relation(text, source.get('whatHappen200', '')) == True:
                    cna_news.append(data)
                    print(f"\n >>> 有相關，加入：{source.get('h1', '')}")
                else:
                    print(f"\n >>> 不相關，跳過：{source.get('h1', '')} ")

        except Exception as e:
            print(f"[Error] 查詢社稿資料過程發生錯誤: {str(e)}")
            return []

    else:  
        print("未找到相似CNA社稿資料。")

    # es search TFC
    print("[Info] 正在搜尋查核中心報告")
    tfc_res = es_vector_search(es, index="lab_tfc_search_test", embedding_column_name="embeddings",
                              input_embedding=text_embedding, recall_size=5)  ## 查核告有時候很舊, 只找5筆

    tfc_report = []
    if tfc_res:
        try:
            print(f"[Info] 找到 {len(tfc_res)} 筆查核中心報告資料")
            for item in tfc_res:
                source = item['_source']
                data = ({
                    "data_type": "TFC",
                    "title": source.get('title', ''),
                    "date": source.get('date', '').replace('/', '-'),
                    "article": source.get('full_content', ''),
                    "summary": source.get('summary', ''),
                    "label": source.get('label', ''),
                    "url": source.get('link', ''),

                })

                # 如果summary跟text有關係才加入
                if es_relation(text, source.get('summary', '')) == True:
                    tfc_report.append(data)
                    print(f"\n >>> 有相關，加入：{source.get('title', '')}")
                else:
                    print(f"\n >>> 不相關，跳過：{source.get('title', '')}")

        except Exception as e:
            print(f"[Error] 查詢查核中心報告資料過程發生錯誤: {str(e)}")
            return []

    else:
        print("未找到相似TFC查核中心報告資料。")

    # return
    all_resources = cna_news + tfc_report
    end_time = time.time()
    print(f"[Debug] ES查詢耗時: {end_time - start_time:.2f} 秒")
    return all_resources


if __name__ == "__main__":
    text = "日本首相石破茂宣布辭去自民黨總裁（黨主席）職務後，日圓聞訊走貶，加上投資人預期新政府將推出新的經濟對策，日股今天應聲上漲。"
    
    check_points_list = get_check_points(text, media_name="Chiming")
    
    if check_points_list["Result"] == "Y":
        check_points = check_points_list["ResultData"]["check_points"]
        print(check_points)
    else:
        check_points = None
        print("[Info] 查核點 API 失敗")

    resources = es_resources(text)
    print(f"[Info] 最終有{len(resources)}筆參考資料")