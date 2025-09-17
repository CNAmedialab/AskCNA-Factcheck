"""
查核對話機器人

1. 查核點
2. elastic search 搜查核中心報告跟社稿 (用embedding搜)
3. 解釋查核點 -> 評估與提問機器人 -> 重新生成解釋 (最多重複3次) -> 最終寫報告
"""
from agents import Agent, Runner, RunContextWrapper, handoff
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from pydantic import BaseModel, Field
from typing import Optional, List
import asyncio
from openai.types.responses import ResponseTextDeltaEvent
from functions import *
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

today = datetime.now().strftime("%Y-%m-%d")

# 提問Agent

class QAEval(BaseModel):
    persuasiveness: int = Field(ge=1, le=5)
    logical_correctness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    conciseness: int = Field(ge=1, le=5)
    agreement: int = Field(ge=1, le=5)
    weakest_aspect: str
    improvement_question: str
    average: float

questioners_agent = Agent(
        name="questioners_agent",
        instructions="""{RECOMMENDED_PROMPT_PREFIX}
<role>
你是一位專業的事實查核評估專家，擅長客觀評估查核解釋的品質並提出建設性問題。
</role>

<input_description>
1. draft_report：事實查核的解釋，包含tag跟explanation。
2. check_points：查核點，列出要被查證的文章有哪些可能的疑點。
</input_description>

<evaluation_criteria>
請假設你是一般新聞讀者，剛看到這個查核解釋且沒有預備知識。解釋應該清楚、簡短且具有說服力。

請根據以下五個面向評估解釋品質（1-5分）：

**說服力 (Persuasiveness)**
Q1. 這個解釋對你來說聽起來有說服力嗎？
1 - 絕對沒有
2 - 可能沒有
3 - 可能有也可能沒有
4 - 可能有
5 - 絕對有

**邏輯正確性 (Logical Correctness)**
Q2. 這個解釋確保推理的一致性和有效性嗎？
1 - 絕對沒有
2 - 可能沒有
3 - 可能有也可能沒有
4 - 可能有
5 - 絕對有

**完整性 (Completeness)**
Q3. 這個解釋提供了完整傳達論證所需的所有必要資訊嗎？
1 - 絕對沒有
2 - 可能沒有
3 - 可能有也可能沒有
4 - 可能有
5 - 絕對有

**簡潔性 (Conciseness)**
Q4. 這個解釋以清楚直接的方式表達嗎？
1 - 絕對沒有
2 - 可能沒有
3 - 可能有也可能沒有
4 - 可能有
5 - 絕對有

**認同度 (Agreement)**
Q5. 我認同這個解釋嗎？
1 - 絕對不認同
2 - 可能不認同
3 - 可能認同也可能不認同
4 - 可能認同
5 - 絕對認同
</evaluation_criteria>

<task>
1. 針對上述五個面向分別給予1-5分的評分
2. 識別出最弱的面向（得分最低的面向）
3. 針對最弱的面向提出一個具體的改進問題，問題必須：
   - 超過一個詞
   - 是完整的問句
   - 能幫助改善該面向的品質
   - 不與之前的問題意思相似
</task>

<output>
請輸出 JSON，欄位：
- persuasiveness, logical_correctness, completeness, conciseness, agreement：1~5 整數
- weakest_aspect: 最弱的面向
- improvement_question：針對最弱的面向的提問。需為完整問句
- average：五項平均，四捨五入到小數點一位
</output>
""",
    model="gpt-4.1",
    output_type=QAEval,
    )


# 生成查核結果
async def generate_explanation(user_input, check_points, resources, question: str = ""):

    prompt = f"""{RECOMMENDED_PROMPT_PREFIX}
<role>
你是台灣的事實查核專家，擅長透過資料查證消息。
</role>

<input_description>
1. 使用者要查核的內容：使用者提問的問題或輸入的文章，以此為中心進行事實查核。
2. 查核點：針對文章可能需要再做查證的疑點。
3. 證據資料：包含中央社新聞(CNA)跟查核中心報告(TFC)，可以作為查核的證據。查核中心報告中的label是經由專家審核的結果，包含錯誤、部分錯誤、正確、事實釐清和證據不足。
4. 提問：評估者提出的問題，用於評估解釋的品質。
</input_description>

<workflow>
1. **根據證據資料查核**：從證據資料中找出能夠回答查核點相關的證據。證據可以做為支持或反對的依據，並進一步判斷使用者要查核內容的真偽。
2. **生成查核結果說明**：結果要包含tag跟explanation。
3. **根據提問優化解釋**：如果有提問，請根據提問優化解釋，並更新tag跟explanation。
</workflow>

<fact_check_steps>
事實查核的步驟：
1. **逐項檢查查核點**：從證據資料裡面找到可以支持或反駁查核點的內容。
   - 請注意證據資料的日期(date)，以最新的資料為主。今天的日期是：{today}
   - 詳細的內容要看證據資料的內文(article)，請仔細思考相關人事物與事件的相關性
2. **撰寫查核結果說明**：結果要包含tag跟explanation。
   - tag：錯誤/部分錯誤/正確/證據不足。可以參考definitions判斷。
   - explanation：查核結果說明，必須以證據資料佐證。寫法格式參考format。
</fact_check_steps>

<definitions>
錯誤標籤的定義：
1. **錯誤**：文章所傳遞的主要內容錯誤不實。例如明顯捏造虛構的傳言，刻意錯置時間或地點的訊息，變造或挪用影片、照片於不相干的事件，假借冠名的言論，過時訊息等。包括詐騙、假借冠名、易生誤解、影音變造、移花接木等內容。
2. **部分錯誤**：傳言內容為真實，但為片面事實，或者傳言內容確有其事，但脈絡、背景有誤，或者影片或照片包含真實影像，但摻雜不正確的背景、事件脈絡等。
3. **正確**：文章所傳遞的主要內容為真實。
4. **證據不足**：文章所傳遞的主要內容無法判斷真偽。
</definitions>

<output_format>
請只輸出查核結果摘要，用100-200個字說明，包含：使用者要查核的內容摘要、查核結果、主要證據(e.g. 相關單位的說明、CNA報導中的內容...)。

格式要求：
1. 明確註明查核標籤：「查核結果: [錯誤/部分錯誤/正確/證據不足]」
2. 純文字摘要，不要有任何標題、分點說明或特殊格式

注意：不要產生完整的查核報告格式，也不用在結尾標註資料來源，只要摘要即可。
</output_format>

<draft_example>
**查核結果: 錯誤**\n\n
高雄興達電廠火災意外，引發全國關注，台電啟用多部緊備機組發電。網路流傳，「電不夠時，靠核二核三當救援投手」，並附核電廠發電量擷圖佐證。經查，根據台電官網和相關報導，發電的是核電廠內的輕油氣渦輪機組，並非核電。台電官網清楚註明為核電廠內的「輕油」機組，但傳言擷取片段資訊，誤導稱為核電廠發電。台電發言人表示，因興達電廠意外啟用輕油氣渦輪機組發電，這是電力調度措施，與核電無關。核工學者也證實氣渦輪發電機雖設於核電廠內，但屬於獨立發電系統。此外，核電廠重新開機需要複雜程序和2天時間，無法作為緊急電力調度。因此傳言內容為引人誤解的訊息。
</draft_example>
"""
    explain_agent = Agent(
        name="explain_agent",
        instructions=prompt,
        handoffs =[],
    )
    
    _input = f"使用者要查核的內容: {user_input}\n查核點: {check_points}\n證據資料: {resources}\n提問: {question}"

    response = Runner.run_streamed(explain_agent, _input)

    full_text = ""
    async for event in response.stream_events():
        # 仍可逐 token 顯示
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)
            full_text += event.data.delta

    print("\n" + "="*50)

    explanation_text = response.final_output  # 純文本草稿
    try:
        usage = response.context_wrapper.usage
        print(f"[Token] 輸入：{usage.input_tokens}，輸出：{usage.output_tokens}")
    except AttributeError:
        pass

    return explanation_text

async def generate_explanation_streaming(user_input, check_points, resources, question: str = ""):
    """
    Streaming版本的generate_explanation，用於Streamlit的st.write_stream
    """
    prompt = f"""{RECOMMENDED_PROMPT_PREFIX}
<role>
你是台灣的事實查核專家，擅長透過資料查證消息。
</role>

<input_description>
1. 使用者要查核的內容：使用者提問的問題或輸入的文章，以此為中心進行事實查核。
2. 查核點：針對文章可能需要再做查證的疑點。
3. 證據資料：包含中央社新聞(CNA)跟查核中心報告(TFC)，可以作為查核的證據。查核中心報告中的label是經由專家審核的結果，包含錯誤、部分錯誤、正確、事實釐清和證據不足。
4. 提問：評估者提出的問題，用於評估解釋的品質。
</input_description>

<workflow>
1. **根據證據資料查核**：從證據資料中找出能夠回答查核點相關的證據。證據可以做為支持或反對的依據，並進一步判斷使用者要查核內容的真偽。
2. **生成查核結果說明**：結果要包含tag跟explanation。
3. **根據提問優化解釋**：如果有提問，請根據提問優化解釋，並更新tag跟explanation。
</workflow>

<fact_check_steps>
事實查核的步驟：
1. **逐項檢查查核點**：從證據資料裡面找到可以支持或反駁查核點的內容。
   - 請注意證據資料的日期(date)，以最新的資料為主。今天的日期是：{today}
   - 詳細的內容要看證據資料的內文(article)，請仔細思考相關人事物與事件的相關性
2. **撰寫查核結果說明**：結果要包含tag跟explanation。
   - tag：錯誤/部分錯誤/正確/證據不足。可以參考definitions判斷。
   - explanation：查核結果說明，必須以證據資料佐證。寫法格式參考format。
</fact_check_steps>

<definitions>
錯誤標籤的定義：
1. **錯誤**：文章所傳遞的主要內容錯誤不實。例如明顯捏造虛構的傳言，刻意錯置時間或地點的訊息，變造或挪用影片、照片於不相干的事件，假借冠名的言論，過時訊息等。包括詐騙、假借冠名、易生誤解、影音變造、移花接木等內容。
2. **部分錯誤**：傳言內容為真實，但為片面事實，或者傳言內容確有其事，但脈絡、背景有誤，或者影片或照片包含真實影像，但摻雜不正確的背景、事件脈絡等。
3. **正確**：文章所傳遞的主要內容為真實。
4. **證據不足**：文章所傳遞的主要內容無法判斷真偽。
</definitions>

<output_format>
請只輸出查核結果摘要，用100-200個字說明，包含：使用者要查核的內容摘要、查核結果、主要證據(e.g. 相關單位的說明、CNA報導中的內容...)。

格式要求：
1. 明確註明查核標籤：「查核結果: [錯誤/部分錯誤/正確/證據不足]」
2. 純文字摘要，不要有任何標題、分點說明或特殊格式

注意：不要產生完整的查核報告格式，也不用在結尾標註資料來源，只要摘要即可。
</output_format>

<draft_example>
**查核結果: 錯誤**\n\n
高雄興達電廠火災意外，引發全國關注，台電啟用多部緊備機組發電。網路流傳，「電不夠時，靠核二核三當救援投手」，並附核電廠發電量擷圖佐證。經查，根據台電官網和相關報導，發電的是核電廠內的輕油氣渦輪機組，並非核電。台電官網清楚註明為核電廠內的「輕油」機組，但傳言擷取片段資訊，誤導稱為核電廠發電。台電發言人表示，因興達電廠意外啟用輕油氣渦輪機組發電，這是電力調度措施，與核電無關。核工學者也證實氣渦輪發電機雖設於核電廠內，但屬於獨立發電系統。此外，核電廠重新開機需要複雜程序和2天時間，無法作為緊急電力調度。因此傳言內容為引人誤解的訊息。
</draft_example>
"""
    explain_agent = Agent(
        name="explain_agent",
        instructions=prompt,
        handoffs=[],
    )

    _input = f"使用者要查核的內容: {user_input}\n查核點: {check_points}\n證據資料: {resources}\n提問: {question}"

    response = Runner.run_streamed(explain_agent, _input)

    # 逐個yield streaming內容
    full_response = ""
    async for event in response.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            full_response += event.data.delta
            yield event.data.delta
    
    print(f"[Callback] Explanation: \n{full_response[:100]}...\n\n")

    # 不需要在這裡print，因為Streamlit會處理顯示

async def final_report_agent(history: str, check_points: str, user_input: str, resources: str):
    final_report_agent = Agent(
        name="final_report_agent",
        model="gpt-4.1",
        instructions="""
<role>你是台灣的事實查核中心專家，擅長撰寫事實查核報告。</role>

<input_description>
1. history: 問答紀錄，包含每一輪的解釋和提問。
2. check_points: 查核點，列出要被查證的文章有哪些可能的疑點。
3. user_input: 使用者要查核的內容。
4. resources: 證據資料，包含中央社新聞(CNA)跟查核中心報告(TFC)，可以作為查核的證據。查核中心報告中的label是經由專家審核的結果，包含錯誤、部分錯誤、正確、事實釐清和證據不足。
</input_description>

<task>
你要負責根據history的問答紀錄，撰寫最終版本的查核結果報告。
最終報告應該包含：查核結果(tag)、查核結果說明(explanation)、資料來源(resources)。
- tag：錯誤/部分錯誤/正確/證據不足。請根據history當中的解釋來判斷。
- explanation：查核結果說明，必須以證據資料佐證。請再一次複查resources的內容，確保查核解釋的說法是正確的、完整的、符合證據資料內容。
- resources：撰寫explanation時，所使用的證據資料的url。請一定要列出正確的url。禁止超出resources的範圍。

請特別注意"提問"，這通常是使用者最疑惑的地方。必須針對提問做出相對應的解釋和回應。
如果resources當中的相關資料無法佐證，請老實說「證據資料不足，無法查證。」
</task>

<output_format>
按照以下結構輸出查核結果：
**查證結果：[tag]**

分段敘述查核結果，每段之間用空行分隔。建議寫1-3段。
每一段後面都要標註參考資料編號，以[1]、[2]、[3]來標註，如果有多筆請以","分隔。

參考資料：
[1]: URL1\n
[2]: URL2\n
[3]: URL3\n

## 格式規則
1. tag必須是：錯誤、部分錯誤、正確、證據不足其中之一
2. 每段explanation都必須加上參考資料編號
3. 參考資料編號必須與resources中的URL對應
4. 不可使用resources中沒有的URL
</output_format>

<output_example>
**查證結果：錯誤**

傳言提到的時速 80 公里行駛於限速 40 公里的匝道，屬於「嚴重超速」，罰則內容大致無誤，不過國道警察於 2025 年 7 月 19 日在南投服務區取締的 4 件超速，皆非「嚴重超速」案例，且取締的地點是在南投服務區「內」的道路，不是高速公路或快速道路，因此是以《道交條例》第 40 條，處 1,200 元以上、2,400 元以下罰鍰，而非以第 33 條「高速公路或快速道路超速」（3,000 元以上、6,000 元以下）或第 43 條「嚴重超速」（6,000 元以上、36,000 元以下）開罰。[1]

國道警方於 7 月 19 日在南投服務區「內」取締 4 件超速，依《道交條例》第 40 條開罰 1,200 元以上、2,400 元以下罰鍰。[2]

報導指出，警方除了例假日編排路檢勤務，也在進入服務區主要車道分流處（匝道）及中寮便道匯流處加強超速取締，並依法設立「警52」標誌提醒駕駛人。警方於 7 月 19 日夜間，在南投服務區取締超速 4 件，依《道路交通管理處罰條例》第 40 條規定，處 1,200 元以上、2,400 元以下罰鍰。[1],[3]

參考資料：\n\n
[1]: https://www.cna.com.tw/news/aall/202509110109.aspx \n
[2]: https://www.cna.com.tw/news/aall/202509090405.aspx \n
[3]: https://www.cna.com.tw/news/aall/202508300207.aspx \n
</output example>
"""
    )

    input_text = f"history: {history}\ncheck_points: {check_points}\nuser_input: {user_input}\nresources: {resources}"

    final_report = Runner.run_streamed(final_report_agent, input_text)
    
    async for event in final_report.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)
    print("\n" + "="*50)

    try:
        usage = final_report.context_wrapper.usage
        print(f"[Token] 輸入：{usage.input_tokens}，輸出：{usage.output_tokens}")
    except AttributeError:
        pass

    return final_report.final_output

async def final_report_agent_streaming(history: str, check_points: str, user_input: str, resources: str):
    """
    Streaming版本的final_report_agent，用於Streamlit的st.write_stream
    """
    final_report_agent = Agent(
        name="final_report_agent",
        model="gpt-4.1",
        instructions="""
<role>你是台灣的事實查核中心專家，擅長撰寫事實查核報告。</role>

<input_description>
1. history: 問答紀錄，包含每一輪的解釋和提問。
2. check_points: 查核點，列出要被查證的文章有哪些可能的疑點。
3. user_input: 使用者要查核的內容。
4. resources: 證據資料，包含中央社新聞(CNA)跟查核中心報告(TFC)，可以作為查核的證據。查核中心報告中的label是經由專家審核的結果，包含錯誤、部分錯誤、正確、事實釐清和證據不足。
</input_description>

<task>
你要負責根據history的問答紀錄，撰寫最終版本的查核結果報告。
最終報告應該包含：查核結果(tag)、查核結果說明(explanation)、資料來源(resources)。
- tag：錯誤/部分錯誤/正確/證據不足。請根據history當中的解釋來判斷。
- explanation：查核結果說明，必須以證據資料佐證。請再一次複查resources的內容，確保查核解釋的說法是正確的、完整的、符合證據資料內容。
- resources：撰寫explanation時，所使用的證據資料的url。請一定要列出正確的url。禁止超出resources的範圍。

請特別注意"提問"，這通常是使用者最疑惑的地方。必須針對提問做出相對應的解釋和回應。
如果resources當中的相關資料無法佐證，請老實說「證據資料不足，無法查證。」
</task>

<output_format>
按照以下結構輸出查核結果：
**查證結果：[tag]**

分段敘述查核結果，每段之間用空行分隔。建議寫1-3段。
每一段後面都要標註參考資料編號，以[1]、[2]、[3]來標註，如果有多筆請以","分隔。

參考資料：
[1]: URL1
[2]: URL2
[3]: URL3

## 格式規則
1. tag必須是：錯誤、部分錯誤、正確、證據不足其中之一
2. 每段explanation都必須加上參考資料編號
3. 參考資料編號必須與resources中的URL對應
4. 不可使用resources中沒有的URL
</output_format>

<output_example>
**查證結果：錯誤**

傳言提到的時速 80 公里行駛於限速 40 公里的匝道，屬於「嚴重超速」，罰則內容大致無誤，不過國道警察於 2025 年 7 月 19 日在南投服務區取締的 4 件超速，皆非「嚴重超速」案例，且取締的地點是在南投服務區「內」的道路，不是高速公路或快速道路，因此是以《道交條例》第 40 條，處 1,200 元以上、2,400 元以下罰鍰，而非以第 33 條「高速公路或快速道路超速」（3,000 元以上、6,000 元以下）或第 43 條「嚴重超速」（6,000 元以上、36,000 元以下）開罰。[1]

國道警方於 7 月 19 日在南投服務區「內」取締 4 件超速，依《道交條例》第 40 條開罰 1,200 元以上、2,400 元以下罰鍰。[2]

報導指出，警方除了例假日編排路檢勤務，也在進入服務區主要車道分流處（匝道）及中寮便道匯流處加強超速取締，並依法設立「警52」標誌提醒駕駛人。警方於 7 月 19 日夜間，在南投服務區取締超速 4 件，依《道路交通管理處罰條例》第 40 條規定，處 1,200 元以上、2,400 元以下罰鍰。[1],[3]

參考資料：

[1]: https://www.cna.com.tw/news/aall/202509110109.aspx
[2]: https://www.cna.com.tw/news/aall/202509090405.aspx
[3]: https://www.cna.com.tw/news/aall/202508300207.aspx
</output example>
"""
    )

    input_text = f"history: {history}\ncheck_points: {check_points}\nuser_input: {user_input}\nresources: {resources}"

    final_report = Runner.run_streamed(final_report_agent, input_text)

    # 逐個yield streaming內容
    full_response = ""
    async for event in final_report.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            full_response += event.data.delta
            yield event.data.delta
    
    print(f"[Callback] Final Report: \n{full_response}...\n\n")

async def run_question_review(draft_report: str, check_points: str):
    review_input = f"draft_report: {draft_report}\ncheck_points: {check_points}"
    result = Runner.run_streamed(questioners_agent, review_input)

    # 不需要逐 token 時，可以只監聽語義事件或直接拿 final
    async for _ in result.stream_events():
        pass

    eval_obj: QAEval = result.final_output  # 已是 QAEval
    return eval_obj

async def run_interactive_fact_check(user_input, check_points, resources, max_rounds: int = 3):
    """
    互動式事實查核流程
    用戶可以選擇AI建議的問題或自己輸入問題
    """
    history = []

    # 第一輪：生成初步查核結果
    print(f"\n{'='*60}")
    print("🔍 生成初步查核結果")
    print(f"{'='*60}")

    draft = await generate_explanation(
        user_input=user_input,
        check_points=check_points,
        resources=resources,
        question=""
    )

    # 互動輪次
    for round_num in range(1, 4):  # 第1-3輪
        print(f"\n{'='*60}")
        print(f"💬 第{round_num}輪: 評估與提問")
        print(f"{'='*60}")

        # AI評估當前草稿
        print("\n🤖 AI正在評估當前查核結果...")
        eval_result = await run_question_review(draft, check_points)

        # 顯示評估結果
        print(f"\n📊 AI評估結果：")
        print(f"   • 說服力: {eval_result.persuasiveness}/5")
        print(f"   • 邏輯正確性: {eval_result.logical_correctness}/5")
        print(f"   • 完整性: {eval_result.completeness}/5")
        print(f"   • 簡潔性: {eval_result.conciseness}/5")
        print(f"   • 認同度: {eval_result.agreement}/5")
        print(f"   • 平均分數: {eval_result.average}/5")
        print(f"   • 最弱面向: {eval_result.weakest_aspect}")

        # 檢查是否分數已經很高
        if eval_result.average >= 4.5:
            print(f"\n✨ AI評估分數已經很高 ({eval_result.average}/5)，建議考慮直接生成最終報告")

        # 提供選擇
        print(f"\n❓ AI建議的改進問題：")
        print(f"   {eval_result.improvement_question}")

        print(f"\n請選擇下一步動作：")
        print(f"1. 採用AI建議的問題")
        print(f"2. 自行輸入問題")
        print(f"3. 結束互動，生成最終報告")

        while True:
            choice = input("\n請輸入選擇 (1/2/3): ").strip()

            if choice == "1":
                # 採用AI建議
                user_question = eval_result.improvement_question
                print(f"✅ 已選擇AI建議: {user_question}")
                break
            elif choice == "2":
                # 用戶自行輸入
                user_question = input("\n請輸入您的問題: ").strip()
                if user_question:
                    print(f"✅ 已收到您的問題: {user_question}")
                    break
                else:
                    print("❌ 問題不能為空，請重新輸入")
            elif choice == "3":
                # 結束互動
                print("✅ 結束互動，準備生成最終報告...")
                user_question = None
                break
            else:
                print("❌ 無效選擇，請輸入 1、2 或 3")

        # 如果選擇結束，跳出循環
        if user_question is None:
            break

        # 記錄歷史
        history.append({
            "round": round_num,
            "ai_evaluation": eval_result,
            "user_question": user_question,
            "question_source": "AI建議" if choice == "1" else "用戶輸入",
            "explanation": draft  # 記錄當前解釋
        })

        # 根據問題重新生成查核結果
        print(f"\n🔄 根據問題重新生成查核結果...")
        draft = await generate_explanation(
            user_input=user_input,
            check_points=check_points,
            resources=resources,
            question=user_question
        )

    # 生成最終報告
    print(f"\n{'='*60}")
    print("📋 生成最終查核報告")
    print(f"{'='*60}")

    # 收集完整的歷史記錄給 final_report_agent
    full_history = f"初步解釋: {draft}\n\n"
    for interaction in history:
        full_history += f"=== 輪次{interaction['round']} ===\n"
        full_history += f"解釋: {interaction.get('explanation', '')}\n"
        full_history += f"提問: {interaction['user_question']}\n"
        full_history += f"AI評分: {interaction['ai_evaluation'].average}/5 (最弱面向: {interaction['ai_evaluation'].weakest_aspect})\n\n"

    # 使用專門的 final_report_agent 生成最終報告
    final_report = await final_report_agent(
            history=full_history,
            check_points=check_points,
            user_input=user_input,
            resources=resources
        )


    return {
        "final_report": final_report,
        "interaction_history": history,
        "total_rounds": len(history) + 1
    }

if __name__ == "__main__":
    user_input = "馬英九當總統的時候有簽訂ECFA，賴清德上街頭抗議，真的嗎？"

    check_points_data = get_check_points(user_input, media_name="Chiming")
    if check_points_data["Result"] == "Y":
        check_points = check_points_data["ResultData"]["check_points"]
    else:
        check_points = None
        print("[Info] 查核點 API 失敗")

    resources = es_resources(user_input)

    # print(">>> 社稿+事實查核中心報告：")
    # print(resources)

    async def main():
        result = await run_interactive_fact_check(
            user_input=user_input,
            check_points=check_points,
            resources=resources,
            max_rounds=3
        )

        print(f"\n" + "="*60)
        print(f"總互動輪數: {result['total_rounds']}")

    asyncio.run(main())



