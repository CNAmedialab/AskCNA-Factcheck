"""
查核對話機器人

1. 查核點
2. elastic search 搜查核中心報告跟社稿 (用embedding搜)
3. 解釋查核點 -> 評估與提問機器人 -> 重新生成解釋 (最多重複5次) -> 最終寫報告
"""
from agents import Agent, Runner
from pydantic import BaseModel
import asyncio
from openai.types.responses import ResponseTextDeltaEvent
from functions import get_check_points, es_resources
from dotenv import load_dotenv

load_dotenv()

# 拿到check points, resources 之後產生查核初稿
async def generate_exlplanation(question, check_points, resources):
    class Explanation(BaseModel):
        tag: str
        explanation: str

    prompt = """
    <role>你是台灣的事實查核專家，擅長透過資料查證消息。</role>

    <input description>
    1. <使用者提問> 是使用者提問的問題，以<使用者提問>為中心，進行事實查核。
    2. <查核點> 是針對文章可能需要再做查證的疑點。
    3. <證據資料>包含中央社新聞(CNA)跟查核中心報告(TFC)，可以作為你查核的證據。查核中心報告終的<label>是經由專家審核的結果，包含錯誤、部分錯誤、正確、事實釐清和證據不足。
    </input description>

    <fake definition>
    錯誤的定義可以分為兩種：
    1. 錯誤：文章所傳遞的主要內容錯誤不實。例如明顯捏造虛構的傳言，刻意錯置時間或地點的訊息，變造或挪用影片、照片於不相干的事件，假借冠名的言論，過時訊息等。包括詐騙、假借冠名、易生誤解、影音變造、移花接木等內容。
    2. 部分錯誤：傳言內容為真實，但為片面事實，或者傳言內容確有其事，但脈絡、背景有誤，或者影片或照片包含真實影像，但摻雜不正確的背景、事件脈絡等。
    3. 正確：文章所傳遞的主要內容為真實。
    4. 證據不足：文章所傳遞的主要內容無法判斷真偽。
    </fake definition>

    <task>
    1. 請你從<證據資料>中找出能夠回答<查核點>相關的證據。證據可以做為支持或反對的依據，並進一步判斷<使用者提問>的真偽。
    2. 最終回答要包含{{tag}}跟{{explanation}}。{{tag}}可以參考<fake definition>來判斷。{{explanation}}必須以<證據資料佐證>。
    </task>

    <output format>
    {
        "tag": "錯誤/部分錯誤/正確/證據不足",
        "explanation": "事實查核判斷的說明，必須以<證據資料佐證>"
    }
    </output format>
    """
    explain_agent = Agent(
        name="explain_agent",
        description="透過查核點跟參考資料生成一個針對使用者提問的事實查核解釋",
        instructions=prompt,
        output_model=Explanation,
    )
    input = f"使用者提問: {question}\n查核點: {check_points}\n證據資料: {resources}"

    response = Runner.run_streamed(explain_agent, input)
    async for event in response.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)

    return response


if __name__ == "__main__":
    question = "日本首相石破茂宣布辭去自民黨總裁（黨主席）職務後，日圓聞訊走貶，加上投資人預期新政府將推出新的經濟對策，日股今天應聲上漲。"

    check_points = get_check_points(question, media_name="Chiming")
    if check_points["Result"] == "Y":
        check_points = check_points["ResultData"]["check_points"]
    else:
        check_points = None
        print("[Info] 查核點 API 失敗")
    resources = es_resources(question)
    
    asyncio.run(generate_exlplanation(question, check_points, resources))


