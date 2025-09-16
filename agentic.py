"""
查核對話機器人

1. 查核點
2. elastic search 搜查核中心報告跟社稿 (用embedding搜)
3. 解釋查核點 -> 評估與提問機器人 -> 重新生成解釋 (最多重複5次) -> 最終寫報告
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
async def generate_explanation(user_input, check_points, resources,  question: str = ""):

    prompt = f'''{RECOMMENDED_PROMPT_PREFIX}
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

<output>
請只輸出查核結果摘要，用200-300個字說明，包含：使用者要查核的內容摘要、查核結果、主要證據(e.g. 相關單位的說明、CNA報導中的內容...)。

格式要求：
1. 明確註明查核標籤：「查核結果: [錯誤/部分錯誤/正確/證據不足]」
2. 純文字摘要，不要有任何標題、分點說明或特殊格式

範例格式：

查核結果: 錯誤
高雄興達電廠火災意外，引發全國關注，台電啟用多部緊備機組發電。網路流傳，「電不夠時，靠核二核三當救援投手」，並附核電廠發電量擷圖佐證。經查，根據台電官網和相關報導，發電的是核電廠內的輕油氣渦輪機組，並非核電。台電官網清楚註明為核電廠內的「輕油」機組，但傳言擷取片段資訊，誤導稱為核電廠發電。台電發言人表示，因興達電廠意外啟用輕油氣渦輪機組發電，這是電力調度措施，與核電無關。核工學者也證實氣渦輪發電機雖設於核電廠內，但屬於獨立發電系統。此外，核電廠重新開機需要複雜程序和2天時間，無法作為緊急電力調度。因此傳言內容為引人誤解的訊息。

注意：不要產生完整的查核報告格式，也不用在結尾標註資料來源，只要摘要即可。
</output>
'''
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

async def final_report_agent(draft_report: str, check_points: str, user_input: str, resources: str):
    final_report_agent = Agent(
        name="final_report_agent",
        model="gpt-4.1",
        instructions="""
        <role>你是台灣的事實查核中心專家，擅長撰寫事實查核報告。</role>

        <task>你要負責根據draft_report跟check_points，撰寫查核結果報告。</task>
        
        <input_description>
        1. draft_report: 事實查核的解釋，包含tag跟explanation。
        2. check_points: 查核點，列出要被查證的文章有哪些可能的疑點。
        3. user_input: 使用者要查核的內容。
        4. resources: 證據資料，包含中央社新聞(CNA)跟查核中心報告(TFC)，可以作為查核的證據。查核中心報告中的label是經由專家審核的結果，包含錯誤、部分錯誤、正確、事實釐清和證據不足。
        </input_description>

        <output>按照以下格式輸出查核結果：

# [標籤]查核報告標題
### 背景說明加查核結果摘要，總共50-100個字

一、第一個查核點標題
針對第一個查核點的詳細解釋說明 (source [1])

二、第二個查核點標題
針對第二個查核點的詳細解釋說明 (source [2])

### 一句話總結整個查核結果

### 資料來源
[{{'1': '第一個證據資料的完整URL'}}, {{'2': '第二個證據資料的完整URL'}}]

格式說明：
- 標籤必須是：錯誤、部分錯誤、正確、證據不足其中之一
- 背景說明需簡潔扼要，50-100字內
- 每個查核點都要有對應的證據來源編號
- 資料來源只列出實際使用的證據URL，按編號順序排列
</output_format>

<example>
# [錯誤]網傳「興達電廠起火，電不夠靠核二核三救援」？
### 高雄興達電廠火災意外，引發全國關注，台電啟用多部緊備機組發電。網路流傳，「電不夠時，靠核二核三當救援投手」，並附核電廠發電量擷圖佐證。經查，發電的是核電廠內的輕油氣渦輪機組，並非核電。

一、網傳「核電廠發電量」擷圖來自台電官網，但實際為核二、核三廠內的氣渦輪機組，以輕柴油為燃料，與核電無關。台電官網清楚註明為核電廠內的「輕油」機組，但傳言擷取片段資訊，誤導稱為核電廠發電。[1]

二、台電發言人表示，近日因興達電廠意外，因此啟用核二、核三的輕油氣渦輪機組發電，過去電力吃緊時也曾如此，是電力調度措施。[1][2]

三、核工學者表示，核電廠內設置氣渦輪發電機，除了提供廠區緊急電源，也能連結外部電網，提供全台電力，但這跟核能發電無關。[2]

四、核電廠已經除役，要重新發電需先安檢、取得核安會許可、試運轉等多道程序，無法臨時發電。而且重新開機需2天才能供給電力，難以當成緊急電力調度使用，因此網傳「核電廠發電救援」的說法無法成立。[2]

### 傳言擷取台電官網資訊，聲稱「電不夠時，靠核二核三救援」；但實際是輕油氣渦輪機發電，並非核電，為引人誤解的訊息。
### 參考資料：[{{'1': 'https://www.cna.com.tw/...'}}, {{'2': 'https://www.cna.com.tw/news/...'}}]
</example>
        """
    )

    input_text = f"draft_report: {draft_report}\ncheck_points: {check_points}\nuser_input: {user_input}\nresources: {resources}"

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
        question="",
        enable_handoff=False,
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
            "question_source": "AI建議" if choice == "1" else "用戶輸入"
        })

        # 根據問題重新生成查核結果
        print(f"\n🔄 根據問題重新生成查核結果...")
        draft = await generate_explanation(
            user_input=user_input,
            check_points=check_points,
            resources=resources,
            question=user_question,
            enable_handoff=False,
        )

    # 生成最終報告
    print(f"\n{'='*60}")
    print("📋 生成最終查核報告")
    print(f"{'='*60}")

    final_report = await final_report_agent(draft, check_points, user_input, resources)

    return {
        "final_report": final_report,
        "interaction_history": history,
        "total_rounds": len(history) + 1
    }

if __name__ == "__main__":
    user_input = "日本首相石破茂宣布辭去自民黨總裁（黨主席）職務後，日圓聞訊走貶，加上投資人預期新政府將推出新的經濟對策，日股今天應聲上漲。"

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

        print("\n" + "="*60)
        print("📄 最終查核報告")
        print("="*60)
        print(result["final_report"])

        print(f"\n" + "="*60)
        print(f"總互動輪數: {result['total_rounds']}")

        # for i, interaction in enumerate(result["interaction_history"], 1):
        #     print(f"\n第{interaction['round']}輪互動:")
        #     print(f"  • 問題來源: {interaction['question_source']}")
        #     print(f"  • 用戶問題: {interaction['user_question']}")
        #     print(f"  • AI評分: {interaction['ai_evaluation'].average}/5")
        #     print(f"  • 最弱面向: {interaction['ai_evaluation'].weakest_aspect}")

    asyncio.run(main())



