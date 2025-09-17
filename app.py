"""
事實查核對話機器人 - Streamlit版本
直接調用現有的functions和agentic模組
"""
import streamlit as st
import asyncio
from functions import get_check_points, es_resources
from agentic import (
    generate_explanation,
    generate_explanation_streaming,
    run_question_review,
    final_report_agent,
    final_report_agent_streaming
)
from datetime import datetime

def run_async_sync(coroutine):
    """同步執行異步函數的輔助函數"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果有活動的事件循環，創建一個新的
            import threading
            import concurrent.futures

            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coroutine)
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
        else:
            return loop.run_until_complete(coroutine)
    except RuntimeError:
        # 沒有事件循環，創建一個新的
        return asyncio.run(coroutine)

def create_streaming_generator_with_result(async_streaming_func, *args, **kwargs):
    """創建同步的 generator 來包裝異步 streaming 函數，並返回完整文本"""
    import threading
    import queue
    import concurrent.futures

    result_queue = queue.Queue()
    full_text_ref = {"text": ""}  # 使用字典來保持引用

    def run_async_streaming():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            async def collect_chunks():
                async for chunk in async_streaming_func(*args, **kwargs):
                    full_text_ref["text"] += chunk
                    result_queue.put(chunk)
                result_queue.put(None)  # 結束信號

            new_loop.run_until_complete(collect_chunks())
        finally:
            new_loop.close()

    # 在背景線程中運行異步函數
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_async_streaming)

        def generator():
            while True:
                try:
                    chunk = result_queue.get(timeout=30)  # 30秒超時
                    if chunk is None:  # 結束信號
                        break
                    yield chunk
                except queue.Empty:
                    break

            # 等待線程完成
            future.result()

        return generator(), full_text_ref

def create_streaming_generator(async_streaming_func, *args, **kwargs):
    """創建同步的 generator 來包裝異步 streaming 函數"""
    generator, _ = create_streaming_generator_with_result(async_streaming_func, *args, **kwargs)
    return generator

class StreamlitFactCheckBot:
    def __init__(self):
        pass

    def reset_fact_check_state(self):
        """重置事實查核狀態"""
        st.session_state.fact_check_state = None
        st.session_state.current_draft = None
        st.session_state.check_points = None
        st.session_state.resources = None
        st.session_state.user_input = None
        st.session_state.round_num = 1
        st.session_state.history = []
        st.session_state.messages = []
        st.session_state.ai_suggested_question = None


    def start_fact_check(self, user_input: str, media_name: str = "Chiming"):
        """開始事實查核流程"""
        # 重置狀態（但保留messages）
        st.session_state.fact_check_state = "starting"
        st.session_state.current_draft = None
        st.session_state.check_points = None
        st.session_state.resources = None
        st.session_state.user_input = user_input
        st.session_state.round_num = 1
        st.session_state.history = []
        st.session_state.ai_suggested_question = None

        print(f"====> 開始對話：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 步驟1: 分析查核點
        with st.spinner("🔍 正在分析查核點..."):
            check_points_data = get_check_points(user_input, media_name)
            if check_points_data["Result"] == "Y":
                check_points = check_points_data["ResultData"]["check_points"]
                st.session_state.check_points = check_points
                with st.chat_message("assistant"):
                    st.markdown("**查核點分析完成：**")
                    if isinstance(check_points, list):
                        # 檢查查核點是否已經有編號，如果有就直接顯示，沒有就加上編號
                        formatted_points = []
                        for i, cp in enumerate(check_points):
                            cp_str = str(cp).strip()
                            # 檢查是否已經以數字開頭（已有編號）
                            if cp_str and cp_str[0].isdigit() and '. ' in cp_str[:5]:
                                formatted_points.append(cp_str)
                            else:
                                formatted_points.append(f"{i+1}. {cp_str}")
                        check_points_text = "\n".join(formatted_points)
                        st.markdown(f"\n{check_points_text}")
                    else:
                        check_points_text = f"{check_points}"
                        st.markdown(f"\n{check_points_text}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"**查核點分析完成:**\n\n{check_points_text}"
                })
            else:
                check_points = None
                st.session_state.check_points = None
                with st.chat_message("assistant"):
                    st.warning("⚠️ 查核點 API 失敗，將使用原始文本進行查核")

                # 記錄查核點失敗訊息
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "⚠️ 查核點 API 失敗，將使用原始文本進行查核"
                })

        # 步驟2: 搜索相關資源
        with st.spinner("📚 正在搜索相關證據資料..."):
            resources = es_resources(user_input)
            st.session_state.resources = resources
            with st.chat_message("assistant"):
                st.markdown(f"**一共找到{len(resources)}筆證據資料**")

        # 記錄證據搜索結果
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"一共找到{len(resources)}筆證據資料"
        })

        # 步驟3: 生成初步查核結果
        with st.chat_message("assistant"):
            with st.spinner("💡 正在生成初步查核結果..."):
                def explanation_generator():
                    return create_streaming_generator(
                        generate_explanation_streaming,
                        user_input, check_points, resources, ""
                    )

                st.markdown("**初步查核結果**")
                draft = st.write_stream(explanation_generator())

        st.session_state.current_draft = draft

        # 記錄查核解釋結果
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"{draft}"
        })

        # 步驟4: AI評估
        # 預先顯示AI評估開始訊息
        with st.chat_message("assistant"):
            with st.spinner("👁️‍🗨️ 正在評估查核結果..."):
                eval_result = run_async_sync(run_question_review(draft, check_points))

                eval_result_content = f"""📊 **AI評估結果：**\n\n
•  說服力: {eval_result.persuasiveness}/5\n
•  邏輯正確性: {eval_result.logical_correctness}/5\n
•  完整性: {eval_result.completeness}/5\n
•  簡潔性: {eval_result.conciseness}/5\n
•  認同度: {eval_result.agreement}/5\n
•  **平均分數: {eval_result.average}/5**\n
•  **最弱面向**: {eval_result.weakest_aspect}"""
                st.markdown(eval_result_content)

        # 記錄AI評估結果
        st.session_state.messages.append({
            "role": "assistant",
            "content": eval_result_content
        })

        # 第二個訊息：AI建議的問題
        with st.chat_message("assistant"):
            ai_question_content = f"""❓ **AI建議的改進問題：**
{eval_result.improvement_question}"""
            st.markdown(ai_question_content)

        # 記錄AI建議問題
        st.session_state.messages.append({
            "role": "assistant",
            "content": ai_question_content
        })

        # 第三個訊息：選擇下一步動作
        with st.chat_message("assistant"):
            next_step_content = """**請選擇下一步動作：**"""
            st.markdown(next_step_content)

        # 記錄選擇選項
        st.session_state.messages.append({
            "role": "assistant",
            "content": next_step_content
        })

        # 記錄評估歷史
        st.session_state.history.append({
            "round": st.session_state.round_num,
            "explanation": draft,
            "evaluation": eval_result,
            "question": "",
            "ai_evaluation": eval_result,
            "user_question": eval_result.improvement_question,
            "question_source": "AI自動建議"
        })
        st.session_state.round_num += 1

        st.session_state.fact_check_state = "waiting_user_choice"
        st.session_state.ai_suggested_question = eval_result.improvement_question

    def handle_user_choice(self, choice: str):
        """處理用戶選擇"""
        choice = choice.strip()

        if choice == "1":
            # 採用AI建議的問題
            question = st.session_state.ai_suggested_question
            self.continue_with_question(question, "AI建議")

        elif choice == "2":
            # 自行輸入問題
            with st.chat_message("assistant"):
                st.markdown("請輸入您的問題：")
            st.session_state.messages.append({"role": "assistant", "content": "請輸入您的問題："})
            st.session_state.fact_check_state = "waiting_custom_question"

        elif choice == "3":
            # 生成最終報告
            self.generate_final_report()

    def apply_improvement(self, improvement_question: str):
        """應用改善問題並重新生成"""
        if st.session_state.fact_check_state != "waiting_for_improvement_choice":
            return

        # 根據改善問題重新生成查核結果
        with st.chat_message("assistant"):
            st.markdown(f"✨ **第{st.session_state.round_num+1}輪改善結果**")

            def improvement_generator():
                return create_streaming_generator(
                    generate_explanation_streaming,
                    st.session_state.user_input,
                    st.session_state.check_points,
                    st.session_state.resources,
                    improvement_question
                )

            st.markdown(f"**第{st.session_state.round_num-1}輪對話結果**")
            draft = st.write_stream(improvement_generator())

        st.session_state.current_draft = draft
        st.session_state.round_num += 1

        # 將改善結果記錄到messages
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"**第{st.session_state.round_num-1}輪對話結果**\n\n{draft}"
        })

        # 檢查是否已達最大輪數
        if st.session_state.round_num > 3:
            st.session_state.fact_check_state = "ready_for_final_report"
            with st.chat_message("assistant"):
                st.markdown("⏱️ 已達提問上限次數，生成最終報告") 
            st.session_state.messages.append({
                "role": "assistant",
                "content": "⏱️ 已達提問上限次數，生成最終報告"
            })
        else:
            # 自動執行AI評估
            eval_result = run_async_sync(run_question_review(draft, st.session_state.check_points))

            # 顯示AI評估結果
            with st.chat_message("assistant"):
                st.markdown(f"🔍 **第{st.session_state.round_num}輪 AI評估結果**")
                st.markdown(f"**整體評分:** {eval_result.average}/5")
                st.markdown(f"**最弱面向:** {eval_result.weakest_aspect}")
                st.markdown(f"**改善問題:** {eval_result.improvement_question}")

            # 將AI評估結果記錄到messages
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"🔍 **第{st.session_state.round_num}輪 AI評估結果**\n\n**整體評分:** {eval_result.average}/5\n**最弱面向:** {eval_result.weakest_aspect}\n**改善問題:** {eval_result.improvement_question}"
            })

            # 記錄評估歷史
            st.session_state.history.append({
                "round": st.session_state.round_num,
                "ai_evaluation": eval_result,
                "user_question": eval_result.improvement_question,
                "question_source": "AI自動建議"
            })

            # 檢查是否品質已達標
            if eval_result.average >= 4.0:
                with st.chat_message("assistant"):
                    st.markdown(f"🎯 **品質達標** - 第{st.session_state.round_num}輪評估分數已達{eval_result.average}/5，品質良好")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"🎯 **品質達標** - 第{st.session_state.round_num}輪評估分數已達{eval_result.average}/5，品質良好"
                })
                st.session_state.fact_check_state = "ready_for_final_report"
            else:
                # 分數低於4，顯示user訊息引導選擇
                with st.chat_message("user"):
                    st.markdown("請選擇下一步：採用AI建議的改善問題、自行輸入改善問題、或直接生成最終報告？")

                st.session_state.messages.append({
                    "role": "user",
                    "content": "請選擇下一步：採用AI建議的改善問題、自行輸入改善問題、或直接生成最終報告？"
                })

                st.session_state.fact_check_state = "waiting_for_improvement_choice"
                st.session_state.ai_suggested_question = eval_result.improvement_question

    def continue_with_question(self, question: str, source: str):
        """繼續查核流程處理問題"""
        with st.chat_message("assistant"):
            with st.spinner("💡 正在重新生成查核解釋..."):
                def regenerate_explanation_generator():
                    return create_streaming_generator(
                        generate_explanation_streaming,
                        st.session_state.user_input,
                        st.session_state.check_points,
                        st.session_state.resources,
                        question
                    )
                st.markdown(f"**第{st.session_state.round_num-1}輪對話結果**")
                draft = st.write_stream(regenerate_explanation_generator())

        # 記錄重新生成的查核解釋
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"**第{st.session_state.round_num-1}輪對話結果**\n\n{draft}"
        })

        st.session_state.current_draft = draft

        # 檢查是否已達最大輪數
        if st.session_state.round_num >= 3:
            with st.chat_message("assistant"):
                st.markdown("⏱️ **已達最大改善輪數**，準備生成最終報告")

            # 記錄最大輪數訊息
            st.session_state.messages.append({
                "role": "assistant",
                "content": "⏱️ **已達最大改善輪數**，準備生成最終報告"
            })

            self.generate_final_report()
        else:
            # 繼續 AI 評估
            # 預先顯示AI評估開始訊息
            with st.chat_message("assistant"):
                st.markdown("👁️‍🗨️ 正在評估查核結果...")

            # 記錄AI評估開始訊息
            st.session_state.messages.append({
                "role": "assistant",
                "content": "AI評估查核結果..."
            })

            with st.spinner("評估中..."):
                eval_result = run_async_sync(run_question_review(draft, st.session_state.check_points))

            # 第一個訊息：AI評估結果
            with st.chat_message("assistant"):
                eval_result_content = f"""📊 **AI評估結果：**
•  說服力: {eval_result.persuasiveness}/5\n
•  邏輯正確性: {eval_result.logical_correctness}/5\n
•  完整性: {eval_result.completeness}/5\n
•  簡潔性: {eval_result.conciseness}/5\n
•  認同度: {eval_result.agreement}/5\n
•  **平均分數: {eval_result.average}/5**\n
•  **最弱面向**: {eval_result.weakest_aspect}"""
                st.markdown(eval_result_content)

            # 記錄AI評估結果
            st.session_state.messages.append({
                "role": "assistant",
                "content": eval_result_content
            })

            # 第二個訊息：AI建議的問題
            with st.chat_message("assistant"):
                ai_question_content = f"""❓ **AI建議的改進問題：**
{eval_result.improvement_question}"""
                st.markdown(ai_question_content)

            # 記錄AI建議問題
            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_question_content
            })

            # 第三個訊息：選擇下一步動作
            with st.chat_message("assistant"):
                next_step_content = """**請選擇下一步動作：**"""
                st.markdown(next_step_content)

            # 記錄選擇選項
            st.session_state.messages.append({
                "role": "assistant",
                "content": next_step_content
            })

            # 記錄評估歷史
            st.session_state.history.append({
                "round": st.session_state.round_num,
                "explanation": draft,
                "evaluation": eval_result,
                "question": question,
                "question_source": source
            })
            st.session_state.round_num += 1

            st.session_state.fact_check_state = "waiting_user_choice"
            st.session_state.ai_suggested_question = eval_result.improvement_question

    def generate_final_report(self):
        """生成最終報告"""
        # 構建完整歷史
        full_history = f"初步解釋: {st.session_state.current_draft}\n\n"
        for interaction in st.session_state.history:
            full_history += f"=== 輪次{interaction['round']} ===\n"
            full_history += f"解釋: {interaction.get('explanation', '')}\n"
            if interaction.get('question'):
                full_history += f"提問: {interaction['question']}\n"
            full_history += f"AI評分: {interaction['evaluation'].average}/5 (最弱面向: {interaction['evaluation'].weakest_aspect})\n\n"

        # 生成最終報告
        with st.chat_message("assistant"):
            with st.spinner("📋 正在生成最終報告..."):
                generator, full_text_ref = create_streaming_generator_with_result(
                    final_report_agent_streaming,
                    full_history,
                    st.session_state.check_points,
                    st.session_state.user_input,
                    st.session_state.resources
                )

            # 顯示標題和 streaming 效果
            st.markdown("**最終查核報告**")
            st.write_stream(generator)

            # 獲取完整文本
            final_report = full_text_ref["text"]

        # 處理 Markdown 格式 - 轉義參考資料編號以避免解析問題
        def escape_references(text):
            import re
            # 將 [1], [2], [3] 等轉換為 \[1\], \[2\], \[3\] 以避免被誤認為列表
            return re.sub(r'\[(\d+)\]', r'\\[\1\\]', text)

        escaped_final_report = escape_references(final_report)

        # 記錄最終查核報告 - 使用轉義後的內容保持 Markdown 格式
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"**最終查核報告：**\n\n{escaped_final_report}"
        })

        with st.chat_message("assistant"):
            st.markdown(f"查核完畢！總共進行了 {st.session_state.round_num-1} 輪互動。")

        # 記錄流程完成訊息
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"查核完畢！總共進行了 {st.session_state.round_num-1} 輪互動。"
        })

        st.session_state.fact_check_state = "completed"


def init_session_state():
    """初始化 session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "fact_check_state" not in st.session_state:
        st.session_state.fact_check_state = None
    if "current_draft" not in st.session_state:
        st.session_state.current_draft = None
    if "check_points" not in st.session_state:
        st.session_state.check_points = None
    if "resources" not in st.session_state:
        st.session_state.resources = None
    if "user_input" not in st.session_state:
        st.session_state.user_input = None
    if "round_num" not in st.session_state:
        st.session_state.round_num = 1
    if "history" not in st.session_state:
        st.session_state.history = []
    if "ai_suggested_question" not in st.session_state:
        st.session_state.ai_suggested_question = None

def display_chat_message(role: str, content: str):
    """顯示聊天訊息"""
    with st.chat_message(role):
        st.markdown(content)


def main():
    """主應用介面"""
    st.set_page_config(
        page_title="AskCNA - 事實查核助手",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 AskCNA - 事實查核助手")

    init_session_state()
    bot = StreamlitFactCheckBot()

    # 側邊欄
    with st.sidebar:
        st.header("📜 使用說明")
        st.markdown("""
        **使用步驟:**
        1. 輸入要查核的新聞或消息
        2. 等待AI分析查核點和生成解釋
        3. 根據AI評估選擇下一步動作
        4. 最終生成完整的查核報告

        **選擇項目:**
        1. 採用AI建議的問題
        2. 自行輸入問題
        3. 生成最終報告
        """)

        def clear_and_log():
            print("====> 清除對話紀錄")
            bot.reset_fact_check_state()
        st.button("🗑️ 清除對話記錄", on_click=clear_and_log)

    # 顯示對話歷史
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])

    # 根據狀態顯示不同UI
    if st.session_state.fact_check_state == "waiting_user_choice":
        # 等待用戶選擇，使用按鈕
        # 使用 session_state 來跟踪按鈕點擊，避免立即rerun
        if "button_clicked" not in st.session_state:
            st.session_state.button_clicked = None

        if st.button("1. 採用AI建議的問題", key="use_ai_btn"):
            st.session_state.button_clicked = "1"

        if st.button("2. 自行輸入問題", key="custom_btn"):
            st.session_state.button_clicked = "2"

        if st.button("3. 不再提問，生成最終結果", key="final_btn"):
            st.session_state.button_clicked = "3"

        # 處理按鈕點擊
        if st.session_state.button_clicked:
            choice = st.session_state.button_clicked
            st.session_state.button_clicked = None  # 重置
            try:
                bot.handle_user_choice(choice)
                st.rerun()
            except Exception as e:
                st.error(f"❌ 處理選擇時發生錯誤: {str(e)}")

    elif st.session_state.fact_check_state == "waiting_custom_question":
        # 等待用戶輸入自定義問題
        if custom_question := st.chat_input("請輸入您的問題..."):
            with st.chat_message("user"):
                st.markdown(custom_question)
            st.session_state.messages.append({"role": "user", "content": custom_question})

            try:
                bot.continue_with_question(custom_question, "用戶輸入")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 處理問題時發生錯誤: {str(e)}")

    elif st.session_state.fact_check_state == "completed":
        pass

    # 聊天輸入 - 只在沒有進行中的查核時顯示
    elif st.session_state.fact_check_state is None:
        if prompt := st.chat_input("請輸入要查核的新聞或消息..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            try:
                bot.start_fact_check(prompt)
                st.rerun()
            except Exception as e:
                with st.chat_message("assistant"):
                    st.error(f"❌ 查核過程發生錯誤: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ 抱歉，查核過程發生錯誤: {str(e)}"
                })

if __name__ == "__main__":
    main()