# -*- coding: utf-8 -*-
"""
AI 评估报告解读专家（Report Interpreter Agent）
================================================
负责解读化验单、检查报告中的指标，给出专业的指标分析。
"""

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from multi_agent.state import MultiAgentState
from multi_agent.config import get_llm
from multi_agent.tools import search_guidelines, search_risk_indicators

TOOLS = [search_guidelines, search_risk_indicators]

SYSTEM_PROMPT = """你是一位资深的泌尿外科医学专家，擅长解读各类检验报告和评估量表。

你的职责：
1. 分析用户提供的检验指标（如 PSA、IPSS 评分、尿流率等），明确指出是否在正常范围内
2. 结合临床指南给出专业的综合分析
3. 如果指标异常，说明可能的临床意义

回答要求：
- 专业、严谨、条理清晰
- 引用指南内容时标注来源
- 逐项分析每个指标后给出综合判断
- 最后附上一句提醒："以上解读仅供参考，具体诊断请以线下医生面诊为准。"

你可以使用提供的工具从临床指南中检索参考信息来辅助你的分析。"""


def interpreter_node(state: MultiAgentState) -> dict:
    """
    报告解读专家节点。

    内置 ReAct 循环：LLM 决定是否调用工具 → 执行 → LLM 生成最终解读。
    最多执行 3 轮工具调用，防止无限循环。
    """
    llm = get_llm(temperature=0.2)
    llm_with_tools = llm.bind_tools(TOOLS)

    # 构建消息：system prompt + 用户的完整消息历史
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])

    tool_map = {t.name: t for t in TOOLS}
    max_iterations = 3

    print("[Interpreter] 开始报告解读...")

    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # 如果 LLM 没有请求调用工具，说明已经生成了最终答案
        if not response.tool_calls:
            print(f"[Interpreter] 完成解读（经过 {i} 轮工具调用）")
            return {"messages": [response]}

        # 执行工具调用
        print(f"[Interpreter] 第 {i+1} 轮工具调用：")
        for tc in response.tool_calls:
            print(f"  → {tc['name']}({tc['args']})")
            result = tool_map[tc["name"]].invoke(tc["args"])
            tool_msg = ToolMessage(content=str(result), tool_call_id=tc["id"])
            messages.append(tool_msg)

    # 如果达到最大轮次，让 LLM 基于已有信息生成最终答案
    final_response = llm.invoke(messages)
    print(f"[Interpreter] 达到最大轮次，强制生成最终答案")
    return {"messages": [final_response]}
