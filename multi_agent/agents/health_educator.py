# -*- coding: utf-8 -*-
"""
AI 科普宣教专家（Health Educator Agent）
=========================================
用通俗易懂的语言向患者讲解疾病知识和生活方式建议。
"""

from langchain_core.messages import SystemMessage, ToolMessage
from multi_agent.state import MultiAgentState
from multi_agent.config import get_llm
from multi_agent.tools import search_guidelines

TOOLS = [search_guidelines]

SYSTEM_PROMPT = """你是一位亲切和蔼的健康科普专家，擅长用通俗易懂的语言向患者讲解医学知识。

你的职责：
1. 用大白话解释疾病是怎么回事（避免堆砌专业术语）
2. 给出具体的、可操作的日常生活建议（饮食、运动、作息、心态）
3. 帮助患者建立正确的健康观念，减轻焦虑

回答要求：
- 语言通俗亲切，像一位老朋友在聊天
- 遇到医学术语时，先用大白话解释，再括号注明术语。例如："排尿不太顺畅（医学上叫下尿路症状，LUTS）"
- 适当使用 emoji 增加亲和力 😊
- 建议要具体实用，比如"每天步行 30 分钟"而不是"适当运动"
- 内容基于临床指南，但表达方式面向普通患者
- 最后用一句鼓励性的话结尾

你可以使用工具从临床指南中检索权威的健康建议。"""


def educator_node(state: MultiAgentState) -> dict:
    """
    科普宣教专家节点。

    内置 ReAct 循环，最多 3 轮工具调用。
    """
    llm = get_llm(temperature=0.5)  # 稍高 temperature 让语言更生动自然
    llm_with_tools = llm.bind_tools(TOOLS)

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])

    tool_map = {t.name: t for t in TOOLS}
    max_iterations = 3

    print("[Educator] 开始科普宣教...")

    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print(f"[Educator] 完成科普（经过 {i} 轮工具调用）")
            return {"messages": [response]}

        print(f"[Educator] 第 {i+1} 轮工具调用：")
        for tc in response.tool_calls:
            print(f"  → {tc['name']}({tc['args']})")
            result = tool_map[tc["name"]].invoke(tc["args"])
            tool_msg = ToolMessage(content=str(result), tool_call_id=tc["id"])
            messages.append(tool_msg)

    final_response = llm.invoke(messages)
    print(f"[Educator] 达到最大轮次，强制生成最终答案")
    return {"messages": [final_response]}
