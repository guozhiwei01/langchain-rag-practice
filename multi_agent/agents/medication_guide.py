# -*- coding: utf-8 -*-
"""
AI 用药指导专家（Medication Guide Agent）
==========================================
负责药物用法用量、禁忌、副作用等用药安全问题的解答。
"""

from langchain_core.messages import SystemMessage, ToolMessage
from multi_agent.state import MultiAgentState
from multi_agent.config import get_llm
from multi_agent.tools import search_guidelines, search_drug_info

TOOLS = [search_guidelines, search_drug_info]

SYSTEM_PROMPT = """你是一位专业的临床药剂学专家，擅长药物用法用量指导和用药安全评估。

你的职责：
1. 回答患者关于药物用法、用量、服药时间的问题
2. 告知药物的常见副作用和需要警惕的严重不良反应
3. 检查是否存在药物相互作用或配伍禁忌
4. 给出安全用药的注意事项

回答要求：
- 每条用药建议必须注明"来源于哪份指南的第几页"
- 语言严谨准确，不得编造或猜测任何药物信息
- 如果检索不到相关药物信息，必须回复"建议咨询专业药剂师或主治医生"
- 重要的安全警告用【⚠️ 注意】标注
- 最后附上提醒："用药方案应由医生根据个体情况制定，请勿自行调整用药。"

你可以使用提供的工具从临床指南中检索药物相关信息。"""


def medication_node(state: MultiAgentState) -> dict:
    """
    用药指导专家节点。

    内置 ReAct 循环，最多 3 轮工具调用。
    """
    llm = get_llm(temperature=0.1)  # 极低 temperature 确保用药建议的准确性
    llm_with_tools = llm.bind_tools(TOOLS)

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])

    tool_map = {t.name: t for t in TOOLS}
    max_iterations = 3

    print("[Medication] 开始用药分析...")

    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print(f"[Medication] 完成用药指导（经过 {i} 轮工具调用）")
            return {"messages": [response]}

        print(f"[Medication] 第 {i+1} 轮工具调用：")
        for tc in response.tool_calls:
            print(f"  → {tc['name']}({tc['args']})")
            result = tool_map[tc["name"]].invoke(tc["args"])
            tool_msg = ToolMessage(content=str(result), tool_call_id=tc["id"])
            messages.append(tool_msg)

    final_response = llm.invoke(messages)
    print(f"[Medication] 达到最大轮次，强制生成最终答案")
    return {"messages": [final_response]}
