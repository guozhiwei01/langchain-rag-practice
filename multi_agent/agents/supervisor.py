# -*- coding: utf-8 -*-
"""
分诊协调员（Supervisor）
========================
系统入口节点。不做实际医学回答，只做意图分类和路由。
通过 LLM 判断用户输入属于哪个场景，输出 next_agent 字段。
"""

from langchain_core.messages import SystemMessage
from multi_agent.state import MultiAgentState
from multi_agent.config import get_llm

# 危急关键词列表（硬编码规则，优先于 LLM 判断）
CRITICAL_KEYWORDS = [
    "急性尿潴留", "无法排尿", "完全不能排尿",
    "血尿", "大量出血", "尿血",
    "剧烈疼痛", "疼痛难忍", "剧痛",
    "高热", "寒战", "意识模糊", "昏迷",
    "呼吸困难", "胸痛",
]

SYSTEM_PROMPT = """你是一个医疗分诊协调员。你的唯一职责是分析用户的输入，判断应该由哪位专家来回答。

分类规则：
- "interpreter"：用户想解读体检报告、化验单、检查结果、某个检验指标（如 PSA、IPSS 评分）的含义
- "medication"：用户询问药物用法用量、药物副作用、药物禁忌、能否一起吃某些药
- "educator"：用户想了解疾病科普知识、日常保养、饮食建议、生活方式、运动建议
- "sentinel"：用户描述了紧急、危急的症状（如完全无法排尿、剧烈疼痛、大量出血等）

请仅输出以下四个单词之一，不要输出任何其他内容：
interpreter
medication
educator
sentinel"""


def supervisor_node(state: MultiAgentState) -> dict:
    """
    分诊协调员节点。

    1. 先用规则引擎检查是否包含危急关键词（快速、可靠）
    2. 若无危急词，用 LLM 做意图分类
    """
    # 获取用户最新的输入
    last_message = state["messages"][-1]
    user_input = last_message.content if hasattr(last_message, "content") else str(last_message)

    # ── 规则引擎：危急关键词快速拦截 ──
    for keyword in CRITICAL_KEYWORDS:
        if keyword in user_input:
            print(f"[Supervisor] 🚨 检测到危急关键词「{keyword}」→ 直接路由到 sentinel")
            return {"next_agent": "sentinel"}

    # ── LLM 意图分类 ──
    llm = get_llm(temperature=0.0)  # temperature=0 确保分类稳定
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        last_message,
    ]
    response = llm.invoke(messages)
    decision = response.content.strip().lower()

    # 解析 LLM 输出，做容错处理
    valid_agents = {"interpreter", "medication", "educator", "sentinel"}
    if decision not in valid_agents:
        # 模糊匹配兜底
        for agent_name in valid_agents:
            if agent_name in decision:
                decision = agent_name
                break
        else:
            decision = "educator"  # 默认兜底到科普宣教

    print(f"[Supervisor] 意图分类结果：{decision}")
    return {"next_agent": decision}
