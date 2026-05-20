# -*- coding: utf-8 -*-
"""
AI 风险预警哨兵（Risk Sentinel Agent）
=======================================
安全审计节点，拥有一票否决权。
采用"规则引擎 + LLM"双重审计机制，规则优先。
"""

import re
from langchain_core.messages import AIMessage, SystemMessage
from multi_agent.state import MultiAgentState
from multi_agent.config import get_llm

# ─── 规则引擎：危急症状与危急值的硬编码匹配规则 ──────────────────────────────
# 优先于 LLM 判断，确保不会因为大模型幻觉而漏报
CRITICAL_PATTERNS = [
    # 危急症状
    r"急性尿潴留",
    r"完全(不能|无法)排尿",
    r"大量(出血|血尿)",
    r"剧烈(疼痛|腹痛|胀痛)",
    r"(高热|寒战|体温\s*[>≥]\s*39)",
    r"意识(模糊|不清|丧失)",
    r"(呼吸困难|胸痛|胸闷)",
    r"(昏迷|休克|抽搐)",
    # 危急指标值
    r"PSA\s*[>≥]\s*10",
    r"肌酐\s*[>≥]\s*(400|500|600)",
    r"血红蛋白\s*[<≤]\s*(60|70)",
]

# 编译正则提高性能
_compiled_patterns = [re.compile(p) for p in CRITICAL_PATTERNS]

AUDIT_SYSTEM_PROMPT = """你是一个医疗安全审计员。请审查以下医学回答是否存在安全风险。

审查要点：
1. 回答是否包含可能导致患者延误就医的不当建议
2. 回答是否遗漏了需要紧急处理的危险信号
3. 回答是否存在明显的医学错误或误导性信息
4. 回答是否缺少必要的就医提醒

请判断风险等级并严格按以下格式输出（仅输出一行）：
- 如果安全无风险：SAFE
- 如果有轻微风险需要补充提醒：WARNING|补充的提醒内容
- 如果有严重风险需要拦截：CRITICAL|重写后的安全回答"""

# ─── 标准免责声明 ─────────────────────────────────────────────────────────────
DISCLAIMER_WARNING = "\n\n---\n⚠️ **温馨提示**：以上内容仅供健康参考，不构成医疗诊断或治疗建议。如有不适，请及时到正规医疗机构就诊。"

DISCLAIMER_CRITICAL = (
    "\n\n🚨🚨🚨 **紧急预警** 🚨🚨🚨\n\n"
    "根据您描述的症状，**可能存在需要紧急处理的医学情况**。\n\n"
    "**请立即采取以下措施：**\n"
    "1. 🏥 **立即前往最近的医院急诊科就诊**\n"
    "2. 📞 如无法自行前往，请拨打 **120 急救电话**\n"
    "3. ⛔ 在就医前，请勿自行服药或进行任何处理\n\n"
    "---\n"
    "*本系统为 AI 辅助工具，不能替代医生的专业诊断。紧急情况请务必寻求线下医疗帮助。*"
)


def _check_rules(text: str) -> str | None:
    """
    规则引擎检查。返回匹配到的危急模式，若无匹配返回 None。
    """
    for pattern in _compiled_patterns:
        match = pattern.search(text)
        if match:
            return match.group()
    return None


def sentinel_node(state: MultiAgentState) -> dict:
    """
    风险预警哨兵节点。

    双重审计机制：
    1. 规则引擎：用正则匹配危急症状和危急值（快速、可靠、不依赖 LLM）
    2. LLM 审计：让 LLM 判断回答是否存在不当建议（补充覆盖规则遗漏的情况）
    """
    messages = state["messages"]

    # 获取需要审计的内容：用户的输入 + 上一个智能体的回复
    user_input = ""
    agent_reply = ""
    for msg in messages:
        if hasattr(msg, "content"):
            content = msg.content
            # HumanMessage 是用户输入
            if msg.__class__.__name__ == "HumanMessage":
                user_input = content
            # AIMessage 是智能体回复
            elif msg.__class__.__name__ == "AIMessage" and content:
                agent_reply = content

    audit_text = f"{user_input} {agent_reply}"

    # ── 第一道防线：规则引擎 ──
    critical_match = _check_rules(audit_text)
    if critical_match:
        print(f"[Sentinel] 🚨 规则引擎触发！匹配到危急模式：「{critical_match}」")
        warning_content = DISCLAIMER_CRITICAL
        return {
            "messages": [AIMessage(content=warning_content)],
            "risk_level": "critical",
        }

    # ── 第二道防线：LLM 审计 ──
    if not agent_reply:
        # 如果没有智能体回复（可能是直接从 supervisor 路由过来的危急情况）
        print("[Sentinel] 无专科智能体回复，直接输出紧急预警")
        return {
            "messages": [AIMessage(content=DISCLAIMER_CRITICAL)],
            "risk_level": "critical",
        }

    llm = get_llm(temperature=0.0)
    audit_response = llm.invoke([
        SystemMessage(content=AUDIT_SYSTEM_PROMPT),
        {"role": "user", "content": f"用户问题：{user_input}\n\n医生的回答：{agent_reply}"},
    ])

    result = audit_response.content.strip()

    if result.startswith("CRITICAL"):
        print("[Sentinel] 🚨 LLM 审计判定：高危风险！")
        # 提取 LLM 重写的内容
        rewritten = result.split("|", 1)[1].strip() if "|" in result else ""
        final_content = rewritten + DISCLAIMER_CRITICAL if rewritten else DISCLAIMER_CRITICAL
        return {
            "messages": [AIMessage(content=final_content)],
            "risk_level": "critical",
        }

    elif result.startswith("WARNING"):
        print("[Sentinel] ⚠️ LLM 审计判定：轻微风险，补充提醒")
        extra_warning = result.split("|", 1)[1].strip() if "|" in result else ""
        # 在原回复后追加提醒
        final_content = agent_reply
        if extra_warning:
            final_content += f"\n\n⚠️ **补充提醒**：{extra_warning}"
        final_content += DISCLAIMER_WARNING
        return {
            "messages": [AIMessage(content=final_content)],
            "risk_level": "warning",
        }

    else:
        # SAFE：原回复通过审计，追加标准免责声明
        print("[Sentinel] ✅ 审计通过：安全")
        final_content = agent_reply + DISCLAIMER_WARNING
        return {
            "messages": [AIMessage(content=final_content)],
            "risk_level": "safe",
        }
