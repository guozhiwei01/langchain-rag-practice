# -*- coding: utf-8 -*-
"""
多智能体系统 CLI 入口
=====================
用法：
  uv run -m multi_agent.main ask "你的问题"
  uv run -m multi_agent.main demo              # 运行 4 个场景的演示
"""

import io
import sys

# 终端强制 UTF-8 输出
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from langchain_core.messages import HumanMessage
from multi_agent.graph import build_graph
from multi_agent.config import get_langfuse_handler


def ask(query: str):
    """单次问答：用户提问 → 多智能体协作 → 输出最终答案。"""
    print("=" * 70)
    print("  🏥 医疗多智能体系统")
    print("=" * 70)
    print(f"\n[用户] {query}\n")

    app = build_graph()
    langfuse_handler = get_langfuse_handler()
    invoke_config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}

    result = app.invoke({
        "messages": [HumanMessage(content=query)],
        "next_agent": "",
        "risk_level": "",
    }, config=invoke_config)

    # 获取最终回复（最后一条 AIMessage）
    final_answer = result["messages"][-1].content
    risk_level = result.get("risk_level", "unknown")

    print(f"\n{'─' * 70}")

    # 根据风险等级添加视觉标识
    if risk_level == "critical":
        print("🚨 [风险等级: 高危]")
    elif risk_level == "warning":
        print("⚠️  [风险等级: 注意]")
    else:
        print("✅ [风险等级: 安全]")

    print(f"{'─' * 70}\n")
    print(final_answer)
    print(f"\n{'=' * 70}")

    return result


def demo():
    """运行 4 个场景的端到端演示。"""
    test_cases = [
        {
            "scene":  "场景 1：AI 评估报告解读",
            "query":  "我的PSA值是6.5ng/mL，IPSS评分18分，这些指标说明什么？",
        },
        {
            "scene":  "场景 2：AI 用药指导",
            "query":  "医生给我开了非那雄胺和坦索罗辛，这两个药怎么吃？有什么副作用？",
        },
        {
            "scene":  "场景 3：AI 科普宣教",
            "query":  "得了前列腺增生，平时生活饮食要注意些什么？",
        },
        {
            "scene":  "场景 4：AI 风险预警",
            "query":  "我突然完全无法排尿，小腹胀痛难忍，已经6个小时了怎么办？",
        },
    ]

    print("\n" + "█" * 70)
    print("  🏥 医疗多智能体系统 —— 四场景端到端演示")
    print("█" * 70)

    app = build_graph()
    langfuse_handler = get_langfuse_handler()
    invoke_config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}

    for i, tc in enumerate(test_cases, 1):
        print(f"\n\n{'━' * 70}")
        print(f"  {tc['scene']}")
        print(f"{'━' * 70}")
        print(f"\n[用户] {tc['query']}\n")

        result = app.invoke({
            "messages": [HumanMessage(content=tc["query"])],
            "next_agent": "",
            "risk_level": "",
        }, config=invoke_config)

        final_answer = result["messages"][-1].content
        risk_level = result.get("risk_level", "unknown")

        print(f"\n{'─' * 70}")
        if risk_level == "critical":
            print("🚨 [风险等级: 高危]")
        elif risk_level == "warning":
            print("⚠️  [风险等级: 注意]")
        else:
            print("✅ [风险等级: 安全]")
        print(f"{'─' * 70}\n")
        print(final_answer)

    print(f"\n\n{'█' * 70}")
    print("  演示完成！")
    print(f"{'█' * 70}\n")


# ─── CLI 入口 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：")
        print("  uv run -m multi_agent.main ask '你的问题'   # 单次问答")
        print("  uv run -m multi_agent.main demo             # 四场景演示")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "ask":
        query = sys.argv[2] if len(sys.argv) > 2 else "前列腺增生怎么治疗？"
        ask(query)

    elif cmd == "demo":
        demo()

    else:
        print(f"未知命令：{cmd}")
        print("可用命令：ask / demo")
