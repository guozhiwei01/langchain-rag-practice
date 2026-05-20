# -*- coding: utf-8 -*-
"""
多智能体工作流图编排
====================
使用 langgraph.StateGraph 组装完整的多智能体协作工作流。

图结构：
  START → supervisor → (条件路由) → interpreter / medication / educator / sentinel
                                          ↓            ↓            ↓
                                      sentinel ← ── ── ── ── ── ──┘
                                          ↓
                                         END
"""

from langgraph.graph import StateGraph, START, END

from multi_agent.state import MultiAgentState
from multi_agent.agents.supervisor import supervisor_node
from multi_agent.agents.report_interpreter import interpreter_node
from multi_agent.agents.medication_guide import medication_node
from multi_agent.agents.health_educator import educator_node
from multi_agent.agents.risk_sentinel import sentinel_node


def _route_by_next_agent(state: MultiAgentState) -> str:
    """
    条件边路由函数：读取 state["next_agent"]，决定下一步走哪个节点。
    """
    return state["next_agent"]


def build_graph():
    """
    构建并编译多智能体工作流图。

    返回编译后的 CompiledGraph（Runnable 协议），
    可直接调用 invoke() / stream()。
    """
    graph = StateGraph(MultiAgentState)

    # ── 添加节点 ──
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("interpreter", interpreter_node)
    graph.add_node("medication", medication_node)
    graph.add_node("educator", educator_node)
    graph.add_node("sentinel", sentinel_node)

    # ── 添加边 ──

    # 入口 → 分诊协调员
    graph.add_edge(START, "supervisor")

    # 分诊协调员 → 条件路由到对应的专科智能体
    graph.add_conditional_edges(
        "supervisor",
        _route_by_next_agent,
        {
            "interpreter": "interpreter",
            "medication":  "medication",
            "educator":    "educator",
            "sentinel":    "sentinel",   # 危急情况直接进哨兵
        },
    )

    # 所有专科智能体执行完毕 → 必须经过风险预警哨兵审计
    graph.add_edge("interpreter", "sentinel")
    graph.add_edge("medication", "sentinel")
    graph.add_edge("educator", "sentinel")

    # 哨兵审计完毕 → 结束
    graph.add_edge("sentinel", END)

    # ── 编译 ──
    app = graph.compile()

    print("[Graph] ✅ 多智能体工作流图编译完成")
    print(f"        节点：supervisor → interpreter / medication / educator → sentinel")
    return app
