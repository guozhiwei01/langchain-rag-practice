# -*- coding: utf-8 -*-
"""
多智能体共享状态定义
====================
所有 Node 都读写这个 State 结构。
"""

from typing import Annotated, TypedDict
from langgraph.graph import add_messages


class MultiAgentState(TypedDict):
    """
    多智能体共享状态。

    Attributes:
        messages: 完整的消息历史。使用 add_messages reducer，
                  所有节点追加消息而非覆盖。
        next_agent: 路由目标。由 supervisor 写入，条件边读取。
                    可选值: "interpreter" / "medication" / "educator" / "sentinel"
        risk_level: 风险等级。由 sentinel 写入。
                    可选值: "safe" / "warning" / "critical"
    """
    messages: Annotated[list, add_messages]
    next_agent: str
    risk_level: str
