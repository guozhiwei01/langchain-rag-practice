# -*- coding: utf-8 -*-
"""
Step 3：LangGraph RAG Agent —— 理解 StateGraph / 条件路由 / ReAct 循环
=====================================================================
学习目标：
  1. 理解 StateGraph 的三要素：State / Node / Edge
  2. 理解条件边（add_conditional_edges）的路由机制
  3. 理解 ReAct Agent 循环（LLM → Tool → LLM → ...）
  4. 理解 Checkpointer 持久化机制

原理说明（面试重点）：
  LangGraph 与 LCEL Chain 的本质区别：
    - LCEL = DAG（有向无环图），数据单向流过，不能循环
    - LangGraph = 有状态的有向图，支持循环（cycle）
    - Agent 需要循环：LLM推理 → 调tool → 看结果 → 再推理... 直到满意
"""

import io
import sys
import os
from pathlib import Path
from typing import Annotated, TypedDict
from dotenv import load_dotenv

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

load_dotenv()

# ─── 配置 ─────────────────────────────────────────────────────────────────────
PG_HOST     = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT     = os.getenv("PG_PORT", "5432")
PG_USER     = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "123456")
PG_DATABASE = os.getenv("PG_DATABASE", "rag_practice")
PG_CONN_STR = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
QWEN_MODEL        = os.getenv("QWEN_MODEL", "qwen-plus")
BGE_MODEL_NAME    = os.getenv("BGE_MODEL_NAME", "BAAI/bge-m3")
COLLECTION_NAME   = "prostate_guidelines"


# ═══════════════════════════════════════════════════════════════════════════════
# Part A：手动构建 StateGraph RAG Agent（理解原理）
# ═══════════════════════════════════════════════════════════════════════════════

# ─── 原理 1：State 定义 ───────────────────────────────────────────────────────
#
# State 是所有 Node 共享的数据结构。
# 面试重点：Annotated[list, add_messages] 中的 add_messages 是 Reducer。
#   - 没有 Reducer：新值覆盖旧值
#   - 有 Reducer（add_messages）：新消息追加到列表，不覆盖
#   - 这解决了"多个 Node 都要往 messages 里加消息"的问题

from langgraph.graph import add_messages

class AgentState(TypedDict):
    """Agent 共享状态 —— 所有 Node 都读写这个结构"""
    messages: Annotated[list, add_messages]  # 消息列表，用 add_messages 做 reducer


# ─── 原理 2：定义 Tools ──────────────────────────────────────────────────────

from langchain_core.tools import tool

@tool
def search_medical_guidelines(query: str) -> str:
    """根据用户的医学问题，从前列腺相关临床指南中检索相关段落。
    当用户问到前列腺炎、前列腺增生(BPH)、排尿困难等泌尿科问题时使用此工具。"""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_postgres import PGVector

    embeddings = HuggingFaceEmbeddings(
        model_name=BGE_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=PG_CONN_STR,
    )
    docs = vectorstore.similarity_search(query, k=3)
    if not docs:
        return "未检索到相关内容。"
    return "\n\n---\n\n".join(
        f"【{Path(doc.metadata.get('source', '?')).name} "
        f"p{doc.metadata.get('page', '?')}】\n{doc.page_content}"
        for doc in docs
    )


@tool
def calculate_bmi(weight_kg: float, height_m: float) -> str:
    """计算 BMI 指数。当用户提供体重(kg)和身高(m)时使用。"""
    bmi = weight_kg / (height_m ** 2)
    cat = "偏瘦" if bmi < 18.5 else "正常" if bmi < 24 else "偏胖" if bmi < 28 else "肥胖"
    return f"BMI = {bmi:.1f}（{cat}）"


TOOLS = [search_medical_guidelines, calculate_bmi]


# ─── 原理 3：Node 定义 ────────────────────────────────────────────────────────
#
# Node 是普通 Python 函数：
#   输入：当前 State
#   输出：partial state update（只返回要更新的字段）
#
# 面试重点：Node 返回 {"messages": [ai_msg]} 时，
#           因为 messages 有 add_messages reducer，
#           所以是追加，不是覆盖！

def get_llm_with_tools():
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=QWEN_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.3,
    )
    return llm.bind_tools(TOOLS)

# 缓存 LLM 实例
_llm_with_tools = None
def _get_llm():
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = get_llm_with_tools()
    return _llm_with_tools


def llm_node(state: AgentState) -> dict:
    """
    LLM 推理节点：
    接收消息历史 → 调用带 tools 的 LLM → 返回 AIMessage
    AIMessage 可能包含 tool_calls（需要调 tool）或 content（最终答案）
    """
    print("[Node: LLM] 推理中...")
    llm = _get_llm()
    response = llm.invoke(state["messages"])

    if response.tool_calls:
        print(f"[Node: LLM] → 决定调用 {len(response.tool_calls)} 个 tool")
        for tc in response.tool_calls:
            print(f"             {tc['name']}({tc['args']})")
    else:
        print(f"[Node: LLM] → 直接回答（前80字）：{response.content[:80]}...")

    return {"messages": [response]}  # add_messages reducer 会追加


def tool_node(state: AgentState) -> dict:
    """
    Tool 执行节点：
    从最后一条 AIMessage 中读取 tool_calls → 执行 → 返回 ToolMessages
    """
    from langchain_core.messages import ToolMessage

    print("[Node: Tools] 执行工具中...")
    tool_map = {t.name: t for t in TOOLS}
    last_msg = state["messages"][-1]  # 最后一条是 AIMessage（含 tool_calls）

    tool_messages = []
    for tc in last_msg.tool_calls:
        print(f"  → 执行 {tc['name']}...")
        result = tool_map[tc["name"]].invoke(tc["args"])
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )
        print(f"  ← 结果（前100字）：{str(result)[:100]}...")

    return {"messages": tool_messages}  # 追加 ToolMessages


# ─── 原理 4：条件边 —— 路由函数 ──────────────────────────────────────────────
#
# 条件边决定 "LLM 节点之后走哪条路"：
#   - 如果 AIMessage 有 tool_calls → 去 tool_node
#   - 如果没有 tool_calls → 结束（END）
#
# 面试重点：路由函数应该是轻量级的判断逻辑，
#           不应该包含业务逻辑（关注点分离）

def should_continue(state: AgentState) -> str:
    """路由函数：判断是否需要继续调用 tool"""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"    # → 去 tool_node
    return "end"          # → 结束


# ─── 原理 5：组装 StateGraph ──────────────────────────────────────────────────
#
# 图结构：
#   START → llm_node ──(条件边)──→ tool_node → llm_node（循环！）
#                     └──────────→ END
#
# 面试重点：
#   compile() 做了什么？
#     ① 验证图结构（所有节点都可达、没有孤立节点）
#     ② 锁定图定义（编译后不能再修改）
#     ③ 返回一个 Runnable 对象（支持 invoke/stream/batch）

def build_agent_graph():
    """手动构建 LangGraph Agent —— 面试必须能手写"""
    from langgraph.graph import StateGraph, START, END

    # 创建图
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_node)

    # 添加边
    graph.add_edge(START, "llm")         # 入口 → LLM
    graph.add_conditional_edges(          # LLM → 条件路由
        "llm",
        should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "llm")       # Tool → LLM（循环！）

    # 编译
    app = graph.compile()

    print("[OK] Agent Graph 构建完成")
    print(f"     类型：{type(app).__name__}")
    print(f"     是 Runnable：{hasattr(app, 'invoke')}")
    return app


# ═══════════════════════════════════════════════════════════════════════════════
# Part B：使用预构建的 create_react_agent（生产推荐）
# ═══════════════════════════════════════════════════════════════════════════════
#
# create_react_agent 内部做的事情和 Part A 完全一样！
# 但它帮你封装好了，一行代码搞定。
#
# 面试点：知道 create_react_agent 等价于手动构建的图，
#         但能用 Part A 的方式解释它的内部原理。

def build_prebuilt_agent():
    """使用预构建的 ReAct Agent"""
    from langgraph.prebuilt import create_react_agent
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=QWEN_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.3,
    )

    agent = create_react_agent(model=llm, tools=TOOLS)

    print("[OK] 预构建 ReAct Agent 创建完成")
    return agent


# ═══════════════════════════════════════════════════════════════════════════════
# Part C：Checkpointer —— 持久化与多轮对话
# ═══════════════════════════════════════════════════════════════════════════════
#
# 面试重点：
#   - MemorySaver：内存中保存，重启丢失（开发用）
#   - SqliteSaver / PostgresSaver：持久化到数据库（生产用）
#   - thread_id：区分不同对话线程
#   - Checkpointer 在每个 Node 执行后自动保存 State 快照

def build_agent_with_memory():
    """带记忆的 Agent —— Checkpointer 演示"""
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=QWEN_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.3,
    )

    checkpointer = MemorySaver()
    agent = create_react_agent(model=llm, tools=TOOLS, checkpointer=checkpointer)

    print("[OK] 带 Checkpointer 的 Agent 创建完成")
    return agent


# ─── 演示入口 ─────────────────────────────────────────────────────────────────

def demo_manual_graph(query: str):
    """演示 1：手动构建的 StateGraph"""
    from langchain_core.messages import HumanMessage

    print("\n" + "=" * 70)
    print("  演示 1：手动构建 StateGraph Agent")
    print("=" * 70)

    agent = build_agent_graph()
    result = agent.invoke({"messages": [HumanMessage(content=query)]})

    print(f"\n{'─' * 70}")
    print(f"最终答案：")
    print(result["messages"][-1].content)
    print(f"{'─' * 70}")
    print(f"总消息数：{len(result['messages'])}")


def demo_memory(query1: str, query2: str):
    """演示 2：Checkpointer 多轮对话"""
    from langchain_core.messages import HumanMessage

    print("\n" + "=" * 70)
    print("  演示 2：Checkpointer 多轮对话记忆")
    print("=" * 70)

    agent = build_agent_with_memory()
    config = {"configurable": {"thread_id": "demo-001"}}

    # 第一轮
    print(f"\n[轮次 1] {query1}")
    r1 = agent.invoke({"messages": [HumanMessage(content=query1)]}, config)
    print(f"回答：{r1['messages'][-1].content[:200]}...")

    # 第二轮（同一个 thread_id，自动带上下文）
    print(f"\n[轮次 2] {query2}")
    r2 = agent.invoke({"messages": [HumanMessage(content=query2)]}, config)
    print(f"回答：{r2['messages'][-1].content[:200]}...")
    print(f"\n[验证] 第二轮总消息数：{len(r2['messages'])}（包含第一轮的历史）")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "前列腺炎的诊断标准是什么？"

    # 演示 1：手动 StateGraph（理解原理）
    demo_manual_graph(query)

    # 演示 2：Checkpointer 记忆
    demo_memory(
        "前列腺炎有哪些类型？",
        "你刚才说的第三种类型具体怎么诊断？"  # 测试是否记住上下文
    )

    print("\n" + "=" * 70)
    print("  面试关键总结")
    print("=" * 70)
    print("""
    ┌─────────────────────────────────────────────────────────┐
    │  LangGraph 核心概念速记                                  │
    ├─────────────────────────────────────────────────────────┤
    │  State     = 共享状态（TypedDict + Reducer）             │
    │  Node      = 处理函数（接收 state → 返回 partial update）│
    │  Edge      = 控制流（普通边 / 条件边）                    │
    │  compile() = 锁定图 → 返回 Runnable                     │
    │  Checkpointer = 每步自动保存 state 快照                  │
    │                                                         │
    │  ReAct 循环 = LLM → (有tool_calls?) → Tools → LLM → ...│
    │  这就是 Agent 的本质！                                   │
    └─────────────────────────────────────────────────────────┘
    """)
