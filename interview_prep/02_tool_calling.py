# -*- coding: utf-8 -*-
"""
Step 2：Tool Calling 实践 —— 理解 bind_tools 与 RunnableBinding
================================================================
学习目标：
  1. 理解 @tool 装饰器如何将函数转为 Tool 对象
  2. 理解 bind_tools() 的底层机制（RunnableBinding）
  3. 理解 AIMessage.tool_calls 的统一格式
  4. 手动实现一个 tool calling 循环

原理说明（面试重点）：
  bind_tools() 做了两件事：
    ① 将 Tool/函数 → OpenAI function calling JSON Schema
    ② 返回 RunnableBinding（不修改原模型，创建新 Runnable，绑定 tools 参数）

  执行时（invoke）：
    ① LLM 收到 messages + tools 定义
    ② LLM 决定是否调用 tool（返回 AIMessage.tool_calls）
    ③ 开发者执行 tool，把结果作为 ToolMessage 返回
    ④ LLM 根据 ToolMessage 生成最终答案

  面试点：bind_tools 返回的不是修改后的 LLM，
          而是一个新的 RunnableBinding 对象
"""

import io
import sys
import os
from pathlib import Path
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
# 原理 1：@tool 装饰器 —— 函数 → Tool 对象
# ═══════════════════════════════════════════════════════════════════════════════
#
# @tool 做了什么？
#   1. 读取函数名 → tool name
#   2. 读取 docstring → tool description（告诉 LLM 这个 tool 的用途）
#   3. 读取函数签名（type hints）→ 参数 JSON Schema
#   4. 包装为 StructuredTool 对象（也是 Runnable）
#
# 面试点：docstring 非常重要！LLM 靠它理解什么时候该调用这个 tool。

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
        f"【来源：{Path(doc.metadata.get('source', '未知')).name} "
        f"第{doc.metadata.get('page', '?')}页】\n{doc.page_content}"
        for doc in docs
    )


@tool
def calculate_bmi(weight_kg: float, height_m: float) -> str:
    """计算 BMI 指数。当用户提供体重(kg)和身高(m)并想知道 BMI 时使用。"""
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        category = "偏瘦"
    elif bmi < 24:
        category = "正常"
    elif bmi < 28:
        category = "偏胖"
    else:
        category = "肥胖"
    return f"BMI = {bmi:.1f}，属于{category}范围。"


# ═══════════════════════════════════════════════════════════════════════════════
# 原理 2：bind_tools() 的底层机制
# ═══════════════════════════════════════════════════════════════════════════════
#
# llm.bind_tools(tools) 的内部过程：
#
#   Step 1: 对每个 tool，调用 convert_to_openai_tool() 转为 JSON Schema：
#     {
#       "type": "function",
#       "function": {
#         "name": "search_medical_guidelines",
#         "description": "根据用户的医学问题，从前列腺相关临床指南中检索相关段落...",
#         "parameters": {
#           "type": "object",
#           "properties": {"query": {"type": "string"}},
#           "required": ["query"]
#         }
#       }
#     }
#
#   Step 2: 返回 RunnableBinding(bound=llm, kwargs={"tools": [...]})
#
#   关键：没有修改原始 llm 对象！创建了一个新的包装器。

def demo_bind_tools():
    """演示 bind_tools 的机制"""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=QWEN_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.3,
    )

    tools = [search_medical_guidelines, calculate_bmi]

    print("=" * 70)
    print("  bind_tools 机制演示")
    print("=" * 70)

    # 查看 tool 的 schema（面试可能会问）
    print("\n[1] Tool 定义：")
    for t in tools:
        print(f"    - name: {t.name}")
        print(f"      description: {t.description[:60]}...")
        print(f"      schema: {t.args_schema.model_json_schema()}")

    # bind_tools → RunnableBinding
    llm_with_tools = llm.bind_tools(tools)

    print(f"\n[2] bind_tools 前后对比：")
    print(f"    原始 LLM 类型：{type(llm).__name__}")
    print(f"    绑定后类型：  {type(llm_with_tools).__name__}")
    print(f"    是新对象吗？  {llm is not llm_with_tools}")

    return llm, llm_with_tools, tools


# ═══════════════════════════════════════════════════════════════════════════════
# 原理 3：手动实现 Tool Calling 循环
# ═══════════════════════════════════════════════════════════════════════════════
#
# 这是理解 LangGraph ReAct Agent 的基础！
#
# 完整流程：
#   1. 用户提问 → messages = [HumanMessage("前列腺炎怎么治疗？")]
#   2. LLM(messages + tools) → AIMessage(tool_calls=[...])
#   3. 执行 tool → ToolMessage(content="检索结果...")
#   4. LLM(messages + AIMessage + ToolMessage) → AIMessage(content="最终答案")
#
# 面试重点：
#   - AIMessage.tool_calls 是 LangChain 统一格式，与 OpenAI 解耦
#   - ToolMessage 必须包含 tool_call_id，对应 AIMessage 中的 tool call
#   - 这个循环就是 LangGraph ReAct Agent 的核心逻辑

def manual_tool_calling_loop(query: str):
    """手动实现 tool calling 循环 —— 理解 Agent 的本质"""
    from langchain_core.messages import HumanMessage, ToolMessage

    llm, llm_with_tools, tools = demo_bind_tools()

    # 构建 tool name → tool function 的映射
    tool_map = {t.name: t for t in tools}

    # Step 1: 用户提问
    messages = [HumanMessage(content=query)]
    print(f"\n[STEP 1] 用户提问：{query}")

    # Step 2: LLM 决定是否调用 tool
    print(f"\n[STEP 2] LLM 推理中...")
    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)

    print(f"    LLM 返回类型：{type(ai_message).__name__}")
    print(f"    有 tool_calls？{bool(ai_message.tool_calls)}")

    if not ai_message.tool_calls:
        # LLM 认为不需要调用 tool，直接回答
        print(f"\n[结果] LLM 直接回答（无需 tool）：")
        print(f"    {ai_message.content[:200]}...")
        return ai_message.content

    # Step 3: 执行 tool（可能有多个 tool calls）
    print(f"\n[STEP 3] 执行 Tool Calls：")
    for tc in ai_message.tool_calls:
        print(f"    调用：{tc['name']}({tc['args']})")
        print(f"    tool_call_id：{tc['id']}")

        # 执行对应的 tool
        tool_fn = tool_map[tc["name"]]
        result = tool_fn.invoke(tc["args"])
        print(f"    结果（前200字）：{str(result)[:200]}...")

        # 构造 ToolMessage（必须包含 tool_call_id）
        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=tc["id"],   # ← 关键：对应 AIMessage 中的 tool call
        )
        messages.append(tool_message)

    # Step 4: LLM 根据 tool 结果生成最终答案
    print(f"\n[STEP 4] LLM 根据 tool 结果生成最终答案...")
    final_response = llm_with_tools.invoke(messages)

    print(f"\n{'=' * 60}")
    print(final_response.content)
    print(f"{'=' * 60}")

    # 打印完整的消息流（面试可能会问消息列表的结构）
    print(f"\n[消息流] 完整的 messages 列表：")
    for i, msg in enumerate(messages + [final_response]):
        print(f"    [{i}] {type(msg).__name__}: {str(msg.content)[:80]}...")

    return final_response.content


# ─── 演示入口 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  Tool Calling 深度实践")
    print("=" * 70)

    query = sys.argv[1] if len(sys.argv) > 1 else "前列腺炎的诊断标准是什么？"

    manual_tool_calling_loop(query)

    print("\n\n" + "─" * 70)
    print("  扩展思考（面试准备）：")
    print("─" * 70)
    print("""
    1. 如果 LLM 连续多次调用 tool 怎么办？→ 需要循环（while tool_calls）
    2. 如果 tool 执行失败怎么办？→ 需要错误处理和重试
    3. 如果想限制 tool 调用次数？→ 设置 max_iterations
    4. 这些问题正是 LangGraph ReAct Agent 解决的！→ 见 03_langgraph_agent.py
    """)
