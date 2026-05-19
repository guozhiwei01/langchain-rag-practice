# -*- coding: utf-8 -*-
"""
Step 1：用 LCEL 重构 RAG 管道 —— 理解 Runnable 协议
=====================================================
学习目标：
  1. 理解 Runnable 统一接口（invoke / stream / batch）
  2. 理解 LCEL 的 | 管道操作符（RunnableSequence）
  3. 理解 RunnableParallel / RunnablePassthrough 的作用
  4. 对比原生 OpenAI SDK vs langchain-openai 的区别

原理说明（面试重点）：
  - | 操作符 → 调用 __or__() → 返回 RunnableSequence
  - RunnableSequence.invoke(x) → step1.invoke(x) → step2.invoke(result1) → ...
  - RunnableParallel({"a": runnable1, "b": runnable2}).invoke(x) 
    → {"a": runnable1.invoke(x), "b": runnable2.invoke(x)}  并行执行
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
# 原理 1：ChatOpenAI 是一个 Runnable
# ═══════════════════════════════════════════════════════════════════════════════
# 
# 你之前用的是原生 openai.OpenAI，它不是 Runnable，无法用 | 管道组合。
# ChatOpenAI 继承自 BaseChatModel → Runnable，所以可以：
#   - llm.invoke(messages)     → AIMessage
#   - llm.stream(messages)     → Iterator[AIMessageChunk]  （流式输出）
#   - llm.batch([msgs1, msgs2]) → [AIMessage, AIMessage]   （批量调用）
#
# 面试点：ChatOpenAI 封装了 API 调用细节，返回统一的 AIMessage 对象，
#         换模型提供商（如 Anthropic）只需换一行 import，其余代码不变。

def get_llm():
    """获取 LLM（Runnable 协议）"""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=QWEN_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.3,
        max_tokens=1024,
    )
    print(f"[OK] LLM 加载完成：{QWEN_MODEL}")
    print(f"     类型：{type(llm).__name__}，是 Runnable: {hasattr(llm, 'invoke')}")
    return llm


# ═══════════════════════════════════════════════════════════════════════════════
# 原理 2：PromptTemplate 也是 Runnable
# ═══════════════════════════════════════════════════════════════════════════════
#
# ChatPromptTemplate.invoke({"context": "...", "question": "..."})
#   → ChatPromptValue（包含格式化后的 messages 列表）
#
# 它不是简单的 f-string！它是一个 Runnable，可以参与 | 管道。
# 面试点：PromptTemplate 的 input_variables 自动校验变量名，
#         还支持 partial_variables（部分预填充）。

def get_prompt():
    """构建 Prompt Template（Runnable 协议）"""
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "你是一位专业的泌尿科医学助手。"
         "请根据以下参考资料回答用户的问题。"
         "回答要求：专业、准确、简洁。"
         "如果资料中没有相关内容请如实告知。"),
        ("human", 
         "参考资料：\n{context}\n\n"
         "用户问题：{question}\n\n"
         "请用中文回答："),
    ])
    print(f"[OK] PromptTemplate 构建完成")
    print(f"     输入变量：{prompt.input_variables}")
    print(f"     是 Runnable: {hasattr(prompt, 'invoke')}")
    return prompt


# ═══════════════════════════════════════════════════════════════════════════════
# 原理 3：Retriever 是 Runnable
# ═══════════════════════════════════════════════════════════════════════════════
#
# vectorstore.as_retriever() 返回 VectorStoreRetriever，
# 它继承自 BaseRetriever → Runnable。
#
# retriever.invoke("查询文本") → List[Document]
#
# 面试点：Retriever 和 VectorStore 的区别
#   - VectorStore：存储层，负责 add_documents / similarity_search
#   - Retriever：检索层，是 Runnable，支持 invoke/stream/batch
#   - as_retriever() 是适配器模式，把 VectorStore 包装成 Runnable

def get_retriever():
    """获取 Retriever（Runnable 协议）"""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_postgres import PGVector

    print(f"[INFO] 加载向量模型：{BGE_MODEL_NAME} ...")
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

    # as_retriever() → VectorStoreRetriever（Runnable）
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.3},
    )
    print(f"[OK] Retriever 加载完成")
    print(f"     类型：{type(retriever).__name__}，是 Runnable: {hasattr(retriever, 'invoke')}")
    return retriever


# ═══════════════════════════════════════════════════════════════════════════════
# 原理 4：LCEL 管道 —— 用 | 组合 Runnable
# ═══════════════════════════════════════════════════════════════════════════════
#
# 核心链路（对比你原来的 rag_pipeline.py 手动拼接）：
#
#   原来的写法：
#     results = retriever.invoke(query)       # 手动调用
#     context = "\n".join(...)                 # 手动拼接
#     prompt = f"...{context}...{query}..."    # 手动格式化
#     response = openai_client.create(...)     # 手动调用 API
#
#   LCEL 写法：
#     chain = (
#       {"context": retriever | format_docs, "question": RunnablePassthrough()}
#       | prompt
#       | llm
#       | StrOutputParser()
#     )
#     answer = chain.invoke("前列腺炎如何治疗？")  # 一行搞定！
#
# 面试重点：这条链的数据流
#   1. RunnableParallel 并行执行：
#      - retriever.invoke("前列腺炎如何治疗？") → List[Document] → format_docs → str
#      - RunnablePassthrough().invoke("前列腺炎如何治疗？") → "前列腺炎如何治疗？"
#      → 合并为 {"context": "...", "question": "前列腺炎如何治疗？"}
#   2. prompt.invoke({"context": "...", "question": "..."}) → ChatPromptValue
#   3. llm.invoke(ChatPromptValue) → AIMessage
#   4. StrOutputParser().invoke(AIMessage) → str（最终答案）

def build_rag_chain():
    """构建 LCEL RAG 链 —— 面试重点代码"""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    retriever = get_retriever()
    prompt    = get_prompt()
    llm       = get_llm()

    # 格式化检索结果的辅助函数
    def format_docs(docs):
        """List[Document] → str，将检索结果格式化为 prompt 可用的文本"""
        return "\n\n---\n\n".join(
            f"【来源：{Path(doc.metadata.get('source', '未知')).name} "
            f"第{doc.metadata.get('page', '?')}页】\n{doc.page_content}"
            for doc in docs
        )

    # ┌─────────────────────────────────────────────────────────────────┐
    # │  LCEL 核心链路 —— 这是面试必须能手写的代码                      │
    # │                                                                 │
    # │  数据流：                                                       │
    # │  "query" → RunnableParallel → dict → prompt → llm → parser     │
    # │              ├→ retriever | format_docs → context               │
    # │              └→ RunnablePassthrough()   → question              │
    # └─────────────────────────────────────────────────────────────────┘
    rag_chain = (
        {
            # RunnableParallel: 并行获取 context 和 question
            "context":  retriever | format_docs,    # retriever 是 Runnable，format_docs 被自动包装为 RunnableLambda
            "question": RunnablePassthrough(),       # 原样传递输入（query string）
        }
        | prompt        # PromptTemplate: dict → ChatPromptValue
        | llm           # ChatOpenAI: ChatPromptValue → AIMessage
        | StrOutputParser()  # StrOutputParser: AIMessage → str
    )

    print(f"\n[OK] RAG Chain 构建完成")
    print(f"     链类型：{type(rag_chain).__name__}")
    return rag_chain


# ═══════════════════════════════════════════════════════════════════════════════
# 原理 5：invoke vs stream 的区别
# ═══════════════════════════════════════════════════════════════════════════════
#
# invoke(): 等待全部完成后一次性返回结果
# stream(): 逐 token 返回结果（只流式传输最后一个 step 的输出）
#
# 面试点：stream() 内部实现
#   RunnableSequence.stream() 会对前面的 step 执行 invoke()，
#   只对最后一个 step 执行 stream()。
#   但如果用 astream_events()，可以获取每个 step 的事件流。

def ask_invoke(chain, query: str):
    """方式1：invoke —— 一次性返回"""
    print(f"\n[QUERY] {query}")
    print("[MODE] invoke（一次性返回）\n")

    answer = chain.invoke(query)

    print("=" * 60)
    print(answer)
    print("=" * 60)
    return answer


def ask_stream(chain, query: str):
    """方式2：stream —— 逐 token 输出（面试常问）"""
    print(f"\n[QUERY] {query}")
    print("[MODE] stream（流式输出）\n")
    print("=" * 60)

    full_answer = ""
    for chunk in chain.stream(query):
        print(chunk, end="", flush=True)  # 实时打印每个 token
        full_answer += chunk

    print(f"\n{'=' * 60}")
    return full_answer


# ─── 演示入口 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  LCEL RAG Chain 演示 —— 用 | 管道重构 RAG 管道")
    print("=" * 70)

    chain = build_rag_chain()

    query = sys.argv[1] if len(sys.argv) > 1 else "前列腺炎的诊断标准是什么？"

    # 演示两种调用方式
    print("\n" + "─" * 70)
    print("  演示 1：invoke 模式")
    print("─" * 70)
    ask_invoke(chain, query)

    print("\n" + "─" * 70)
    print("  演示 2：stream 模式（逐 token 输出）")
    print("─" * 70)
    ask_stream(chain, query)
