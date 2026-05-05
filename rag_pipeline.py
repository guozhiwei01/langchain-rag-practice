# -*- coding: utf-8 -*-
"""
RAG 完整链路实践 —— 前列腺指南智能问答
=========================================
数据流：PDF → 切块 → bge-m3 向量化 → pgvector 存储 → 语义检索 → 千问生成答案
"""

import io
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Windows 终端强制 UTF-8 输出，避免中文乱码
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

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

PDF_DIR        = Path(__file__).parent / "pdf"
COLLECTION_NAME = "prostate_guidelines"

# ─── 1. 文档加载 ──────────────────────────────────────────────────────────────
def load_documents():
    from langchain_community.document_loaders import PyPDFLoader

    docs = []
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    print(f"[INFO] 找到 {len(pdf_files)} 个 PDF 文件：")
    for f in pdf_files:
        print(f"   - {f.name}")
        loader = PyPDFLoader(str(f))
        docs.extend(loader.load())

    print(f"[OK] 共加载 {len(docs)} 页文档")
    return docs


# ─── 2. 文档切块 ──────────────────────────────────────────────────────────────
def split_documents(docs):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[OK] 切块完成：{len(chunks)} 个 chunks（chunk_size=500, overlap=50）")
    return chunks


# ─── 3. 向量化模型（bge-m3）────────────────────────────────────────────────────
def get_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings

    print(f"[INFO] 加载向量模型：{BGE_MODEL_NAME} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=BGE_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("[OK] 向量模型加载完成")
    return embeddings


# ─── 4. 存入 pgvector ─────────────────────────────────────────────────────────
def build_vectorstore(chunks, embeddings):
    from langchain_postgres import PGVector

    print("[INFO] 向量化并写入 pgvector ...")
    vectorstore = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection=PG_CONN_STR,
        pre_delete_collection=True,   # 每次重新索引时清空旧数据
    )
    print(f"[OK] 已将 {len(chunks)} 个 chunks 写入 pgvector（集合：{COLLECTION_NAME}）")
    return vectorstore


# ─── 5. 加载已有向量库（不重新索引时使用）────────────────────────────────────
def load_vectorstore(embeddings):
    from langchain_postgres import PGVector

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=PG_CONN_STR,
    )
    return vectorstore


# ─── 6. 检索 Top-K 相关段落 ───────────────────────────────────────────────────
def retrieve(vectorstore, query: str, k: int = 3, score_threshold: float = 0.3):
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold},
    )
    results = retriever.invoke(query)
    return results


# ─── 7. 千问生成答案 ───────────────────────────────────────────────────────────
def generate_answer(query: str, retrieved_docs: list) -> str:
    from openai import OpenAI

    if not retrieved_docs:
        return "⚠️ 未检索到相关内容，请换一个问题试试。"

    context = "\n\n---\n\n".join(
        [f"【来源：{doc.metadata.get('source', '未知')} 第{doc.metadata.get('page', '?')}页】\n{doc.page_content}"
         for doc in retrieved_docs]
    )

    prompt = f"""你是一位专业的泌尿科医学助手，请根据以下参考资料回答用户的问题。
回答要求：专业、准确、简洁，如果资料中没有相关内容请如实告知。

参考资料：
{context}

用户问题：{query}

请用中文回答："""

    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    response = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ─── 主流程 ───────────────────────────────────────────────────────────────────
def ingest():
    """一次性：加载 PDF → 切块 → 向量化 → 存入 pgvector"""
    docs   = load_documents()
    chunks = split_documents(docs)
    emb    = get_embeddings()
    build_vectorstore(chunks, emb)
    print("\n[DONE] 索引构建完成！可以开始问答了。")


def ask(query: str, k: int = 3):
    """查询：语义检索 + 千问生成"""
    emb         = get_embeddings()
    vs          = load_vectorstore(emb)

    print(f"\n[QUERY] {query}")
    results     = retrieve(vs, query, k=k)

    print(f"[RECALL] 召回 {len(results)} 个相关段落：")
    for i, doc in enumerate(results, 1):
        src  = Path(doc.metadata.get("source", "")).name
        page = doc.metadata.get("page", "?")
        print(f"\n  [{i}] 来源：{src}  第{page}页")
        print(f"      {doc.page_content[:150].strip()}...")

    print("\n[INFO] 生成答案中...")
    answer = generate_answer(query, results)
    print(f"\n{'='*60}")
    print(answer)
    print(f"{'='*60}\n")
    return answer


# ─── CLI 入口 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法：")
        print("  uv run rag_pipeline.py ingest          # 建立索引（首次运行）")
        print("  uv run rag_pipeline.py ask '你的问题'  # 问答")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "ingest":
        ingest()

    elif cmd == "ask":
        query = sys.argv[2] if len(sys.argv) > 2 else "前列腺炎如何治疗？"
        ask(query)

    else:
        print(f"未知命令：{cmd}")
