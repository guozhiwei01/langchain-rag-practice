# -*- coding: utf-8 -*-
"""
统一配置模块
============
从 .env 加载配置，提供在线大模型 LLM 和本地高效 FAISS 向量库的工厂函数。
如果本地没有 FAISS 索引，会在首次调用时自动读取 pdf/ 目录下的 PDF 文件并构建本地索引，
通过阿里 DashScope 官方 Embedding REST 接口，实现 100% 极速、轻量、高可用启动。
"""

import os
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

# 加载项目根目录的 .env
load_dotenv(Path(__file__).parent.parent / ".env")

# ─── 模型与 API 配置 ───────────────────────────────────────────────────────────
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
QWEN_MODEL       = os.getenv("QWEN_MODEL", "qwen-plus")

# ─── Langfuse 可观测性配置 ───────────────────────────────────────────────────────
# 未配置时自动跳过，不影响系统正常运行
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST       = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")  # 生产环境换成内网地址

# ─── 向量库本地保存路径 ─────────────────────────────────────────────────────────
FAISS_INDEX_DIR = Path(__file__).parent / "faiss_index"
PDF_DIR         = Path(__file__).parent.parent / "pdf"


class DashScopeApiEmbeddings(Embeddings):
    """
    极度轻量、高可控的阿里 DashScope 在线 Embedding 实现。
    直接用 requests 调用官方 API，支持分批请求（单次最多 25 个文本），
    完全绕过第三方库的协议解析冲突与单包限制。
    """
    def __init__(self, api_key: str, model: str = "text-embedding-v2"):
        self.api_key = api_key
        self.model = model
        self.url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"

    def _embed(self, texts: list) -> list:
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 官方接口单次最多支持 25 个文本
        batch_size = 25
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[i:i+batch_size]]
            payload = {
                "model": self.model,
                "input": {
                    "texts": batch
                }
            }
            
            response = requests.post(self.url, headers=headers, json=payload, timeout=20)
            if response.status_code != 200:
                raise ValueError(f"阿里 Embedding 接口调用失败 ({response.status_code}): {response.text}")
                
            data = response.json()
            if "output" not in data or "embeddings" not in data["output"]:
                raise ValueError(f"阿里 Embedding 接口返回数据格式错误: {data}")
                
            # 提取向量列表
            for item in data["output"]["embeddings"]:
                results.append(item["embedding"])
                
        return results

    def embed_documents(self, texts: list) -> list:
        return self._embed(texts)

    def embed_query(self, text: str) -> list:
        return self._embed([text])[0]

    def __call__(self, text: str) -> list:
        return self.embed_query(text)


def get_langfuse_handler():
    """
    获取 Langfuse CallbackHandler 实例。
    如果未配置 Langfuse API Key，返回 None（优雅降级）。
    上生产私有化部署时，只需把 LANGFUSE_HOST 改为内网地址即可。

    Langfuse v4 通过环境变量自动读取配置：
      LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST
    """
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        return None

    # v4 通过环境变量自动配置，确保环境变量已设置
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", LANGFUSE_PUBLIC_KEY)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", LANGFUSE_SECRET_KEY)
    os.environ.setdefault("LANGFUSE_HOST", LANGFUSE_HOST)

    from langfuse.langchain import CallbackHandler
    handler = CallbackHandler()
    print(f"[Langfuse] ✅ 可观测性追踪已启用 → {LANGFUSE_HOST}")
    return handler


def get_llm(temperature: float = 0.3, max_tokens: int = 1024):
    """
    获取 ChatOpenAI 实例（Runnable 协议）。
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=QWEN_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=temperature,
        max_tokens=max_tokens,
    )


@lru_cache(maxsize=1)
def get_embeddings():
    """
    获取自定义阿里官方 API Embedding 实例。
    """
    return DashScopeApiEmbeddings(api_key=DASHSCOPE_API_KEY)


def get_vectorstore():
    """
    获取 FAISS 向量库实例。
    若本地无索引，会自动加载 pdf/*.pdf，切块并构建 FAISS 索引，保存到本地。
    """
    from langchain_community.vectorstores import FAISS

    embeddings = get_embeddings()

    if FAISS_INDEX_DIR.exists():
        # 本地已存在 FAISS 索引，直接加载
        return FAISS.load_local(
            str(FAISS_INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )

    # 首次运行，本地无索引，触发自动构建
    print("\n[INFO] 未检测到本地 FAISS 索引，开始自动构建...")
    
    # 1. 递归加载 PDF
    from langchain_community.document_loaders import PyPDFLoader
    docs = []
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"在 {PDF_DIR} 目录下没有找到任何 PDF 指南文件，请确认路径！")
        
    print(f"[INFO] 找到 {len(pdf_files)} 个 PDF 文件，开始加载：")
    for f in pdf_files:
        print(f"   - {f.name}")
        loader = PyPDFLoader(str(f))
        docs.extend(loader.load())
    print(f"[OK] 成功加载 {len(docs)} 页 PDF")

    # 2. 切块
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    
    # 极度强健的预处理与过滤：强行转为 str、去噪、防超长、防空值
    valid_chunks = []
    for c in chunks:
        if c.page_content is not None:
            content_str = str(c.page_content).strip()
            if len(content_str) > 5:
                c.page_content = content_str[:1500]
                valid_chunks.append(c)
                
    chunks = valid_chunks
    print(f"[OK] 切块完成：过滤后共生成 {len(chunks)} 个高品质有效 chunks")

    # 3. 在线 Embedding 并存入 FAISS 向量库
    print("[INFO] 正在调用 DashScope 接口进行在线 Embedding 并构建 FAISS 索引（分批请求，预计需要数秒）...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # 4. 保存到本地
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_INDEX_DIR))
    print(f"[OK] FAISS 索引已构建并成功保存至本地：{FAISS_INDEX_DIR}\n")
    
    return vectorstore
