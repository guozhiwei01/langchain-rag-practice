# -*- coding: utf-8 -*-
"""
RAG 检索工具定义
================
所有子智能体可调用的工具函数。
三个工具本质都走同一个 pgvector 向量库，通过不同的 docstring 引导 LLM 选择。
"""

from pathlib import Path
from langchain_core.tools import tool
from multi_agent.config import get_vectorstore


def _format_search_results(docs: list) -> str:
    """将检索到的 Document 列表格式化为带来源标注的文本。"""
    if not docs:
        return "未检索到相关内容。"
    return "\n\n---\n\n".join(
        f"【来源：{Path(doc.metadata.get('source', '未知')).name} "
        f"第{doc.metadata.get('page', '?')}页】\n{doc.page_content}"
        for doc in docs
    )


@tool
def search_guidelines(query: str) -> str:
    """从前列腺相关临床指南中检索与查询最相关的段落。
    当需要获取疾病概述、诊断标准、治疗方案、生活方式建议等综合信息时使用此工具。
    适用于科普宣教、一般性医学问题解答等场景。"""
    vs = get_vectorstore()
    docs = vs.similarity_search(query, k=3)
    return _format_search_results(docs)


@tool
def search_drug_info(query: str) -> str:
    """从前列腺相关临床指南中检索药物治疗相关的段落。
    当用户询问药物名称、用法用量、给药方案、药物副作用、不良反应、
    药物相互作用、配伍禁忌、停药注意事项时，使用此工具。
    检索时会自动增强查询，聚焦于药物治疗相关内容。"""
    # 增强查询：在原始 query 前追加药物语境关键词，提升药物相关 chunk 的召回率
    enhanced_query = f"药物治疗 用药方案 剂量 副作用 禁忌 {query}"
    vs = get_vectorstore()
    docs = vs.similarity_search(enhanced_query, k=3)
    return _format_search_results(docs)


@tool
def search_risk_indicators(query: str) -> str:
    """从前列腺相关临床指南中检索诊断标准、检验指标参考值、危急值、
    分级分型标准相关的段落。
    当用户提供了具体的检验指标数值（如 PSA、IPSS 评分），
    或需要判断某个指标是否正常、是否达到手术指征时，使用此工具。"""
    # 增强查询：追加诊断与指标相关关键词
    enhanced_query = f"诊断标准 参考值 指标 分级 评分 {query}"
    vs = get_vectorstore()
    docs = vs.similarity_search(enhanced_query, k=3)
    return _format_search_results(docs)
