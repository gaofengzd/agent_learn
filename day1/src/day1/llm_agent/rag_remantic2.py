from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from typing import List
from loguru import logger
from src.day1.config import AppConfig, BASE_DIR
from openai import OpenAI
import torch


config = AppConfig()
model = config.llm_model
embed_model_name = config.EMBED_MODEL_NAME
chroma_persist_path = config.chroma_persist_path
collection_name = config.COLLECTION_NAME

llm_client = OpenAI(
    api_key=config.api_key,
    base_url=config.api_base_url
)

# 配置日志文件：每天零点生成新文件，保留7天日志，只记录INFO及以上级别
logger.add(
    sink="logs/rag_remantic1{time:YYYY-MM-DD}.log",  # 日志文件路径，{time}会自动替换成日期
    rotation="00:00",  # 轮转规则：每天0点生成新文件
    retention="7 days",  # 保留规则：只保留最近7天的日志
    compression="zip",  # 旧日志自动压缩成zip，节省空间
    level="INFO",  # 只记录INFO及以上级别的日志
    encoding="utf-8",  # 编码，避免中文乱码
    enqueue=True  # 异步写入，多线程/异步场景不阻塞
)

# ====================== 配置 ======================
LOCAL_BGE_EMBED_PATH = r"E:\00project\02agent\models\bge-large-zh-v1.5"  # 向量化模型
LOCAL_BGE_RERANK_PATH = r"E:\00project\02agent\models\bge-reranker-v2-m3"  # 重排模型
PDF_FILE = "../documents/test.pdf"
CHROMA_SAVE_PATH = "../../chroma_semantic_db"
COLLECTION_NAME = "rag_docs"

SPLIT_TYPE = "percentile"
SPLIT_THRESHOLD = 92
RECALL_TOP_K = 5
FINAL_TOP_K = 3
# =================================================
def get_device():
    """检测CUDA，有则返回'cuda'，否则'cpu'"""
    if torch.cuda.is_available():
        print(f"✅ 使用 GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        print("ℹ️ 使用 CPU")
        return "cpu"


DEVICE = get_device()

class BGEEmbedding(HuggingFaceEmbeddings):
    """BGE embedding封装：查询自动添加前缀"""
    def embed_query(self, text: str):
        prefix = "为这个句子生成表示以用于检索相关文章："
        return super().embed_query(prefix + text)


def get_embedding():
    embeddings = BGEEmbedding(
        model_name=LOCAL_BGE_EMBED_PATH,
        model_kwargs={
            "device": DEVICE,
            "trust_remote_code": True
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )
    return embeddings


def get_reranker():
    """
    初始化本地Rerank模型（使用 LangChain 标准 CrossEncoderReranker）
    """
    # 1. 创建交叉编码器模型（底层使用 sentence-transformers）
    cross_encoder = HuggingFaceCrossEncoder(
        model_name=LOCAL_BGE_RERANK_PATH,
        model_kwargs={"device": DEVICE, "trust_remote_code": True}
    )
    # 2. 包装成 LangChain 的压缩器
    reranker = CrossEncoderReranker(
        model=cross_encoder,
        top_n=FINAL_TOP_K  # 最终返回的文档数量
    )
    return reranker


def build_vector_db():
    embedder = get_embedding()

    loader = PyMuPDFLoader(PDF_FILE)
    raw_docs = loader.load()
    print(f"原始文档页数：{len(raw_docs)}")

    splitter = SemanticChunker(
        embeddings=embedder,
        breakpoint_threshold_type=SPLIT_TYPE,
        breakpoint_threshold_amount=SPLIT_THRESHOLD
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"语义切片总数：{len(chunks)}")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=CHROMA_SAVE_PATH,
        collection_name=COLLECTION_NAME
    )
    print("✅ 向量数据库构建完成")
    return db


def load_vector_db():
    embedder = get_embedding()
    db = Chroma(
        persist_directory=CHROMA_SAVE_PATH,
        embedding_function=embedder,
        collection_name=COLLECTION_NAME
    )
    print("✅ 向量库加载成功")
    return db


def search_with_rerank(db, query: str):
    """
    完整检索链路：
    1. 相似度粗召回（RECALL_TOP_K 条）
    2. CrossEncoderReranker 重排序，输出 FINAL_TOP_K 条
    """
    reranker = get_reranker()

    # 向量粗召回
    raw_docs = db.similarity_search(query, k=RECALL_TOP_K)
    print(f"\n【向量粗召回共 {len(raw_docs)} 条，开始 Rerank 重排】")

    # 使用 compress_documents 进行重排（注意：不是 rerank 方法）
    ranked_docs = reranker.compress_documents(
        documents=raw_docs,
        query=query
    )
    # ranked_docs 已经是按相关性降序排列的列表，且数量 <= top_n

    print(f"\n================ 用户问题：{query} ================")
    for idx, doc in enumerate(ranked_docs):
        print(f"\n【重排结果 {idx+1}】")
        print("元信息(文档溯源):", doc.metadata)
        print("文本内容：")
        print(doc.page_content)

    return ranked_docs


def build_rag_prompt(user_query: str, context_chunks: list) -> str:
    """组装RAG提示词"""
    # 提取每个 Document 对象的 page_content 文本
    text_list = [doc.page_content for doc in context_chunks]

    # 使用纯文本列表进行拼接
    context_text = "\n---\n".join(text_list)

    prompt = f"""
你是知识库问答助手，请严格依据下面【知识库内容】回答用户问题。
如果知识库没有相关信息，直接回答“知识库中未查询到相关内容”，禁止编造信息。

【知识库内容】
{context_text}

【用户问题】
{user_query}
"""
    return prompt


def rag_chat(db, user_query: str, stream: bool = True):
    """完整RAG链路：检索 + LLM生成"""
    # 1.向量检索
    context = search_with_rerank(db, user_query)

    # 2.构造prompt
    prompt = build_rag_prompt(user_query, context)
    messages = [
        {"role": "user", "content": prompt}
    ]
    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            temperature=0.3
        )
        if stream:
            logger.info("开始流式输出回答：")
            full_answer = ""
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_answer += delta
                    print(delta, end="")
            print("\n")
            return full_answer
        else:
            answer = response.choices[0].message.content
            print(answer)
            return answer
    except Exception as e:
        logger.error(f"LLM调用失败: {e}")
        return "服务调用异常"


if __name__ == "__main__":
    # 首次运行构建库
    # db = build_vector_db()
    # 后续运行直接加载，注释上行，启用下面
    db = load_vector_db()

    # ========== RAG问答循环 ==========
    logger.info("===== RAG命令行问答启动，输入exit退出 =====")
    while True:
        question = input("\n请输入你的问题：")
        if question.strip().lower() == "exit":
            logger.info("程序退出")
            break
        rag_chat(db, question)
# 实事求是的思想路线是什么