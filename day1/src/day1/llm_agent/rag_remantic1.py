import torch
from FlagEmbedding import FlagModel, FlagReranker
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from typing import List
from loguru import logger
from src.day1.config import AppConfig, BASE_DIR
from openai import OpenAI


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

# ====================== 【请修改此处配置】 ======================
LOCAL_BGE_EMBED_PATH = r"E:\00project\02agent\models\bge-large-zh-v1.5"  # 向量化模型
LOCAL_BGE_RERANK_PATH = r"E:\00project\02agent\models\bge-reranker-v2-m3"  # 重排模型
PDF_FILE = "../documents/test.pdf"
CHROMA_SAVE_PATH = "../../chroma_semantic_db"
COLLECTION_NAME = "rag_docs"

# 语义切块参数
SPLIT_TYPE = "percentile"
SPLIT_THRESHOLD = 92

# 检索参数
RECALL_TOP_K = 5  # 向量先粗召回5条
FINAL_TOP_K = 3  # rerank之后最终保留3条


# =================================================================


# ================== 自动选择设备 ==================
def get_device():
    """检测CUDA，有则返回'cuda'，否则'cpu'"""
    if torch.cuda.is_available():
        print(f"✅ 使用 GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        print("ℹ️ 使用 CPU")
        return "cpu"


DEVICE = get_device()


# ================== FlagEmbedding 适配 LangChain Embeddings ==================
class FlagEmbeddingAdapter(Embeddings):
    """将 FlagModel 包装成 LangChain 的 Embeddings 接口，支持查询前缀"""

    def __init__(
            self,
            model_path: str,
            device: str = "cpu",
            normalize_embeddings: bool = True,
            query_instruction: str = "为这个句子生成表示以用于检索相关文章："
    ):
        self.model = FlagModel(
            model_path,
            query_instruction_for_retrieval=query_instruction,  # 自动添加查询前缀
            use_fp16=(device == "cuda"),  # GPU 用 fp16 加速，CPU 关闭
            normalize_embeddings=normalize_embeddings,
            device=device
        )

    def embed_query(self, text: str):
        """查询单条向量"""
        # FlagModel.encode(text) 传入单字符串时，返回一维数组 (dim,)
        vec = self.model.encode(text)

        # 做个兼容性保护：如果是二维 (1, dim) 取 [0]，如果是一维 (dim,) 直接转
        if len(vec.shape) == 2:
            return vec[0].tolist()

        return vec.tolist()

    def embed_documents(self, texts: list):
        """文档批量向量化，用于构建索引"""
        # 返回 shape (len(texts), dim) 的列表
        return self.model.encode(texts).tolist()


# ================== 初始化模型 ==================
def get_embedding():
    """返回适配了 LangChain 的 embedding 对象"""
    return FlagEmbeddingAdapter(
        model_path=LOCAL_BGE_EMBED_PATH,
        device=DEVICE,
        normalize_embeddings=True
    )


def get_reranker():
    """初始化 FlagReranker"""
    use_fp16 = (DEVICE == "cuda")
    reranker = FlagReranker(
        LOCAL_BGE_RERANK_PATH,
        use_fp16=use_fp16,
        device=DEVICE
    )
    print(f"✅ Reranker 加载完成 (设备: {DEVICE}, fp16: {use_fp16})")
    return reranker


# ================== 向量库构建与加载 ==================
def build_vector_db():
    """PDF加载 → 语义切分 → 存入Chroma"""
    embedder = get_embedding()

    loader = PyMuPDFLoader(PDF_FILE)
    raw_docs = loader.load()
    print(f"原始文档页数：{len(raw_docs)}")

    # 语义切块
    splitter = SemanticChunker(
        embeddings=embedder,
        breakpoint_threshold_type=SPLIT_TYPE,
        breakpoint_threshold_amount=SPLIT_THRESHOLD
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"语义切片总数：{len(chunks)}")

    # 持久化向量库
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=CHROMA_SAVE_PATH,
        collection_name=COLLECTION_NAME
    )
    print("✅ 向量数据库构建完成")
    return db


def load_vector_db():
    """加载已有向量库"""
    embedder = get_embedding()
    db = Chroma(
        persist_directory=CHROMA_SAVE_PATH,
        embedding_function=embedder,
        collection_name=COLLECTION_NAME
    )
    print("✅ 向量库加载成功")
    return db


# ================== 检索 + 重排 ==================
def search_with_rerank(db, query: str):
    """
    完整检索链路：
    1. 相似度粗召回
    2. FlagReranker 重排序
    """
    reranker = get_reranker()

    # 第一步：向量粗召回
    raw_docs = db.similarity_search(query, k=RECALL_TOP_K)
    print(f"\n【向量粗召回共 {len(raw_docs)} 条，开始 Rerank 重排】")

    # 第二步：构建句子对并计算相关性分数
    pairs = [[query, doc.page_content] for doc in raw_docs]
    scores = reranker.compute_score(pairs)  # 返回 list of scores

    # 按分数降序排列，取前 FINAL_TOP_K
    scored_docs = list(zip(raw_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    ranked_docs = [doc for doc, _ in scored_docs[:FINAL_TOP_K]]

    # 打印结果
    print(f"\n================ 用户问题：{query} ================")
    for idx, doc in enumerate(ranked_docs):
        print(f"\n【重排结果 {idx + 1}】")
        print("元信息(文档溯源):", doc.metadata)
        print("文本内容：")
        print(doc.page_content)

    return ranked_docs


def build_rag_prompt(user_query: str, context_chunks: List[str]) -> str:
    """组装RAG提示词"""
    context_text = "\n---\n".join(context_chunks)
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


# ================== 主程序 ==================
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