from typing import List
from loguru import logger
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI
from src.day1.config import AppConfig, BASE_DIR


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
    sink="logs/rag_{time:YYYY-MM-DD}.log",  # 日志文件路径，{time}会自动替换成日期
    rotation="00:00",  # 轮转规则：每天0点生成新文件
    retention="7 days",  # 保留规则：只保留最近7天的日志
    compression="zip",  # 旧日志自动压缩成zip，节省空间
    level="INFO",  # 只记录INFO及以上级别的日志
    encoding="utf-8",  # 编码，避免中文乱码
    enqueue=True  # 异步写入，多线程/异步场景不阻塞
)

# -------------------------- 初始化向量数据库 --------------------------
# 本地持久化Chroma
chroma_client = chromadb.PersistentClient(path=chroma_persist_path)
embed_func = SentenceTransformerEmbeddingFunction(model_name=embed_model_name)

# 获取/创建向量集合
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=embed_func
)


# -------------------------- 工具函数 --------------------------
def load_txt_file(file_path: str) -> str:
    """加载txt文档"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.success(f"文档加载成功：{file_path}")
        return content
    except Exception as e:
        logger.error(f"文档读取失败 {file_path}: {e}")
        return ""


def split_text(text: str, chunk_size: int = 400, chunk_overlap: int = 80) -> List[str]:
    """简单文本切片（演示用；生产推荐 LangChain RecursiveCharacterTextSplitter）"""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - chunk_overlap)
    logger.info(f"文本切分完成，共 {len(chunks)} 个片段")
    return chunks


def add_documents_to_vector_db(chunks: List[str]):
    """把文本片段存入向量库，自动向量化"""
    # 生成唯一id
    ids = [f"doc_{i}" for i in range(len(chunks))]
    try:
        collection.add(
            documents=chunks,
            ids=ids
        )
        logger.success(f"{len(chunks)} 条文本写入向量数据库完成")
    except Exception as e:
        logger.error(f"写入向量库异常：{e}")


def search_vector_db(query: str, top_n: int = 3) -> List[str]:
    """检索向量库，返回相似度最高的文本片段"""
    result = collection.query(
        query_texts=[query],
        n_results=top_n
    )
    # result["documents"] -> [[片段1,片段2,片段3]]
    hit_chunks = result["documents"][0]
    logger.info(f"检索到 {len(hit_chunks)} 条相关知识库内容")
    return hit_chunks


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


def rag_chat(user_query: str, stream: bool = True):
    """完整RAG链路：检索 + LLM生成"""
    # 1.向量检索
    context = search_vector_db(user_query)
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

# -------------------------- 主入口 --------------------------
if __name__ == "__main__":
    # ========== 第一次运行执行：加载文档、切片、存入向量库 ==========
    doc_content = load_txt_file(BASE_DIR/"./src/day1/documents/test.txt")
    text_chunks = split_text(doc_content)
    add_documents_to_vector_db(text_chunks)

    # ========== RAG问答循环 ==========
    logger.info("===== RAG命令行问答启动，输入exit退出 =====")
    while True:
        question = input("\n请输入你的问题：")
        if question.strip().lower() == "exit":
            logger.info("程序退出")
            break
        rag_chat(question)