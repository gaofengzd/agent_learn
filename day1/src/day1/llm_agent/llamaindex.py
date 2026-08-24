from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from src.day1.config import AppConfig
from llama_index.llms.zhipuai import ZhipuAI



# ================== 配置本地BGE模型 ==================
# 替换成你电脑上的模型本地路径
config = AppConfig()
LOCAL_BGE_PATH = r"E:\00project\02agent\models\bge-large-zh-v1.5"

Settings.embed_model = HuggingFaceEmbedding(
    model_name=LOCAL_BGE_PATH,
    device="cpu",          # 无显卡改为 "cpu"
    normalize=True,         # BGE必须开启向量归一化
    # BGE专属：查询自动加前缀，检索效果关键！
    query_instruction="为这个句子生成表示以用于检索相关文章："
)

# 配置大模型，替换 base_url、model、api_key
# Settings.llm = OpenAI(
#     api_key=config.api_key,
#     # api_base=config.api_base_url,   # deepseek；智谱填 https://open.bigmodel.cn/api/paas/v4/
#     api_base="https://open.bigmodel.cn/api/paas/v4/",   # deepseek；智谱填 https://open.bigmodel.cn/api/paas/v4/
#     # model=config.llm_model,
#     model="glm-4.7",
#     temperature=0.7,
#     context_window=128000   # 重点！手动指定上下文窗口，绕过查表报错
#     # max_tokens=2048
# )

Settings.llm = ZhipuAI(
    api_key=config.api_key,
    model="glm-4",
    temperature=0.7
)

# 可选：自定义文本切分参数（默认chunk_size=1024）
Settings.chunk_size = 500
Settings.chunk_overlap = 80

documents = SimpleDirectoryReader(
    "../documents"
).load_data()

index = VectorStoreIndex.from_documents(
    documents
)

query_engine = index.as_query_engine(
    similarity_top_k=3
)

response = query_engine.query("什么是rag？")

print(response)