from pathlib import Path
import hashlib
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch

LOCAL_BGE_EMBED_PATH = r"E:\00project\02agent\models\bge-large-zh-v1.5"  # 向量化模型
LOCAL_BGE_RERANK_PATH = r"E:\00project\02agent\models\bge-reranker-v2-m3"  # 重排模型
CHROMA_SAVE_PATH = "../langchain_chroma_db"
PDF_FILE = "../documents/test.pdf"

def get_device():
    """检测CUDA，有则返回'cuda'，否则'cpu'"""
    if torch.cuda.is_available():
        print(f"✅ 使用 GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        print("ℹ️ 使用 CPU")
        return "cpu"
DEVICE = get_device()

def load_markdown(path: str) -> list[Document]:
    loader = TextLoader(
        path,
        encoding="utf-8",
    )

    documents = loader.load()
    print(documents)

    for document in documents:
        document.metadata.update(
            {
                "source": str(Path(path).resolve()),
                "file_type": "markdown",
            }
        )

    return documents


def load_pdf(path: str) -> list[Document]:
    loader = PyPDFLoader(path)
    documents = loader.load()
    print(documents)

    for document in documents:
        page = document.metadata.get("page")

        document.metadata.update(
            {
                "source": str(Path(path).resolve()),
                "file_type": "pdf",
                "page_label": page + 1 if page is not None else None,
            }
        )

    return documents


def load_web(url: str) -> list[Document]:
    loader = WebBaseLoader(
        web_paths=(url,)
    )

    documents = loader.load()
    print(documents)

    for document in documents:
        document.metadata.update(
            {
                "source": url,
                "url": url,
                "file_type": "webpage",
            }
        )

    return documents


def add_chunk_metadata(
    documents: list[Document],
) -> list[Document]:
    for index, document in enumerate(documents):
        source = document.metadata.get(
            "source",
            "unknown",
        )

        document.metadata["chunk_id"] = (
            f"{source}::{index}"
        )

    return documents


def make_document_id(
    document: Document,
    index: int,
) -> str:
    source = document.metadata.get(
        "source",
        "unknown",
    )

    content_hash = hashlib.md5(
        document.page_content.encode("utf-8")
    ).hexdigest()[:12]

    return f"{source}::{index}::{content_hash}"

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


def pre_make():
    documents = []

    documents.extend(
        load_markdown("../documents/rag_notes.md")
    )

    documents.extend(
        load_pdf("../documents/example.pdf")
    )

    documents.extend(
        load_web("https://zhuanlan.zhihu.com/p/1999160608529072176")
    )

    print(f"原始 Document 数量：{len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            "，",
            "、",
            " ",
            "",
        ],
        chunk_size=500,
        chunk_overlap=80,
        length_function=len,
    )

    chunks = text_splitter.split_documents(
        documents
    )

    chunks = add_chunk_metadata(chunks)

    print(f"切分后 chunk 数量：{len(chunks)}")

    embeddings = get_embedding()

    vector_store = Chroma(
        collection_name="langchain_knowledge_base",
        embedding_function=embeddings,
        persist_directory=CHROMA_SAVE_PATH,
    )

    document_ids = [
        make_document_id(document, index)
        for index, document in enumerate(chunks)
    ]

    vector_store.add_documents(
        documents=chunks,
        ids=document_ids,
    )

    print("知识库构建完成。")

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 12,
        },
    )

    question = "什么是 RAG？"

    results = retriever.invoke(question)

    for index, document in enumerate(
        results,
        start=1,
    ):
        print(f"\n--- 结果 {index} ---")
        print("来源：", document.metadata)
        print("内容：")
        print(document.page_content[:500])

def main():
    embeddings = get_embedding()

    vector_store = Chroma(
        collection_name="langchain_knowledge_base",
        embedding_function=embeddings,
        persist_directory=CHROMA_SAVE_PATH,
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 12,
        },
    )

    question = "实事求是路线是什么"

    results = retriever.invoke(question)

    for index, document in enumerate(
            results,
            start=1,
    ):
        print(f"\n--- 结果 {index} ---")
        print("来源：", document.metadata)
        print("内容：")
        print(document.page_content[:500])


if __name__ == "__main__":
    main()