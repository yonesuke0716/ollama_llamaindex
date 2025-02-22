import os
from ollama import generate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_cloud_services import LlamaParse
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from llama_index.core import StorageContext, load_index_from_storage
from dotenv import load_dotenv
import duckdb


def list_tables(database_path: str):
    conn = duckdb.connect(database_path)
    tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
    conn.close()
    return [table[0] for table in tables]


# print(list_tables("./persist/class.duckdb"))
# breakpoint()
load_dotenv()

Settings.llm = None

persist_dir = "./persist/"
os.makedirs(persist_dir, exist_ok=True)

# DuckDBVectorStore の作成
vector_store = DuckDBVectorStore(database_name="class.duckdb", persist_dir=persist_dir)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# PDFファイルのパース
parser = LlamaParse(
    result_type="markdown"  # "markdown" and "text" are available
)
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./src/class_blog", file_extractor=file_extractor).load_data()

# Embedding
embed_model = OllamaEmbedding(model_name="nomic-embed-text")
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

retriever = VectorIndexRetriever(index=index, similarity_top_k=3)  # 上位3件を取得
query_engine = RetrieverQueryEngine(retriever)

def query_ollama(query: str) -> str:
    retrieved_docs = query_engine.retrieve(query)
    context = "\n".join([doc.text for doc in retrieved_docs])
    # print(context)
    prompt = f"""以下の文脈を元に質問に答えてください:

{context}


質問: {query}
"""
    response = generate(model="llama3.1", prompt=prompt, options={"temperature": 0, "top_p": 0.9})
    return response["response"]

# クエリの実行
response = query_ollama("優れた教育者になるためのスキルと自信が重要な理由について教えてください")
print(response)
