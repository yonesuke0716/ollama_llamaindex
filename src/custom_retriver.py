from tomllib import loads
from ollama import generate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

# from libs.llms import analyze_image, translate_to_japanese

# with open("params.toml", encoding="utf-8") as fp:
#     params = loads(fp.read())

Settings.llm = None

documents = SimpleDirectoryReader("./data").load_data()
embed_model = OllamaEmbedding(model_name="nomic-embed-text")
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
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
    response = generate(model="llama3.2-vision:11b", prompt=prompt, images=["./images/image01.jpg"], options={"temperature": 0, "top_p": 0.9})  # Ollama で回答生成
    return response["response"]

# クエリの実行
response = query_ollama("上記の文脈から名前と年齢と性別を抽出し、写真の状況を整理してください。")
print(response)
