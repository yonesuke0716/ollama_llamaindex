import os
import json
from ollama import generate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_cloud_services import LlamaParse
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from llama_index.core import StorageContext
from dotenv import load_dotenv
import duckdb


load_dotenv()

Settings.llm = None

persist_dir = "./persist/"
os.makedirs(persist_dir, exist_ok=True)
# Embedding
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# "./persist/class.duckdb"にテーブルが存在するか確認
if "class.duckdb" not in os.listdir("./persist"):
    # DuckDBVectorStore の作成
    vector_store = DuckDBVectorStore(
        database_name="class.duckdb", persist_dir=persist_dir
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # PDFファイルのパース
    parser = LlamaParse(result_type="markdown")  # "markdown" and "text" are available
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(
        "./src/class_blog", file_extractor=file_extractor
    ).load_data()

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )
else:
    # DuckDBの中身を確認
    conn = duckdb.connect("./persist/class.duckdb")
    df = conn.table("documents").df()
    df["metadata_"] = df["metadata_"].apply(json.loads)
    # df["metadata_"]から"file_path"キーのユニークな値を取得
    filepaths = df["metadata_"].apply(lambda x: x["file_path"]).unique()
    # 後ろから"/"までの文字列を取得
    filenames = [filepath.split("/")[-1] for filepath in filepaths]
    # src/class_blogに含まれるファイル名のリストを取得
    filenames_in_dir = os.listdir("./src/class_blog")
    # .gitignoreに含まれるファイル名を除外
    filenames_in_dir = [
        filename for filename in filenames_in_dir if filename != ".gitignore"
    ]
    # ファイル名のリストを比較
    new_files = set(filenames_in_dir) - set(filenames)
    # 値にsrc/class_blogを追加
    new_files = [os.path.join("./src/class_blog", file) for file in new_files]

    # Load from disk
    vector_store = DuckDBVectorStore.from_local("./persist/class.duckdb")
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    if new_files:
        print(f"新しいファイル{len(new_files)}個を追加します")
        parser = LlamaParse(
            result_type="markdown"  # "markdown" and "text" are available
        )
        file_extractor = {".pdf": parser}
        documents = SimpleDirectoryReader(
            input_files=list(new_files), file_extractor=file_extractor
        ).load_data()
        parser = SentenceSplitter()
        nodes = parser.get_nodes_from_documents(documents)
        # class.duckdb更新
        index.insert_nodes(nodes)
    else:
        print("新しいファイルはありません")
    conn.close()


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
    response = generate(
        model="llama3.1", prompt=prompt, options={"temperature": 0, "top_p": 0.9}
    )
    return response["response"]


# クエリの実行
response = query_ollama(
    "優れた教育者になるためのスキルと自信が重要な理由について教えてください"
)
print(response)
