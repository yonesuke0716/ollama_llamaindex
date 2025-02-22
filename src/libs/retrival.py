# DuckDBファイルのパス
def add_doc(duckdb_path="vector_store.duckdb", embed_model="nomic-embed-text") -> None:
    # DuckDBの接続を作成
    con = duckdb.connect(database=duckdb_path, read_only=False)
    # DuckDBVectorStore の作成
    vector_store = DuckDBVectorStore(database_path=duckdb_path)
    # 既存のテーブルがあるか確認
    table_exists = con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'documents'").fetchone()[0] > 0

    if table_exists:
        # 既存のデータを利用してStorageContextを作成
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = load_index_from_storage(storage_context)

        # すでに登録されているドキュメントのID一覧を取得
        existing_doc_ids = set(row[0] for row in con.execute("SELECT doc_id FROM documents").fetchall())
    else:
        # 新規のStorageContextを作成
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = None  # まだ何もない状態
        existing_doc_ids = set()

    # PDFファイルのパース
    parser = LlamaParse(
        result_type="markdown"  # "markdown" and "text" are available
    )
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader("./src/class_blog", file_extractor=file_extractor).load_data()
    for doc in documents:
        doc.metadata["file_name"] = doc.metadata.get("file_name", "unknown")  # ファイル名を取得
        doc.doc_id = doc.metadata["file_name"]  # doc_id にファイル名を設定
    embed_model = OllamaEmbedding(model_name=embed_model)
    # 既存のDBにないドキュメントだけを追加
    new_documents = [doc for doc in documents if doc.doc_id not in existing_doc_ids]
    # Embedding
    if new_documents:
        if index is None:
            # 新規作成する場合
            index = VectorStoreIndex.from_documents(new_documents, storage_context=storage_context, embed_model=embed_model)
        else:
            # 既存のIndexに追加
            index.insert_documents(new_documents)

        print(f"追加したドキュメント数: {len(new_documents)}")
    else:
        print("新しいドキュメントはありません。")

    # DuckDBの接続を閉じる
    con.close()