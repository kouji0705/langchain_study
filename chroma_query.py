# chroma_query.py
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY が設定されていません")

PERSIST_DIR = "chroma_db_example"

# Embedding モデル（検索時も同じモデルを使う必要あり）
emb = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

# 既に保存済みの Chroma DB を開く
db = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name="demo_collection",
    embedding_function=emb,
)

print("Chroma DB をロードしました。")

while True:
    query = input("\n質問を入力してください（空Enterで終了）> ").strip()
    if not query:
        print("終了します。")
        break

    docs = db.similarity_search(query, k=3)

    print("\n🔎 類似文書 Top3:")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.page_content}")
