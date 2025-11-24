# pokemon_query.py
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. .env 読み込み
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# 2. Embedding モデル
emb = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

# 3. 永続化された DB を読み込む
PERSIST_DIR = "chroma_pokemon_30"

db = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name="pokemon_30",
    embedding_function=emb,
)

print("📁 Chroma データベース読み込み完了")

# 4. 質問を受け付ける
while True:
    q = input("\n質問 > ").strip()
    if not q:
        print("終了します。")
        break

    # 意味検索（k=3）
    docs = db.similarity_search(q, k=3)

    print("\n🔎 類似ポケモン Top3:")
    for i, doc in enumerate(docs, start=1):
        print(f"{i}. {doc.page_content}")