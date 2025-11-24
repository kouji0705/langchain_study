# pokemon_chroma_store.py
import csv
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. .env の読み込み
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY が設定されていません")

# 2. Embedding モデル（OpenAI：text-embedding-3-small）
emb = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

# 3. CSV の読み込み
csv_path = "pokemon_zukan_30.csv"
texts = []

with open(csv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # RAG に使いやすいよう1行にまとめる
        content = f"{row['name_jp']}（{row['name_en']}）: タイプ={row['type1']} {row['type2']}。説明: {row['description']}"
        texts.append(content)

print(f"CSV 読み込み完了：{len(texts)} 件")

# 4. Chroma に登録（永続化）
PERSIST_DIR = "chroma_pokemon_30"

db = Chroma.from_texts(
    texts=texts,
    embedding=emb,
    collection_name="pokemon_30",
    persist_directory=PERSIST_DIR,
)

print("🔥 Chroma にポケモンデータ 30件を保存しました！")