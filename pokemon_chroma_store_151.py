# pokemon_chroma_store_151.py
import csv
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma  # 新しい Chroma パッケージを使用

# 1. .env 読み込み
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY が設定されていません")

# 2. Embedding モデル（OpenRouter 経由の OpenAI Embedding）
emb = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

# 3. CSV 読み込み
CSV_PATH = "pokemon_151_with_image.csv"
texts: list[str] = []
metadatas: list[dict] = []

with open(CSV_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # ベクトル化に使うテキスト（意味検索用）
        text = (
            f"{row['name_jp']}（{row['name_en']}）: "
            f"タイプ={row['type1']} {row['type2']}。"
            f"説明: {row['description']}"
        )
        texts.append(text)

        # メタデータ（タイプ・名前・画像URLなど）
        metadatas.append(
            {
                "id": int(row["id"]),
                "name_jp": row["name_jp"],
                "name_en": row["name_en"],
                "type1": row["type1"],
                "type2": row["type2"],
                "image_url": row["image_url"],
            }
        )

print(f"CSV 読み込み完了: {len(texts)} 件")

# 4. Chroma に保存（永続化）
PERSIST_DIR = "chroma_pokemon_151"

db = Chroma.from_texts(
    texts=texts,
    embedding=emb,
    metadatas=metadatas,
    collection_name="pokemon_151",
    persist_directory=PERSIST_DIR,
)

print("🔥 Chroma にポケモン151匹を保存しました！")