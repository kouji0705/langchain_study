import json
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# .env 読み込み
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY が設定されていません。")

# テキスト一覧（わかりやすいカテゴリ）
texts = [
    "りんごは甘い果物です",
    "バナナは黄色い果物です",
    "犬は人懐っこい動物です",
    "トラは野生の動物です",
    "車は一般的な乗り物です",
    "バスは公共の乗り物です",
]

# OpenRouter 経由で OpenAI Embedding を使用
emb = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key,
    base_url="https://openrouter.ai/api/v1",  # ここが重要！
)

# ベクトル生成
vectors = emb.embed_documents(texts)

# embeddings.json に保存
with open("embeddings.json", "w", encoding="utf-8") as f:
    json.dump({"texts": texts, "vectors": vectors}, f, ensure_ascii=False, indent=2)

print("embeddings.json を保存しました！")