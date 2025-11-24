# chroma_store.py
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY が設定されていません")

# 保存用のディレクトリ
PERSIST_DIR = "chroma_db_example"

# 登録したい知識（今回はサンプル）
texts = [
    "りんごは甘い果物です。",
    "バナナは黄色い果物です。",
    "犬は人懐っこい動物です。",
    "トラは野生の動物です。",
    "車は一般的な乗り物です。",
    "バスは公共の乗り物です。",
]

# Embedding モデル：OpenRouter 経由の OpenAI Embedding
emb = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

# Chroma に保存（初回のみ）
db = Chroma.from_texts(
    texts=texts,
    embedding=emb,
    collection_name="demo_collection",
    persist_directory=PERSIST_DIR,
)

print("Chroma DB に保存完了！")
