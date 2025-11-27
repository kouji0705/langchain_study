# pokemon_search_pokemon_151.py
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 1. .env 読み込み
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY が設定されていません")

# 2. Embedding モデル
emb = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

# 3. 永続化された Chroma をロード
PERSIST_DIR = "chroma_pokemon_151"

db = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name="pokemon_151",
    embedding_function=emb,
)

print("📁 Chroma ポケモン151 データベース読み込み完了")


def semantic_search_with_filters() -> None:
    """
    意味検索 + メタ情報フィルタをまとめて行う関数。

    - query: 自然文（例: 「砂の中に潜って丸くなって身を守るポケモン」）
    - type1/type2: 必要なら英語タイプで絞り込み（例: ground, fire, water）
    """
    print("\n=== 意味検索 + メタ情報フィルタ ===")
    query = input("どんなポケモンを探したい？ > ").strip()
    if not query:
        print("クエリが空です。")
        return

    print("\nタイプで絞り込む場合は、英語タイプ名を入力してください。")
    print("例: grass, fire, water, electric, ground, rock, psychic, ice, dragon, normal, poison, bug, flying, steel, fairy ...")
    type1 = input("type1 で絞り込み（空なら指定なし）> ").strip()
    type2 = input("type2 で絞り込み（空なら指定なし）> ").strip()

    filter_dict: dict | None = None
    if type1 or type2:
        filter_dict = {}
        if type1:
            filter_dict["type1"] = type1
        if type2:
            filter_dict["type2"] = type2

    # filter_dict が None なら全体から意味検索、あればその条件内で意味検索
    docs = db.similarity_search(
        query,
        k=5,
        filter=filter_dict,
    )

    if not docs:
        print("該当するポケモンが見つかりませんでした。")
        return

    print("\n🔎 検索結果 Top5:")
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        print(f"\n[{i}] #{meta['id']} {meta['name_jp']}（{meta['name_en']}）")
        print(f"  タイプ: {meta['type1']}, {meta['type2']}")
        print(f"  画像: {meta['image_url']}")
        print(f"  内容: {doc.page_content[:80]}...")  # 説明長いので先頭だけ表示


def metadata_only_search() -> None:
    """
    メタ情報だけでの絞り込み（おまけ）。
    例: type1 = 'ground' のポケモン一覧を見たいだけ、など。
    """
    print("\n=== メタ情報だけで検索（一覧用） ===")
    print("type1 / type2 は英語表記です（例: fire, water, ground, electric ...）")
    key = input("どのキーで絞り込みますか？(type1 / type2) > ").strip()
    value = input("値を入力してください > ").strip()

    if key not in ("type1", "type2"):
        print("type1 / type2 以外は未対応です。")
        return

    result = db.get(where={key: value})
    ids = result.get("ids", [])
    metadatas = result.get("metadatas", [])

    if not ids:
        print("該当するポケモンがいませんでした。")
        return

    print(f"\n🔎 {key} = {value} のポケモン: {len(ids)} 件")
    for meta in metadatas:
        print(
            f"- #{meta['id']:>3} {meta['name_jp']}（{meta['name_en']}） "
            f"[{meta['type1']}, {meta['type2']}]"
        )


def main() -> None:
    while True:
        print("\n==============================")
        print("1: 意味検索 + メタ情報フィルタ")
        print("2: メタ情報だけで一覧を出す（type1 / type2）")
        print("Enter: 終了")
        print("==============================")
        choice = input("モードを選んでください > ").strip()

        if choice == "":
            print("終了します。")
            break
        elif choice == "1":
            semantic_search_with_filters()
        elif choice == "2":
            metadata_only_search()
        else:
            print("1 / 2 / Enter のいずれかを選んでください。")


if __name__ == "__main__":
    main()