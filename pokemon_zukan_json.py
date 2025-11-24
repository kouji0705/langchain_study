# pokemon_zukan_json.py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
import json
from typing import Any, Dict, List, Optional


# --------------------------
# 1. JSON から独自図鑑データを読み込む
# --------------------------
def load_pokemon_data(path: str = "pokemon_data.json") -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_pokemon_by_name(name_jp: str, data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """日本語名で完全一致検索（とりあえずシンプルに）"""
    for entry in data:
        if entry["name_jp"] == name_jp:
            return entry
    return None


# --------------------------
# 2. LangChain / Gemini の準備
# --------------------------
load_dotenv()

model = ChatOpenAI(
    model="google/gemini-2.5-flash",          # OpenRouter 上の Gemini モデル
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# 「このデータだけを使って答えろ」と明示するプロンプト
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "あなたはポケモン図鑑です。"
            "与えられたデータだけを利用して回答してください。"
            "与えられていない情報を推測で補ったり、新しい事実を作ったりしてはいけません。"
        ),
        (
            "human",
            "以下のポケモンデータだけを使って、図鑑風に説明してください。\n\n"
            "{pokedex_entry}"
        ),
    ]
)

chain = prompt | model


# --------------------------
# 3. メイン処理
# --------------------------
def format_entry_for_prompt(entry: Dict[str, Any]) -> str:
    """LLM に渡すために、1匹分のデータをテキストに整形"""
    types = ", ".join(entry.get("types", []))
    return (
        f"図鑑番号: {entry.get('id')}\n"
        f"名前: {entry.get('name_jp')} ({entry.get('name_en')})\n"
        f"タイプ: {types}\n"
        f"高さ: {entry.get('height_m')} m\n"
        f"重さ: {entry.get('weight_kg')} kg\n"
        f"説明: {entry.get('description')}\n"
    )


def main() -> None:
    print("=== ポケモン図鑑（独自JSONナレッジ版 × Gemini） ===")
    print("※ この図鑑は pokemon_data.json に登録されているポケモンだけを参照します。")
    print("ポケモン名を入力してください（例: ピカチュウ, フシギダネ）")
    print("空で Enter を押すと終了します。")

    # JSON を読み込む（起動時に1回だけ）
    try:
        data = load_pokemon_data()
    except FileNotFoundError:
        print("pokemon_data.json が見つかりません。先に作成してください。")
        return

    while True:
        name = input("\nあなた > ").strip()
        if not name:
            print("終了します。")
            break

        entry = find_pokemon_by_name(name, data)
        if entry is None:
            print("\n図鑑 >")
            print("そのポケモンは、現在この図鑑データ（pokemon_data.json）には登録されていません。")
            print("pokemon_data.json に追記すれば、この図鑑からも参照できるようになります。")
            continue

        # 見つかったら、そのデータだけを LLM に渡して説明してもらう
        pokedex_entry_text = format_entry_for_prompt(entry)

        res = chain.invoke({"pokedex_entry": pokedex_entry_text})

        print("\n図鑑 >")
        print(res.content)


if __name__ == "__main__":
    main()
