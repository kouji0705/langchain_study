# pokemon_search.py
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


# --------------------------
# 2. 名前正規化系ユーティリティ
# --------------------------
def hira_to_kata(text: str) -> str:
    """ひらがなをカタカナに変換（その他の文字はそのまま）"""
    result = []
    for ch in text:
        code = ord(ch)
        # ひらがな: 0x3041〜0x3096 → カタカナ: +0x60
        if 0x3041 <= code <= 0x3096:
            result.append(chr(code + 0x60))
        else:
            result.append(ch)
    return "".join(result)


def normalize_name(name: str) -> str:
    """前後の空白を削って、ひらがなをカタカナにそろえる"""
    name = name.strip()
    name = hira_to_kata(name)
    return name


# --------------------------
# 3. 検索ロジック
# --------------------------
def find_by_id(poke_id: int, data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for entry in data:
        if entry.get("id") == poke_id:
            return entry
    return None


def search_by_name(
    name_jp: str, data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    日本語名で検索する。
    - まず「正規化して完全一致」を探す
    - 見つからなければ「部分一致」を探して候補一覧を返す
    """
    query_norm = normalize_name(name_jp)

    exact_matches: List[Dict[str, Any]] = []
    partial_matches: List[Dict[str, Any]] = []

    for entry in data:
        entry_name = entry.get("name_jp", "")
        entry_norm = normalize_name(entry_name)

        if entry_norm == query_norm:
            exact_matches.append(entry)
        elif query_norm and query_norm in entry_norm:
            partial_matches.append(entry)

    if exact_matches:
        return {"mode": "exact", "matches": exact_matches}
    if partial_matches:
        return {"mode": "partial", "matches": partial_matches}

    return {"mode": "none", "matches": []}


# --------------------------
# 4. LangChain / Gemini の準備
# --------------------------
load_dotenv()

model = ChatOpenAI(
    model="google/gemini-2.5-flash",          # OpenRouter 上の Gemini モデル
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

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
# 5. 出力整形
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


# --------------------------
# 6. メイン処理
# --------------------------
def main() -> None:
    print("=== ポケモン図鑑（独自JSONナレッジ版 × Gemini） ===")
    print("※ この図鑑は pokemon_data.json に登録されているポケモンだけを参照します。")
    print("入力例:")
    print("  ・ポケモン名      : ピカチュウ / ぴかちゅう / ピカ")
    print("  ・図鑑番号        : 25")
    print("空で Enter を押すと終了します。")

    # JSON を読み込む（起動時に1回だけ）
    try:
        data = load_pokemon_data()
    except FileNotFoundError:
        print("pokemon_data.json が見つかりません。先に作成してください。")
        return

    while True:
        query = input("\nあなた > ").strip()
        if not query:
            print("終了します。")
            break

        # 1) 数字だけなら「図鑑番号」とみなす
        if query.isdigit():
            poke_id = int(query)
            entry = find_by_id(poke_id, data)
            if entry is None:
                print("\n図鑑 >")
                print(f"図鑑番号 {poke_id} のポケモンは、現在この図鑑データには登録されていません。")
                continue

            pokedex_entry_text = format_entry_for_prompt(entry)
            res = chain.invoke({"pokedex_entry": pokedex_entry_text})
            print("\n図鑑 >")
            print(res.content)
            continue

        # 2) それ以外は「名前検索」
        result = search_by_name(query, data)
        mode = result["mode"]
        matches: List[Dict[str, Any]] = result["matches"]

        if mode == "none":
            print("\n図鑑 >")
            print("その名前に該当するポケモンは、現在この図鑑データには登録されていません。")
            continue

        if mode == "partial" and len(matches) > 1:
            # あいまいすぎる場合は候補一覧を出す
            print("\n図鑑 >")
            print("いくつかの候補が見つかりました。もう少し詳しく指定してください。")
            for e in matches:
                print(f"  - 図鑑番号 {e.get('id')}: {e.get('name_jp')}")
            continue

        # exact または partial で1件だけ → 図鑑表示
        entry = matches[0]
        pokedex_entry_text = format_entry_for_prompt(entry)
        res = chain.invoke({"pokedex_entry": pokedex_entry_text})

        print("\n図鑑 >")
        print(res.content)


if __name__ == "__main__":
    main()