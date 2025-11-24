# pokemon_zukan_simple.py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os

# .env から OPENROUTER_API_KEY を読み込む
load_dotenv()

# OpenRouter 経由の Gemini を使う
model = ChatOpenAI(
    model="google/gemini-2.5-flash",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",  # OpenRouter の共通エンドポイント
)

# ポケモン図鑑用のプロンプトテンプレート
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "あなたはポケモン図鑑です。"
            "ユーザーが指定したポケモンについて、図鑑風に日本語でわかりやすく説明してください。"
            "存在しないポケモン名の場合は、その旨を丁寧に伝えてください。"
        ),
        (
            "human",
            "ポケモン名: {pokemon_name}\n"
            "このポケモンのタイプと、簡単な説明を教えてください。"
        ),
    ]
)

# Prompt → Model のチェーン
chain = prompt | model


def main() -> None:
    print("=== ポケモン図鑑（Gemini 版 × LangChain × OpenRouter） ===")
    print("ポケモン名を入力してください（例: ピカチュウ, ヒトカゲ）")
    print("空で Enter を押すと終了します。")

    while True:
        pokemon_name = input("\nあなた > ").strip()
        if not pokemon_name:
            print("終了します。")
            break

        # チェーンを実行して結果を取得
        res = chain.invoke({"pokemon_name": pokemon_name})

        print("\n図鑑 >")
        print(res.content)


if __name__ == "__main__":
    main()