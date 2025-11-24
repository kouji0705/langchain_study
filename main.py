# main.py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate  # ← ここを修正
import os

# ① .env の読み込み
load_dotenv()

# ② OpenRouter 用の ChatOpenAI モデル設定
# OpenAI互換APIなので base_url を変更するだけでOK
model = ChatOpenAI(
    model="anthropic/claude-3.5-sonnet",  # OpenRouter経由で利用するモデル
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",  # ←超重要
)

# ③ プロンプトテンプレート定義
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは優秀な日本語アシスタントです。わかりやすく説明してください。"),
        ("human", "{question}"),
    ]
)

# ④ Prompt → Model のパイプライン（Chain）
chain = prompt | model


def main():
    print("=== OpenRouter × LangChain チャット ===")
    print("質問を入力してください（空でEnter → 終了）")

    while True:
        question = input("\nあなた > ").strip()
        if not question:
            print("終了します。")
            break

        # ⑤ チェーン実行
        res = chain.invoke({"question": question})

        # ChatMessage形式なので .content を参照
        print("\nAI > " + res.content)


if __name__ == "__main__":
    main()