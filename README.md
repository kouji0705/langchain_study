# 🐟 環境構築手順（fish シェル前提）

## 1. Python 仮想環境（venv）の作成

```fish
python3 -m venv venv
```

---

## 2. 仮想環境を有効化（activate）

fish シェルでは activate.fish を使います。

```fish
source venv/bin/activate.fish
```

---

## 3. 必要パッケージのインストール

```fish
pip install langchain langchain-openai python-dotenv
```

---

## 4. `.env` の作成

OpenRouter の API キーを設定します。

```env
OPENROUTER_API_KEY=あなたのOpenRouterのAPIキー
```

---

## 5. 動作確認（main.py を実行）

```fish
python3 main.py
```

---

## 6. 仮想環境の終了

```fish
deactivate
```