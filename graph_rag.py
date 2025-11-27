import os
from dataclasses import dataclass
from typing import Dict, List

import networkx as nx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# =========================================================
# 0. 環境設定（.env 読み込み & OpenRouter モデル設定）
# =========================================================

load_dotenv()

# あなたが以前使っていたスタイルに合わせて OpenRouter を設定
# モデル名はお好みで変更OK（例: "anthropic/claude-3.5-sonnet", "openai/gpt-4o" など）
llm = ChatOpenAI(
    model="anthropic/claude-3.5-sonnet",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


# =========================================================
# 1. 共通データ: ドキュメント & 質問
# =========================================================

documents: Dict[str, str] = {
    "doc_1": "旧システムでは、夜間救急の受付処理に平均30分かかっていた。",
    "doc_2": "新システムでは、受付情報の自動入力とカルテ自動生成により、処理時間が5分に短縮された。",
}

query = "なぜ夜間救急の受付時間が短縮されたの？"


# =========================================================
# 2. ナイーブ RAG 風（比較用）
# =========================================================

KEYWORDS: List[str] = ["夜間救急", "受付", "短縮", "時間"]


def naive_score(doc: str, query_keywords: List[str]) -> int:
    """めちゃくちゃ単純なスコアリング（キーワード出現回数）"""
    return sum(doc.count(k) for k in query_keywords)


def build_naive_rag_prompt() -> str:
    scores = {
        doc_id: naive_score(text, KEYWORDS)
        for doc_id, text in documents.items()
    }

    top_doc_id = max(scores, key=scores.get)
    top_doc = documents[top_doc_id]

    prompt = f"""
[Naive RAG 用プロンプト]

質問:
{query}

参考情報:
{top_doc}

上記の情報だけを使って答えてください。
"""
    return prompt.strip()


# =========================================================
# 3. GraphRAG 風: 因果関係をグラフで保持
# =========================================================

@dataclass
class GraphConfig:
    """ナレッジグラフ構成用の設定（サンプルでは固定）"""
    query_related_nodes: List[str]


def build_knowledge_graph() -> nx.DiGraph:
    """サンプル用に手書きで因果グラフを構築（本番は LLM 抽出などで自動化）"""
    G = nx.DiGraph()

    # ノード: どのドキュメントに書かれているか docs 属性で紐づけ
    G.add_node("旧システム", docs={"doc_1"})
    G.add_node("新システム", docs={"doc_2"})
    G.add_node("夜間救急の受付", docs={"doc_1", "doc_2"})
    G.add_node("処理時間30分", docs={"doc_1"})
    G.add_node("処理時間5分", docs={"doc_2"})
    G.add_node("自動入力", docs={"doc_2"})
    G.add_node("カルテ自動生成", docs={"doc_2"})

    # エッジ: relation 属性で関係ラベルを付ける（任意）
    G.add_edge("旧システム", "夜間救急の受付", relation="handles")
    G.add_edge("夜間救急の受付", "処理時間30分", relation="takes_time")

    G.add_edge("新システム", "夜間救急の受付", relation="handles")
    G.add_edge("夜間救急の受付", "処理時間5分", relation="takes_time")

    G.add_edge("新システム", "自動入力", relation="uses")
    G.add_edge("新システム", "カルテ自動生成", relation="uses")
    G.add_edge("自動入力", "処理時間5分", relation="contributes_to")
    G.add_edge("カルテ自動生成", "処理時間5分", relation="contributes_to")

    return G


def extract_subgraph_docs(G: nx.DiGraph, config: GraphConfig) -> Dict[str, str]:
    """
    質問に関連しそうなノードから 1〜2 ホップ辿ってサブグラフを作り、
    そこにぶら下がっている doc_id を集める。
    """
    sub_nodes = set()

    for start in config.query_related_nodes:
        if start not in G:
            continue
        sub_nodes.add(start)

        # 1ホップ先
        sub_nodes.update(G.predecessors(start))
        sub_nodes.update(G.successors(start))

        # 2ホップ先（簡易）
        for n in list(G.predecessors(start)) + list(G.successors(start)):
            sub_nodes.update(G.predecessors(n))
            sub_nodes.update(G.successors(n))

    subgraph = G.subgraph(sub_nodes)

    related_doc_ids = set()
    for _, attrs in subgraph.nodes(data=True):
        if "docs" in attrs:
            related_doc_ids.update(attrs["docs"])

    return {doc_id: documents[doc_id] for doc_id in related_doc_ids}


def build_graph_rag_prompt() -> str:
    G = build_knowledge_graph()

    config = GraphConfig(
        query_related_nodes=[
            "夜間救急の受付",
            "処理時間30分",
            "処理時間5分",
        ]
    )

    related_docs = extract_subgraph_docs(G, config)

    context_text = "\n".join(
        f"[{doc_id}] {text}" for doc_id, text in related_docs.items()
    )

    prompt = f"""
[GraphRAG 用プロンプト]

質問:
{query}

ナレッジグラフから取得した関連ノード・エッジに紐づくドキュメント:

{context_text}

これらの情報をもとに、
「なぜ受付時間が短縮されたのか」を日本語でわかりやすく説明してください。
"""
    return prompt.strip()


# =========================================================
# 4. OpenRouter × LangChain で LLM を叩く関数
# =========================================================

def call_llm_with_openrouter(prompt: str) -> str:
    """
    ChatOpenAI(=OpenRouter) にそのまま文字列プロンプトを渡して呼び出す。
    ChatOpenAI は str を渡すと "human" メッセージとして扱ってくれる。
    """
    res = llm.invoke(prompt)
    # ChatMessage 形式なので .content を参照
    return res.content


# =========================================================
# 5. 実行例（Naive RAG vs GraphRAG）
# =========================================================

def main():
    print("=== Naive RAG ===")
    naive_prompt = build_naive_rag_prompt()
    print("----- prompt -----")
    print(naive_prompt)
    print("\n----- answer -----")
    naive_answer = call_llm_with_openrouter(naive_prompt)
    print(naive_answer)

    print("\n\n=== GraphRAG ===")
    graph_prompt = build_graph_rag_prompt()
    print("----- prompt -----")
    print(graph_prompt)
    print("\n----- answer -----")
    graph_answer = call_llm_with_openrouter(graph_prompt)
    print(graph_answer)


if __name__ == "__main__":
    main()