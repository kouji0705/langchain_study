import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import rcParams

# ★ 日本語フォントを指定（Mac想定）
#   うまくいかなければ "Hiragino Kaku Gothic ProN" なども試してみてください
rcParams["font.family"] = "Hiragino Sans"
# マイナス記号の文字化け防止
rcParams["axes.unicode_minus"] = False

with open("embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = data["texts"]
vectors = np.array(data["vectors"])

# PCAで2次元に圧縮
pca = PCA(n_components=2)
X_2d = pca.fit_transform(vectors)

plt.figure(figsize=(7, 7))

for i, (x, y) in enumerate(X_2d):
    plt.scatter(x, y, s=80)
    # ★ 日本語ラベルがそのまま表示されるようになる
    plt.text(x + 0.02, y + 0.02, texts[i], fontsize=9)

plt.title("OpenRouter + OpenAI Embedding (text-embedding-3-small) - 2D Plot", fontsize=11)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.tight_layout()
plt.show()