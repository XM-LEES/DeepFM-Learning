### 💡 ArXiv 数据说明文档 (Feature Description)

这两份文件的逻辑和 Steam 高度一致，方便你在论文里进行对比叙述。

#### 文件一：`arxiv_raw_data.csv` (原始全量)

| 列名 | 描述 |
| :--- | :--- |
| `item_id` | 论文 ID (如 `2312.12345`)，去除了 URL 前缀。 |
| `abstract` | 完整的摘要文本。这是后续进行 **Content-based Recommendation** 的核心素材。 |
| `authors_raw` | JSON 格式的作者列表，保留了全量作者数据。 |

#### 文件二：`train_arxiv_Researcher_LLM.csv` (训练用)

| 列名 | DeepFM 用途 | Dify 用途 | 备注 |
| :--- | :--- | :--- | :--- |
| `item_id` | Sparse Feature | 唯一索引 | 需做 LabelEncoder。 |
| `category` | Sparse Feature | 显示分类 | 如 `cs.AI`, `cs.CV`。 |
| **`abstract`** | **Dense Feature (待转换)** | **核心内容** | **DeepFM 必须将此文本转换为向量 (768维)**。 |
| `label` | **Target** | *不展示* | 基于关键词匹配生成的“模拟点击”。 |
| `display_authors` | *不入模型* | **显示作者** | 格式化为 `Zhang et al.`，界面更整洁。 |
| `pdf_url` | *不入模型* | **跳转链接** | 点击即可下载 PDF。 |

### ⚠️ 研究生特别提示：关于 Abstract 到 Embedding 的处理

在 DeepFM 训练代码中，你不能直接喂字符串。你有两个低成本的选择：

1.  **简单版 (Baseline)**：
    直接**丢弃** `abstract` 和 `title` 文本，**只使用 `category` (类别) 作为特征**。

      * *优点*：代码极其简单，不用跑 BERT。
      * *缺点*：模型变得很笨，只能推荐同类别的文章，无法区分同一类别下的不同内容。

2.  **进阶版 (推荐)**：
    使用 `sentence-transformers` 离线生成向量。

    ```python
    # 这是一个附加脚本思路，不需要集成在 crawler 里
    from sentence_transformers import SentenceTransformer
    import pandas as pd
    import numpy as np

    model = SentenceTransformer('all-MiniLM-L6-v2') # 很小的模型
    df = pd.read_csv("train_arxiv_Researcher_LLM.csv")

    # 把 Title 和 Abstract 拼起来编码
    sentences = (df['title'] + ". " + df['abstract']).tolist()
    embeddings = model.encode(sentences) # 得到 shape (N, 384) 的矩阵

    # 保存为 numpy 文件，训练时直接加载
    np.save("arxiv_text_vectors.npy", embeddings)
    ```

    **在论文里可以写**：

    > “为了处理非结构化的文本数据，本研究使用预训练的 MiniLM 模型提取论文摘要的语义向量（Semantic Vectors），并将其作为 Dense Feature 输入 DeepFM 的 DNN 部分。” —— **这就显得很专业了！**