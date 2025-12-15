# Steam 推荐系统数据工程技术手册 (v1.0)

## 1\. 系统概述

本模块旨在构建一个高质量、包含用户偏好的数据集，用于训练 **DeepFM 点击率预估模型**，并支持 **Dify 智能体**的前端展示。系统采用 **ELT (Extract-Load-Transform)** 架构，将爬虫（I/O 密集型）与数据处理（计算密集型）分离，确保数据获取的稳定性与特征构建的灵活性。

## 2\. 架构流程

  * **Step 1 数据获取层 (Extract)**: 绕过 Steam 年龄验证，获取原始 HTML 数据，保留全量特征。
  * **Step 2 数据处理层 (Transform)**: 清洗价格与格式，注入“模拟用户画像”生成监督信号 (Label)。

-----

## 3\. 详细数据字典 & 映射表 (Mapping Schema)

### 3.1 核心字段定义

| 字段名 (Column) | 数据类型 | 语义描述 | 推荐系统用途 (DeepFM) | Dify 用途 |
| :--- | :--- | :--- | :--- | :--- |
| `item_id` | Int | Steam AppID (主键) | **Sparse Feature** (需 Embedding) | 唯一索引 |
| `title` | String | 游戏名称 | *不入模型* | **展示标题** |
| `price` | Float | 价格 (RMB) | **Dense Feature** (归一化处理) | 展示价格 |
| `tags_list` | List[Int] | 标签 ID 列表 `[19, 122]` | **Sparse Feature** (Multi-hot) | *隐式特征* |
| `tag_names` | String | 标签名称 `"动作, RPG"` | *不入模型* | **展示标签** |
| `label` | Int (0/1) | 用户是否偏好 | **Target (训练目标)** | *不展示* |
| `cover_url` | String | 封面图片链接 | *不入模型* | **Markdown 渲染** |

### 3.2 完整 Steam 标签映射表 (Tag ID Mapping)

请将此字典直接写入 `step2_processor.py` 中。这是基于 Steam 数据库整理的高频标签 Top 50，涵盖了 DeepFM 所需的 95% 特征覆盖率。

```python
STEAM_TAG_FULL_MAP = {
    # --- 核心分类 (Genre) ---
    19: "动作 (Action)",
    122: "角色扮演 (RPG)",
    599: "策略 (Strategy)",
    21: "冒险 (Adventure)",
    1662: "模拟 (Simulation)",
    597: "休闲 (Casual)",
    701: "体育 (Sports)",
    699: "竞速 (Racing)",
    492: "独立 (Indie)",
    
    # --- 玩法机制 (Gameplay) ---
    1774: "射击 (Shooter)",
    1663: "类 Rogue (Roguelike)",
    1695: "开放世界 (Open World)",
    1664: "解谜 (Puzzle)",
    1742: "视觉小说 (Visual Novel)",
    1669: "大逃杀 (Battle Royale)",
    1625: "平台跳跃 (Platformer)",
    1734: "卡牌构建 (Deckbuilder)",
    1743: "格斗 (Fighting)",
    1756: "塔防 (Tower Defense)",
    1708: "战术 (Tactical)",
    
    # --- 题材/风格 (Theme/Style) ---
    3942: "科幻 (Sci-fi)",
    1684: "奇幻 (Fantasy)",
    1667: "恐怖 (Horror)",
    4172: "中世纪 (Medieval)",
    1755: "赛博朋克 (Cyberpunk)",
    3839: "后末日 (Post-apocalyptic)",
    1036: "动漫 (Anime)",
    4085: "像素风 (Pixel Graphics)",
    
    # --- 社交属性 (Social) ---
    3859: "多人 (Multiplayer)",
    4182: "单人 (Singleplayer)",
    3843: "在线合作 (Co-op)",
    3878: "竞技 (Competitive)",
    
    # --- 特殊属性 (Misc) ---
    113: "免费开玩 (Free to Play)",
    4175: "剧情丰富 (Story Rich)",
    4667: "抢先体验 (Early Access)",
    4345: "血腥 (Gore)",
    4637: "类魂 (Souls-like)",
    1685: "沙盒 (Sandbox)",
    1687: "潜行 (Stealth)",
    1697: "砍杀 (Hack and Slash)",
    1698: "点击 (Point & Click)",
    1716: "节奏 (Rhythm)",
    1719: "喜剧 (Comedy)"
}
```

-----

## 4\. 模块详细说明

### 4.1 模块一：数据爬取 (`step1_steam_spider.py`)

  * **输入**：Steam Store URL (`https://store.steampowered.com/search/`).
  * **输出**：`steam_raw_data.csv` (约 200-500 条原始数据).
  * **核心技术点**：
    1.  **Cookie 注入**：Header 中携带 `birthtime=946684801`，强制通过 Steam 的年龄验证网关，防止重定向。
    2.  **鲁棒性解析**：针对不同国家的定价格式（`Free`, `¥ 199.00`, `NO PRICE`）做兼容性提取。
    3.  **防封策略**：随机间隔 (1-3s) + 仅爬取列表页（避免高频访问详情页触发 IP Ban）。

### 4.2 模块二：特征工程与打标 (`step2_processor.py`)

  * **输入**：`steam_raw_data.csv` + `USER_PROFILE` (配置字典).
  * **输出**：`train_data_{profile}.csv` (清洗完毕，带 Label 的训练集).
  * **核心算法：基于规则的偏好注入 (Rule-based Preference Injection)**
      * **公式**：$$Label = \mathbb{I} ( (0.2 \times \text{GlobalRating}) + (0.4 \times \text{TagMatch}) - (0.5 \times \text{TagDislike}) - (0.3 \times \text{PricePenalty}) \ge \text{Threshold} )$$
      * **逻辑**：
        1.  **标签匹配 (Tag Matching)**：计算 Item Tags 与 User Profile 中 `fav_tags` 和 `dislike_tags` 的交集。
        2.  **价格惩罚 (Price Penalty)**：若用户为价格敏感型 (`price_sensitive=True`)，对高价游戏降权。
        3.  **阈值截断**：最终得分 $\ge 0.6$ 标记为 **1 (Positive)**，否则为 **0 (Negative)**。

-----

## 5\. 文件结构

采用了 **Step 1 (爬取)** 和 **Step 2 (处理)** 分离的架构，两个 CSV 文件的定位完全不同。

-----

#### 文件一：原始数据备份 (`steam_raw_data.csv`)

**定位**：数据湖 / 原始归档 (Raw Archive)。
**来源**：`step1_steam_spider.py` 的直接产出。
**用途**：防止爬虫被封时的冷备份；保留原始 HTML 细节供后续挖掘。

| 列名 (Column) | 数据类型 | 示例数据 | 说明与注意点 |
| :--- | :--- | :--- | :--- |
| **item\_id** | String / Int | `730` | **Steam AppID**。这是游戏的身份证，后续所有处理的主键。 |
| **title** | String | `Counter-Strike 2` | 游戏名。保留原始多语言格式，未做特殊清洗。 |
| **price\_raw** | String | `¥ 0.00` 或 `Free to Play` | **原始价格字符串**。注意这里包含了货币符号、逗号或文字，**不能直接入模型**。 |
| **tags\_raw** | String (JSON) | `"[1663, 1774, 3859]"` | **标签 ID 的字符串形式**。为了防止 CSV 读写错误，这里强制存为字符串。读取时需要 `json.loads`。 |
| **review\_raw** | String (HTML) | `&lt;div class=...&gt;好评...` | **原始好评栏 HTML**。包含了“好评如潮”、“褒贬不一”等文字信息，用于第二步辅助判断基础质量。 |
| **cover\_url** | String (URL) | `https://.../header.jpg` | **封面图原始链接**。直接从 CDN 获取，通常带有 `?t=...` 时间戳参数。 |

-----

#### 文件二：模型训练集 (`train_data_UserProfile.csv`)

**定位**：特征工程产物 / 训练集 (Feature Set)。
**来源**：`step2_processor.py` 清洗并注入偏好后的产物。
**用途**：**直接喂给 DeepFM 进行训练**，以及作为 **Dify 知识库**的展示源。

| 列名 (Column) | 数据类型 | 示例数据 | 推荐系统用途 (DeepFM) | Dify / 前端用途 |
| :--- | :--- | :--- | :--- | :--- |
| **item\_id** | Int | `730` | **Sparse Feature** (ID Embedding)<br>需建立 `ID -> Index` 映射。 | 唯一标识符，用于检索。 |
| **title** | String | `CS2` | *不入模型* | **展示核心**。<br>LLM 回复时的标题。 |
| **price** | Float | `0.0` 或 `198.0` | **Dense Feature** (数值特征)<br>模型会学习“价格敏感度”。 | 显示价格。<br>如：“¥ 198.0”。 |
| **tags\_list** | List[Int] | `[1774, 19, 3859]` | **Sparse Feature** (VarLenSparse)<br>这是**最核心**的特征。<br>需进行 Multi-hot 编码或 Pooling。 | *隐式特征*。<br>用户看不懂数字 ID。 |
| **tag\_names** | String | `"射击, 动作, 多人"` | *不入模型* | **展示标签**。<br>让 LLM 能用自然语言介绍游戏类型。 |
| **label** | Int | `1` 或 `0` | **Target (训练目标)**<br>`1`=喜欢 (点击)<br>`0`=不喜欢 (未点击) | *不展示*。<br>这是 Ground Truth。 |
| **cover\_url** | String | `https://...` | *不入模型* | **视觉展示**。<br>Markdown 语法 `![img](url)`。 |


1.  **价格处理 (`price_raw` vs `price`)**：

      * 原始数据是脏的字符串（含 `¥`），模型无法计算。
      * 训练数据已清洗为纯浮点数（Float），可以直接进行 Min-Max 归一化。

2.  **标签处理 (`tags_raw` vs `tags_list` vs `tag_names`)**：

      * `tags_raw`: 存储时的 JSON 字符串。
      * `tags_list`: **给模型看的**。DeepFM 通过 ID 查 Embedding 表。
      * `tag_names`: **给人看的**。Dify 通过这个字段生成“为您推荐射击游戏”的文案。

3.  **Label 的来源**：

      * 原始数据**没有 Label**。
      * 训练数据的 Label 是通过 **Step 2 中的算法**，结合 `review_raw` (全网口碑) 和 `tags_list` (个人偏好) 动态生成的。

-----

## 6\. 快速开始指南

1.  **环境准备**：
    ```bash
    pip install requests pandas beautifulsoup4
    ```
2.  **第一步：获取数据**
    运行 `step1_steam_spider.py`。
    > *注：此步依赖网络，只需运行一次即可获得本地数据备份。*
3.  **第二步：生成训练集**
    修改 `step2_processor.py` 中的 `USER_PROFILE` 字典（例如修改为“动作游戏爱好者”）。
    运行脚本，生成 `train_data_Hardcore.csv`。
4.  **第三步：下游对接**
      * **DeepFM**: 读取 CSV，使用 `tags_list` 列进行 Embedding 训练，使用 `label` 计算 LogLoss。
      * **Dify**: 在提示词中调用 `{{tag_names}}` 和 `{{cover_url}}` 进行富文本渲染。

-----

## 7\. 常见问题 (FAQ)

  * **Q: 为什么 Tags 是列表格式？**
      * A: 游戏通常属于多个分类（如既是“动作”又是“RPG”）。DeepFM 需使用 `VarLenSparseFeat` (变长离散特征) 处理此类数据，通常做法是补齐长度或取 TopK。
  * **Q: 为什么没有具体的评分 (1-5分)？**
      * A: 点击率预估 (CTR) 任务关注的是“点击概率”，二分类标签 (0/1) 比稀疏的评分矩阵更适合神经网络训练，且更符合工业界推荐流场景。