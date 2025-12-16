# Steam 推荐系统数据工程技术手册 (v2.0)

## 1\. 系统概述 (System Overview)

本模块构建了一个多模态、双链路的数据处理系统，旨在同时服务于 **DeepFM 排序模型** 和 **Qwen 大语言模型 (SFT)**。系统采用 **“一次爬取，分流处理” (One-Crawl, Two-Process)** 的架构设计：

1.  **DeepFM 流**：通过**用户模拟增强 (Simulation-Augmented)** 技术，基于有限的物品池生成大规模（10w+）用户交互数据，解决冷启动与数据稀疏问题。
2.  **SFT 流**：提取高质量的用户评论，构建 **Alpaca 格式** 指令微调数据集，赋予大模型“资深玩家”的语言风格与领域知识。

-----

## 2\. 架构流程 (Architecture Pipeline)

  * **Step 1 数据获取层 (Extract & Enrich)**:
      * 全量爬取 Steam 游戏元数据（价格、标签、封面）。
      * **[新增]** 调用 Steam API 抓取热门中文评论（User Reviews），作为 SFT 的语料来源。
  * **Step 2 数据处理层 (Transform & Augment)**:
      * **分支 A (DeepFM)**: 构建 15 类虚拟用户画像，进行笛卡尔积采样，生成 `User-Item` 交互矩阵。
      * **分支 B (SFT)**: 清洗评论文本，构造 `Instruction-Input-Output` 对，生成 JSON 微调数据。

-----

## 3\. 详细数据字典 (Data Schema)

### 3.1 原始数据湖 (`steam_full_data.csv`)

**来源**: `step1_steam_spider_plus.py`
**用途**: 全量冷备份，DeepFM 和 SFT 的公共底座。

| 字段名 | 类型 | 语义描述 | 备注 |
| :--- | :--- | :--- | :--- |
| `item_id` | Int | Steam AppID | 主键 |
| `title` | String | 游戏名称 | |
| `price_raw` | String | 原始价格 | 含货币符号，需清洗 |
| `tags_raw` | String | 标签 ID 列表 | `"[19, 122]"` |
| **`user_reviews`** | **String** | **热门评论列表** | **[新增] JSON 字符串 `["评论1", "评论2"]`** |
| `cover_url` | String | 封面链接 | |

### 3.2 DeepFM 训练集 (`deepfm_train_100k.csv`)

**来源**: `step2_deepfm_processor.py` (含 15 种用户画像模拟)
**用途**: 训练点击率预估模型，包含显性的 User 特征。

| 字段名 | 类型 | 描述 | DeepFM 用途 |
| :--- | :--- | :--- | :--- |
| **`user_id`** | Int | **虚拟用户 ID** | **Sparse Feature** (0-999) |
| **`user_type`** | String | **用户人设** | **Sparse Feature** (如 `Hardcore_FPS`) |
| `item_id` | Int | 游戏 ID | Sparse Feature |
| `price` | Float | 归一化价格 | Dense Feature |
| `tags_list` | List[Int]| 标签序列 | VarLenSparseFeat |
| `label` | Int | **模拟交互结果** | **Target** (1=点击, 0=未点击) |

### 3.3 SFT 微调数据集 (`steam_sft_train.json`)

**来源**: `step2_sft_processor.py`
**用途**: LLaMA-Factory 微调 Qwen2.5-7B。采用标准 Alpaca 格式。

```json
[
  {
    "instruction": "请以资深玩家的身份，点评一下《黑神话：悟空》这款游戏。",
    "input": "游戏类型标签：动作, RPG, 类魂",
    "output": "这游戏打击感真不错，但这空气墙也太搞人心态了..."
  }
]
```

-----

## 4\. 核心技术实现

### 4.1 模块一：增强型爬虫 (`Spider Plus`)

  * **API 注入**: 在爬取 HTML 列表页的同时，异步请求 `store.steampowered.com/appreviews/{appid}` 接口。
  * **策略**: 每个游戏抓取 15-20 条评论，保留 10 条高质量（长度适中）评论。

### 4.2 模块二：交互数据增强 (DeepFM Data Augmentation)

为了解决仅爬取 1500 个游戏导致的数据量不足问题，本系统引入**用户模拟器 (User Simulator)**。

1.  **用户池构建**: 生成 1000 个虚拟 Agent，均匀分布在 **15 种典型画像** 中（如 `Souls_Veteran` 受苦玩家, `Free_Loader` 白嫖党, `Anime_Weeb` 二次元等）。
2.  **交互生成**:
      * 每个 Agent 随机采样 60-100 个游戏。
      * 应用 `generate_interaction_label(user, item)` 函数计算点击概率。
      * **扩增倍数**: $1,500 \text{ Items} \times \text{Sampling} \approx \mathbf{100,000 \text{ Interactions}}$。

### 4.3 模块三：指令微调数据构建 (SFT Construction)

  * **标签语义化**: 利用 `STEAM_TAG_MAP` 将数字 ID (`19`) 转译为自然语言 (`动作`)，作为 Prompt 的 Input 部分，辅助 LLM 理解游戏背景。
  * **噪音清洗**: 过滤掉纯符号、过短（\<5字）或过长（\>800字）的无效评论。

-----

## 5\. 快速操作指南 (Quick Start)

### Step 1: 获取全量数据 (One-Click Fetch)

```bash
# 运行增强版爬虫，同时获取元数据和评论
python step1_steam_spider_plus.py
# 输出: steam_full_data.csv
```

### Step 2: 并行处理 (Two-Stream Process)

**流向 A: 生成 DeepFM 训练数据 (10万条)**

```bash
# 运行用户模拟器
python step2_processor_deepfm.py
# 输出: deepfm_train_100k.csv -> 喂给 DeepFM 训练脚本
```

**流向 B: 生成 SFT 微调数据 (5000条)**

```bash
# 运行评论清洗器
python step2_processor_sft.py
# 输出: steam_sft_train.json -> 喂给 LLaMA-Factory
```

-----

## 6\. 常见问题 (FAQ)

  * **Q: 为什么要模拟 1000 个用户？直接用 1500 条游戏数据训练不行吗？**
      * A: 不行。DeepFM 深度学习模型需要大量的样本来拟合特征交叉。仅用 1500 条数据会导致严重的过拟合（模型记住了每一个游戏，而不是学会了推荐规律）。通过模拟 1000 个用户的不同喜好，模型才能学会 *"Hardcore 玩家喜欢射击"* 这种泛化规律。
  * **Q: SFT 数据里的 `input` 字段有什么用？**
      * A: `input` 提供了游戏的客观信息（Tags）。这让模型在生成评论时，能够将“老玩家的语气”与“当前游戏的类型”结合起来，避免产生幻觉（例如给一个射击游戏生成“剧情感人”的评论）。