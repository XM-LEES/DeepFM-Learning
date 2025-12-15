# DeepFM 推荐系统：全链路设计与实现技术手册 (v3.0)

## 1\. 系统架构概述 (System Architecture)

本系统旨在构建一个支持**多模态数据输入**与**实时在线推理**的推荐算法服务。系统采用 **ELT (Extract-Load-Transform)** 架构，将离线计算与在线服务解耦。

### 1.1 核心分层架构

  * **数据层 (Data Layer)**: 负责多源异构数据（Steam 结构化标签 vs ArXiv 非结构化文本）的爬取、清洗与特征工程。
  * **算法层 (Model Layer)**: 基于 **DeepCTR-Torch** 框架，搭建 **DeepFM** 模型，融合 FM (低阶特征交叉) 与 DNN (高阶非线性特征) 能力。
  * **服务层 (Serving Layer)**: 基于 **Flask** + **Ngrok** 构建 RESTful API 网关，支持热加载模型权重，为 Dify 智能体提供实时推荐服务。

-----

## 2\. 数据工程实现 (Data Engineering)

### 2.1 异构特征处理策略

系统针对两种截然不同的数据场景，设计了定制化的特征处理流水线：

| 场景 | 核心特征挑战 | 解决方案 | 技术实现细节 |
| :--- | :--- | :--- | :--- |
| **Steam** | **变长离散序列**<br>(Tags: `[19, 122]`) | **Sequence Padding** | 使用 `pad_sequences` 将不定长列表补齐为固定长度 (MaxLen=5)，不足补 0。 |
| **ArXiv** | **非结构化长文本**<br>(Abstract) | **Semantic Embedding** | 引入 `Sentence-BERT` (all-MiniLM-L6-v2) 提取语义，生成 384维 稠密向量。 |
| **通用** | **ID 类稀疏特征** | **Label Encoding** | 建立 `String -> Integer` 索引映射，压缩 Embedding 空间。 |
| **通用** | **数值特征** | **MinMax Scaling** | 将价格、年份等归一化至 `[0, 1]` 区间。 |

### 2.2 监督信号生成 (Label Generation)

由于缺乏真实用户日志，本研究采用**基于规则的用户画像模拟 (Rule-based Profile Simulation)** 生成训练标签：

  * **Steam**: 结合 `Tags` 匹配度 + 全网好评率 + 价格敏感度。
  * **ArXiv**: 结合 `Keywords` (如 LLM, Agent) 匹配度 + 引用热度模拟。

-----

## 3\. 模型训练与配置 (Model Training)

### 3.1 网络结构超参数 (Hyperparameters)

| 参数名 | Steam 模型配置 | ArXiv 模型配置 | 设计理由 |
| :--- | :--- | :--- | :--- |
| **DNN Hidden Units** | `(128, 128)` | `(256, 128)` | ArXiv 输入特征维度高 (384维)，首层需更宽以捕捉信息。 |
| **Embedding Dim** | 16 | 16 | 平衡模型表达能力与显存占用。 |
| **Dropout** | 0.2 | 0.5 | ArXiv 数据相对稀疏且维度高，增加 Dropout 防止过拟合。 |
| **Loss Function** | `binary_crossentropy` | `binary_crossentropy` | 点击率预估 (CTR) 本质为二分类问题。 |
| **Optimizer** | `Adam (lr=0.001)` | `Adam (lr=0.001)` | 自适应学习率，加速稀疏特征的收敛。 |

### 3.2 训练输入张量结构 (Tensor Schema)

用于调试 Dimension Mismatch 错误的关键参考：

  * **Steam Input**: `{ "item_id": (N,), "price": (N,1), "tags": (N,5) }`
  * **ArXiv Input**: `{ "item_id": (N,), "category": (N,), "v_0"..."v_383": (N,1) }`

-----

## 4\. 模型持久化与热加载 (Persistence & Serving)

### 4.1 权重序列化 (Serialization)

训练结束后，仅保存模型的参数字典 (`state_dict`)，以实现轻量化存储。

  * **保存指令**: `torch.save(model.state_dict(), 'deepfm_weights.pth')`
  * **文件产出**: `deepfm_steam_weights.pth`, `deepfm_arxiv_weights.pth`

### 4.2 API 服务架构 (Flask + Ngrok)

服务层不依赖训练环境，可独立运行。

  * **初始化流程**:

    1.  解析 CSV Header，构建 Feature Columns。
    2.  初始化 DeepFM 空网络结构。
    3.  加载 `.pth` 权重文件至设备 (CPU/CUDA)。
    4.  执行 `model.eval()` 冻结梯度。

  * **接口定义 (API Contract)**:

      * **Endpoint**: `POST /recommend`
      * **Router Logic**: 根据请求体中的 `type` 字段，动态分发至 `model_steam` 或 `model_arxiv` 进行推理。

-----

## 5\. 接口调用指南 (API Reference)

以下命令用于验证服务可用性，可直接在终端运行。

### 5.1 通用调用参数

  * **Header**: `Content-Type: application/json`
  * **URL**: 使用 Ngrok 生成的公网地址 (例如 `https://xyz.ngrok-free.app/recommend`)

### 5.2 场景 A: 请求 Steam 游戏推荐

**cURL 命令**:

```bash
curl -X POST https://你的ngrok域名.ngrok-free.app/recommend \
     -H "Content-Type: application/json" \
     -d '{
           "type": "steam",
           "top_k": 3
         }'
```

**预期响应**:

```json
{
  "status": "success",
  "recommendations": [
    {
      "title": "Black Myth: Wukong",
      "score": 0.982,
      "type": "动作, RPG",
      "cover": "https://..."
    }
  ]
}
```

### 5.3 场景 B: 请求 ArXiv 论文推荐

**cURL 命令**:

```bash
curl -X POST https://你的ngrok域名.ngrok-free.app/recommend \
     -H "Content-Type: application/json" \
     -d '{
           "type": "arxiv",
           "top_k": 3
         }'
```

**预期响应**:

```json
{
  "status": "success",
  "recommendations": [
    {
      "title": "Attention Is All You Need",
      "score": 0.954,
      "type": "cs.CL",
      "cover": "https://arxiv.org/pdf/..."
    }
  ]
}
```

-----

## 6\. 环境与依赖 (Dependencies)

```text
python >= 3.8
torch >= 1.10.0
deepctr-torch >= 0.2.9
flask >= 2.0.0
pyngrok >= 7.0.0
sentence-transformers (仅 ArXiv 预处理阶段需要)
```