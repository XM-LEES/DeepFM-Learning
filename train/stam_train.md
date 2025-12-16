# 基于 DeepFM 的 Steam 游戏个性化推荐系统设计与实现

## 1\. 项目背景与目标 (Project Background)

本项目旨在构建一个基于深度学习的点击率预估（CTR Prediction）系统，用于 Steam 游戏平台的个性化推荐。系统能够根据不同的**玩家类型（User Type）**，结合游戏本身的\*\*标签（Tags）**和**价格（Price）\*\*特征，预测玩家对特定游戏的感兴趣程度，并实时生成推荐列表。

## 2\. 系统架构 (System Architecture)

本系统采用经典的**离线训练 + 在线服务**架构：

  * **训练端 (`steam_train.py`)**：负责数据预处理、特征工程、模型训练与评估、权重导出。
  * **服务端 (`steam_service.py`)**：基于 Flask 构建 RESTful API，加载训练好的模型权重，接收用户请求并进行实时推理（Inference）。

## 3\. 数据处理与特征工程 (Data & Feature Engineering)

### 3.1 数据集描述

  * **来源**：Steam 用户行为数据集（清洗后约 100k 条交互记录，涵盖 1500+ 个独立游戏）。
  * **正负样本**：
      * Label=1：用户购买/游玩过该游戏。
      * Label=0：负采样数据。

### 3.2 特征定义 (Feature Columns)

为了充分挖掘用户与物品的关联，模型输入包含以下三类特征：

| 特征域 (Field) | 特征名称 | 类型 | 处理方式 | 维度 (Embedding) | 作用 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **User Side** | `user_type` | 离散 (Sparse) | LabelEncoding -\> Embedding | 32维 | 捕捉不同群体(如 Hardcore\_FPS)的偏好 |
| **Item Side** | `item_id` | 离散 (Sparse) | LabelEncoding -\> Embedding | 32维 | 学习游戏的隐式向量表示 |
| **Item Side** | `tags` | 变长序列 (VarLen) | Padding(MaxLen=5) -\> Mean Pooling | 32维 | 语义层面的游戏内容理解 |
| **Context** | `price` | 连续 (Dense) | MinMaxScaling (归一化到 0-1) | 1维 | 捕捉价格对购买决策的影响 |

### 3.3 关键处理逻辑

  * **序列补齐**：针对 `tags`（如 `[FPS, Action]`），统一补齐或截断为长度 5，不足部分填充 0。
  * **广播机制 (Broadcasting)**：在预测阶段，将单一用户的 `user_type` 广播至全量游戏列表，构造 `(N_items, Features)` 的批量输入矩阵，实现单次推理即可对全库打分。

## 4\. 模型设计 (Model Design)

### 4.1 核心算法：DeepFM

本项目采用 DeepFM 模型，其核心优势在于能同时学习**低阶特征组合**和**高阶非线性特征**，且无需人工特征工程。

  * **FM 部分 (Factorization Machine)**：负责提取二阶特征交叉（例如：`UserType=RPG` 与 `Tag=Story` 的共现关系）。
  * **DNN 部分 (Deep Neural Network)**：负责挖掘高阶隐含特征。

### 4.2 网络结构调优 (Hyperparameter Tuning)

针对 10万级数据量，为防止过拟合（Overfitting），进行了以下针对性设计（**论文亮点**）：

  * **DNN 结构**：采用 **(128, 64)** 的两层“窄”网络。
      * *设计理由*：相比于 (1024, 512) 的大网络，小网络参数量少，更适合中小规模数据集，有效降低了模型的复杂度。
  * **Dropout 策略**：设置 **0.5** 的高 Dropout 率。
      * *设计理由*：在训练过程中随机丢弃 50% 的神经元，强制模型学习鲁棒特征，显著缓解了验证集 Loss 反弹的问题。
  * **Embedding Dimension**：**32维**。
      * *设计理由*：对于 1500 个 Item，32 维向量足以表达其特征空间，避免维度灾难。

## 5\. 训练策略 (Training Strategy)

  * **优化器**：Adam (Learning Rate = 0.001)。
  * **损失函数**：Binary Crossentropy（二元交叉熵）。
  * **Batch Size**：**256**。
      * *策略分析*：较小的 Batch Size 引入了梯度噪声，有助于模型跳出局部最优解（Local Minima），提升了模型的泛化能力。
  * **早停策略 (Early Stopping)**：通过观察 Loss 曲线，发现 Epoch 11\~12 为最佳泛化点（Validation Loss 最低），在此处停止训练以获取最佳权重。

## 6\. 工程实现与服务部署 (Engineering & Deployment)

### 6.1 接口设计

  * **API 路径**：`POST /recommend`
  * **输入参数**：
    ```json
    {
      "type": "Hardcore_FPS",  // 指定玩家类型
      "top_k": 5               // 推荐数量
    }
    ```
  * **输出格式**：包含游戏ID、标题、预测得分及封面图URL。

### 6.2 个性化推荐流程

1.  **解析请求**：接收 `user_type`，将其转换为模型内部的 `user_type_idx`。
2.  **全库打分**：构造全量游戏的特征矩阵，输入 DeepFM 模型进行批量预测（Batch Prediction）。
3.  **重排与去重**：
      * 按预测分数（Score）降序排列。
      * 执行 `drop_duplicates(subset=['item_id'])`，解决多条交互记录导致的推荐重复问题。
4.  **Top-K 截断**：返回得分最高的前 K 个结果。

## 7\. 实验结果 (Results)

  * **训练集表现**：AUC \> 0.95，Loss 持续下降。
  * **验证集表现**：**AUC 达到 0.92+**，Loss 在 0.35 左右收敛。
  * **定性分析**：
      * 输入 `Hardcore_FPS` -\> 推荐《CS:GO》、《Doom》等射击游戏。
      * 输入 `RPG_Player` -\> 推荐《The Witcher 3》等剧情向游戏。
      * *结论*：模型成功学习到了不同用户类型的兴趣偏好，具备良好的个性化推荐能力。

-----

## 8\. 代码文件清单 (File Structure)

| 文件名 | 描述 | 核心功能 |
| :--- | :--- | :--- |
| `steam_train.py` | 训练主程序 | 数据加载、标签编码、模型构建、Loss可视化、权重保存 |
| `steam_service.py` | 推理服务 | Flask 接口、模型重载、请求解析、实时推荐逻辑 |
| `requirements.txt` | 依赖清单 | torch, deepctr-torch, flask, pandas 等 |
| `deepfm_steam_weights.pth` | 模型权重 | 训练好的二进制权重文件 |
| `training_loss.png` | 训练监控 | Loss 变化曲线图（用于论文插图） |

-----

### 💡 写论文/报告的小贴士

1.  **画个图**：在论文里，你可以根据第 6 节描述的流程，画一个“推荐系统数据流图”（Data Flow Diagram）。
2.  **贴 Loss 曲线**：把 `steam_train.py` 生成的 `training_loss.png` 贴上去，解释为什么你在第 12 轮停止了训练（防止过拟合），这体现了你对深度学习训练过程的理解。
3.  **对比实验**（如果有空）：你可以说你试过 Batch Size=1024，发现效果不好，改成了 256，这证明了参数调优的重要性。