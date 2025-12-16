import pandas as pd
import numpy as np
import ast
import torch
import os
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
from deepctr_torch.models import DeepFM
from deepctr_torch.callbacks import EarlyStopping
import torch.optim as optim

# ==========================================
# ⚙️ 配置中心
# ==========================================
class SteamConfig:
    CSV_PATH = '../data/steam/deepfm_train_100k.csv'
    MODEL_PATH = 'deepfm_steam_weights.pth'
    PLOT_PATH = 'training_loss.png'
    
    MAX_TAG_LEN = 5
    EMBEDDING_DIM = 32
    DNN_HIDDEN_UNITS = (128, 64)
    DNN_DROPOUT = 0.5
    
    BATCH_SIZE = 256
    EPOCHS = 20
    LEARNING_RATE = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 2025

cfg = SteamConfig()

# --- 固定随机种子 (保证结果可复现) ---
def seed_everything(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(cfg.SEED)

# ==========================================
# 🛠️ 核心工具函数
# ==========================================
def pad_sequences(sequences, maxlen, value=0):
    result = np.full((len(sequences), maxlen), value, dtype=np.int32)
    for i, seq in enumerate(sequences):
        if len(seq) > 0:
            trunc = seq[:maxlen]
            result[i, :len(trunc)] = trunc
    return result

def load_steam_data(csv_path, config):
    print(f"📂 [Train] 正在加载数据: {csv_path} ...")
    data = pd.read_csv(csv_path)
    
    # 1. Tags 处理
    data['tags_list'] = data['tags_list'].apply(lambda x: ast.literal_eval(x))
    all_tags = [tag for sublist in data['tags_list'] for tag in sublist]
    tag_lbe = LabelEncoder()
    tag_lbe.fit(all_tags)
    max_tag_id = len(tag_lbe.classes_) + 1
    data['tags_list'] = data['tags_list'].apply(lambda x: [i+1 for i in tag_lbe.transform(x)] if len(x)>0 else [])
    
    # 2. ItemID 处理
    item_lbe = LabelEncoder()
    data['item_id_idx'] = item_lbe.fit_transform(data['item_id'])
    max_item_id = data['item_id_idx'].max() + 1
    
    # 3. UserType 处理 (🔥 新增部分)
    # 我们需要记录下有哪些 User Type，以便服务时使用
    user_lbe = LabelEncoder()
    data['user_type_idx'] = user_lbe.fit_transform(data['user_type'])
    max_user_type_id = data['user_type_idx'].max() + 1
    print(f"🔥 识别到玩家类型: {list(user_lbe.classes_)}")
    
    # 4. Price 处理
    mms = MinMaxScaler(feature_range=(0, 1))
    data['price_norm'] = mms.fit_transform(data[['price']])
    
    # 5. 特征定义
    tags_padded = pad_sequences(list(data['tags_list']), maxlen=config.MAX_TAG_LEN, value=0)
    
    fixlen_feature_columns = [
        SparseFeat('item_id_idx', vocabulary_size=max_item_id, embedding_dim=config.EMBEDDING_DIM),
        # 🔥 新增：把用户类型也作为一个特征放入模型
        SparseFeat('user_type_idx', vocabulary_size=max_user_type_id, embedding_dim=config.EMBEDDING_DIM),
        DenseFeat('price_norm', dimension=1)
    ]
    
    varlen_feature_columns = [
        VarLenSparseFeat(
            SparseFeat('tags', vocabulary_size=max_tag_id, embedding_dim=config.EMBEDDING_DIM),
            maxlen=config.MAX_TAG_LEN, combiner='mean', length_name=None
        )
    ]
    
    linear_cols = fixlen_feature_columns + varlen_feature_columns
    dnn_cols = fixlen_feature_columns + varlen_feature_columns
    
    # 6. 组装输入
    model_input = {
        'item_id_idx': data['item_id_idx'].values,
        'user_type_idx': data['user_type_idx'].values, # 🔥 放入训练数据
        'price_norm': data['price_norm'].values,
        'tags': tags_padded
    }
    
    return model_input, linear_cols, dnn_cols, data['label'].values

def plot_and_save_loss(history, save_path):
    loss = history.history['loss']
    val_loss = history.history.get('val_loss', history.history.get('val_binary_crossentropy'))
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    if val_loss: plt.plot(epochs, val_loss, 'r--', label='Validation Loss')
    plt.title(f'Loss (Batch:{cfg.BATCH_SIZE})')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"📊 Loss 曲线已保存: {save_path}")

if __name__ == "__main__":
    input_dict, linear_cols, dnn_cols, target = load_steam_data(cfg.CSV_PATH, cfg)
    
    print(f"🔧 初始化 DeepFM (含 UserType 特征)...")
    model = DeepFM(linear_feature_columns=linear_cols, 
                   dnn_feature_columns=dnn_cols, 
                   task='binary', 
                   dnn_hidden_units=cfg.DNN_HIDDEN_UNITS, 
                   dnn_dropout=cfg.DNN_DROPOUT,
                   device=cfg.DEVICE)
    
    model.compile(optimizer=optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE), 
              loss="binary_crossentropy", 
              metrics=["binary_crossentropy", "auc"])
    
    es = EarlyStopping(monitor='val_auc', min_delta=0, patience=2, mode='max')
    
    print(f"🚀 开始训练 (Epochs: {cfg.EPOCHS}, Batch: {cfg.BATCH_SIZE})...")
    history = model.fit(input_dict, target, batch_size=cfg.BATCH_SIZE, epochs=cfg.EPOCHS, verbose=2, validation_split=0.2, callbacks=[es])
    
    torch.save(model.state_dict(), cfg.MODEL_PATH)
    print(f"✅ 模型已保存: {cfg.MODEL_PATH}")
    plot_and_save_loss(history, cfg.PLOT_PATH)