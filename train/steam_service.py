import pandas as pd
import numpy as np
import ast
import torch
from flask import Flask, request, jsonify
from pyngrok import ngrok
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
from deepctr_torch.models import DeepFM

class ServiceConfig:
    CSV_PATH = '../data/steam/deepfm_train_100k.csv'
    MODEL_PATH = 'deepfm_steam_weights.pth'
    MAX_TAG_LEN = 5
    EMBEDDING_DIM = 32
    DNN_HIDDEN_UNITS = (128, 64)
    DNN_DROPOUT = 0.5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    PORT = 5000
    NGROK_TOKEN = "这里粘贴你的_Ngrok_Token"

cfg = ServiceConfig()

def pad_sequences(sequences, maxlen, value=0):
    result = np.full((len(sequences), maxlen), value, dtype=np.int32)
    for i, seq in enumerate(sequences):
        if len(seq) > 0:
            trunc = seq[:maxlen]
            result[i, :len(trunc)] = trunc
    return result

# 全局保存 user encoder，用于把 "Hardcore_FPS" 转成数字
global_user_lbe = None

def load_data_struct(csv_path, config):
    print(f"📂 [Service] 读取数据索引: {csv_path} ...")
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError: return None, None, None

    data['tags_list'] = data['tags_list'].apply(lambda x: ast.literal_eval(x))
    
    # 1. Tag
    all_tags = [tag for sublist in data['tags_list'] for tag in sublist]
    tag_lbe = LabelEncoder()
    tag_lbe.fit(all_tags)
    max_tag_id = len(tag_lbe.classes_) + 1
    data['tags_list_idx'] = data['tags_list'].apply(lambda x: [i+1 for i in tag_lbe.transform(x)] if len(x)>0 else [])
    
    # 2. Item
    item_lbe = LabelEncoder()
    data['item_id_idx'] = item_lbe.fit_transform(data['item_id'])
    max_item_id = data['item_id_idx'].max() + 1
    
    # 3. UserType (🔥 关键：保存这个 encoder)
    global global_user_lbe
    global_user_lbe = LabelEncoder()
    # 必须 fit 所有的类型，保证和训练时 ID 一致
    data['user_type_idx'] = global_user_lbe.fit_transform(data['user_type'])
    max_user_type_id = data['user_type_idx'].max() + 1
    print(f"🔥 支持的玩家类型: {list(global_user_lbe.classes_)}")
    
    # 4. Price
    mms = MinMaxScaler(feature_range=(0, 1))
    data['price_norm'] = mms.fit_transform(data[['price']])
    
    fixlen_feature_columns = [
        SparseFeat('item_id_idx', vocabulary_size=max_item_id, embedding_dim=config.EMBEDDING_DIM),
        SparseFeat('user_type_idx', vocabulary_size=max_user_type_id, embedding_dim=config.EMBEDDING_DIM), # 🔥
        DenseFeat('price_norm', dimension=1)
    ]
    varlen_feature_columns = [
        VarLenSparseFeat(
            SparseFeat('tags', vocabulary_size=max_tag_id, embedding_dim=config.EMBEDDING_DIM),
            maxlen=config.MAX_TAG_LEN, combiner='mean', length_name=None
        )
    ]
    return fixlen_feature_columns + varlen_feature_columns, fixlen_feature_columns + varlen_feature_columns, data

app = Flask(__name__)
model_steam = None
full_data_df = None

def init_model():
    global model_steam, full_data_df
    linear_cols, dnn_cols, df = load_data_struct(cfg.CSV_PATH, cfg)
    if linear_cols is None: return
    full_data_df = df
    
    model_steam = DeepFM(linear_feature_columns=linear_cols, dnn_feature_columns=dnn_cols, task='binary', 
                         dnn_hidden_units=cfg.DNN_HIDDEN_UNITS, dnn_dropout=cfg.DNN_DROPOUT, device=cfg.DEVICE)
    try:
        model_steam.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=cfg.DEVICE))
        model_steam.eval()
        print("✅ 模型加载成功！")
    except Exception as e: print(f"❌ 错误: {e}")

@app.route('/recommend', methods=['POST'])
def recommend():
    if model_steam is None: return jsonify({"error": "Model not loaded"}), 500
    try:
        req_json = request.json
        top_k = req_json.get('top_k', 3)
        # 获取请求的玩家类型，默认 Hardcore_FPS
        user_type_str = req_json.get('type', 'Hardcore_FPS') 
        
        # 🔥 1. 将类型字符串转为 ID
        if user_type_str not in global_user_lbe.classes_:
            return jsonify({"error": f"Unknown type: {user_type_str}. Supported: {list(global_user_lbe.classes_)}"}), 400
        
        # 拿到对应的数字 ID (例如 2)
        user_type_id = global_user_lbe.transform([user_type_str])[0]
        
        print(f"🎮 收到请求: Type={user_type_str}(ID={user_type_id}), Top {top_k}")

        # 🔥 2. 构造全量输入 (关键步骤)
        # 我们有 N 个游戏，需要构造 N 个 user_type_id
        # 也就是：[2, 2, 2, ..., 2] (长度等于游戏数量)
        # 意思是：预测“这个特定的玩家”对“每一个游戏”的喜好
        num_items = len(full_data_df)
        user_type_col = np.full(num_items, user_type_id) 

        tags_padded = pad_sequences(list(full_data_df['tags_list_idx']), maxlen=cfg.MAX_TAG_LEN)
        
        model_input = {
            'item_id_idx': full_data_df['item_id_idx'].values,
            'user_type_idx': user_type_col,  # 🔥 这里传入的是全量的单一用户ID
            'price_norm': full_data_df['price_norm'].values,
            'tags': tags_padded
        }
        
        with torch.no_grad():
            scores = model_steam.predict(model_input, batch_size=4096)
            
        res_df = full_data_df.copy()
        res_df['score'] = scores
        # 排序并去重
        top_items = res_df.sort_values(by='score', ascending=False).drop_duplicates(subset=['item_id']).head(top_k)
        
        results = [{
            "id": str(r['item_id']),
            "title": r['title'],
            "score": float(r['score']),
            "cover": r.get('cover_url', ''),
            "tags": r.get('tag_names', '')
        } for _, r in top_items.iterrows()]
        
        return jsonify({"code": 200, "type": user_type_str, "data": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    init_model()
    if not cfg.NGROK_TOKEN.startswith("这里"):
        ngrok.set_auth_token(cfg.NGROK_TOKEN)
        ngrok.kill()
        try: print(f"🌍 {ngrok.connect(cfg.PORT, bind_tls=True).public_url}/recommend")
        except: pass
    app.run(port=cfg.PORT, use_reloader=False)