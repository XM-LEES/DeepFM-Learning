import pandas as pd
import json
import re
import ast

# === 🕹️ 可以在这里修改模拟用户的喜好！ ===
# 场景 A: 暴力硬核玩家
USER_PROFILE_HARDCORE = {
    "name": "Hardcore_Gamer",
    "fav_tags": [19, 1774, 3859, 1663], # 动作, 射击, 多人, 肉鸽
    "dislike_tags": [597, 1742],        # 休闲, 视觉小说
    "price_sensitive": False
}

# 场景 B: 贫穷的休闲玩家 (用于对比测试)
USER_PROFILE_CASUAL = {
    "name": "Casual_Gamer",
    "fav_tags": [597, 1662, 492],       # 休闲, 模拟, 独立
    "dislike_tags": [1774, 4345],       # 射击, 血腥
    "price_sensitive": True             # 嫌贵
}

# 场景 C：喜欢动作/射击/RPG，讨厌休闲/体育的硬核玩家
USER_PROFILE_RPGFAN = {
    "name": "Action_RPG_Fan",
    "fav_tags": [19, 1774, 122, 3859, 1695, 1663],  # 动作, 射击, RPG, 多人, 开放世界, 肉鸽
    "dislike_tags": [597, 599, 701],                # 休闲, 策略, 体育
    "price_sensitive": False                        # 是否价格敏感
}

# Tag ID 映射表 (为了 Dify 展示)
TAG_MAP = {19: "动作", 1774: "射击", 597: "休闲", 122: "RPG", 1663: "肉鸽", 3859: "多人", 1662: "模拟"}

def clean_price(price_str):
    """清洗价格"""
    if pd.isna(price_str) or "免费" in str(price_str) or "Free" in str(price_str):
        return 0.0
    clean = re.sub(r'[^\d.]', '', str(price_str))
    try:
        return float(clean)
    except:
        return 0.0

def generate_label(row, profile):
    """根据 Profile 生成 Label"""
    score = 0.5
    tags = row['tags_list']
    
    # 1. 基础好评修正
    if "好评" in str(row['review_raw']): score += 0.2
    
    # 2. 兴趣匹配
    for tag in tags:
        if tag in profile['fav_tags']: score += 0.2
        if tag in profile['dislike_tags']: score -= 0.4 # 讨厌的权重更大
    
    # 3. 价格敏感
    if profile['price_sensitive'] and row['price'] > 50:
        score -= 0.3
        
    return 1 if score >= 0.6 else 0

def process_data(user_profile):
    print(f"⚙️ [Step 2] 正在为用户 [{user_profile['name']}] 生成训练数据...")
    
    # 读取原始数据
    df = pd.read_csv("../../data/steam/steam_raw_data.csv")
    
    # --- 数据清洗 (ETL) ---
    # 1. 还原 Tags (从 string "[1,2]" -> list [1,2])
    df['tags_list'] = df['tags_raw'].apply(lambda x: json.loads(x) if pd.notna(x) else [])
    
    # 2. 清洗价格
    df['price'] = df['price_raw'].apply(clean_price)
    
    # 3. 生成可读 Tag (Dify 展示用)
    df['tag_names'] = df['tags_list'].apply(lambda x: ",".join([TAG_MAP.get(i, "") for i in x if i in TAG_MAP][:3]))
    
    # --- 核心：打标 (Labeling) ---
    df['label'] = df.apply(lambda row: generate_label(row, user_profile), axis=1)
    
    # --- 导出 ---
    # 只保留模型和展示需要的列
    final_cols = ['item_id', 'title', 'price', 'tags_list', 'tag_names', 'label', 'cover_url']
    output_file = f"../../data/steam/train_data_{user_profile['name']}.csv"
    df[final_cols].to_csv(output_file, index=False)
    
    print(f"   ✅ 生成完毕: {output_file}")
    print(f"   📊 正样本比例: {df['label'].mean():.2%}\n")

if __name__ == "__main__":
    # 你可以一次性生成两份数据，看看区别！
    process_data(USER_PROFILE_HARDCORE)
    process_data(USER_PROFILE_CASUAL)
    process_data(USER_PROFILE_RPGFAN)