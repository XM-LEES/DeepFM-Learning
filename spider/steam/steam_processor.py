import pandas as pd
import json
import re
import random
import os

# ==========================================
# 1. 定义 15 种典型的 Steam 玩家画像
# ==========================================
PROFILES = {
    # --- 基础三巨头 ---
    "Hardcore_FPS": { # 1. 枪男：只玩 CS/Apex/COD
        "fav_tags": [1774, 3878, 3859, 19], # 射击, 竞技, 多人, 动作
        "dislike_tags": [1742, 597, 1662],  # 讨厌视觉小说, 休闲, 模拟
        "price_sensitive": False
    },
    "Casual_Relax": { # 2. 休闲党：星露谷/动森
        "fav_tags": [597, 1662, 492, 4182], # 休闲, 模拟, 独立, 单人
        "dislike_tags": [1774, 4345, 1667], # 讨厌射击, 血腥, 恐怖
        "price_sensitive": True
    },
    "RPG_Story": { # 3. 剧情党：巫师/大表哥
        "fav_tags": [122, 4175, 1695, 21],  # RPG, 剧情丰富, 开放世界, 冒险
        "dislike_tags": [701, 699, 3878],   # 讨厌体育, 竞速, 竞技
        "price_sensitive": False
    },

    # --- 细分硬核群体 ---
    "Souls_Veteran": { # 4. 受苦学家：魂系/老头环
        "fav_tags": [4637, 19, 1684, 4172], # 类魂, 动作, 奇幻, 中世纪
        "dislike_tags": [597, 1742, 1036],  # 讨厌休闲, 视觉小说, 动漫(部分)
        "price_sensitive": False
    },
    "Strategy_Brain": { # 5. 策略大师：文明/P社
        "fav_tags": [599, 1662, 1708, 4172],# 策略, 模拟, 战术, 中世纪
        "dislike_tags": [19, 1774, 1697],   # 讨厌动作, 射击, 砍杀(无脑点鼠标)
        "price_sensitive": False
    },
    "Indie_Hipster": { # 6. 独立游戏迷：Hades/空洞骑士
        "fav_tags": [492, 1663, 4085, 1625],# 独立, 类Rogue, 像素风, 平台跳跃
        "dislike_tags": [3859, 113],        # 讨厌多人, 免费网游(通常有内购)
        "price_sensitive": True
    },

    # --- 特殊兴趣群体 ---
    "Anime_Weeb": { # 7. 二次元死宅
        "fav_tags": [1036, 1742, 122],      # 动漫, 视觉小说, RPG
        "dislike_tags": [1774, 701, 1667],  # 讨厌射击, 体育, 恐怖
        "price_sensitive": False            # 为爱买单不手软
    },
    "Horror_Fan": { # 8. 恐怖片爱好者
        "fav_tags": [1667, 4345, 1685, 21], # 恐怖, 血腥, 沙盒(生存), 冒险
        "dislike_tags": [597, 1036],        # 讨厌休闲, 萌系
        "price_sensitive": True
    },
    "SciFi_Geek": { # 9. 科幻迷
        "fav_tags": [3942, 1755, 1774, 1695], # 科幻, 赛博朋克, 射击, 开放世界
        "dislike_tags": [1684, 4172],         # 相对没那么喜欢纯奇幻/中世纪
        "price_sensitive": False
    },
    "Sports_Racer": { # 10. 体育/车枪球
        "fav_tags": [701, 699, 3859, 1774], # 体育, 竞速, 多人, 射击
        "dislike_tags": [122, 1742, 599],   # 讨厌文字量大的RPG, 视觉小说, 策略
        "price_sensitive": False
    },

    # --- 机制类群体 ---
    "Coop_Player": { # 11. 现充/开黑党：双人成行/分手厨房
        "fav_tags": [3859, 597, 21, 1664],  # 多人, 休闲, 冒险, 解谜
        "dislike_tags": [4182, 1667],       # 讨厌纯单人, 恐怖(吓跑妹子)
        "price_sensitive": True
    },
    "Puzzle_Solver": { # 12. 解谜爱好者：传送门/锈湖
        "fav_tags": [1664, 492, 21, 3942],  # 解谜, 独立, 冒险, 科幻
        "dislike_tags": [1774, 19, 3859],   # 讨厌突突突, 动作, 社交
        "price_sensitive": True
    },
    
    # --- 经济特征群体 ---
    "Free_Loader": { # 13. 白嫖怪
        "fav_tags": [113, 3859],            # 免费开玩, 多人
        "dislike_tags": [],
        "price_sensitive": "Strict"         # 特殊逻辑：只玩免费的
    },
    "Rich_Collector": { # 14. 喜加一/土豪
        "fav_tags": [],
        "dislike_tags": [],
        "price_sensitive": "Inverse"        # 特殊逻辑：只买贵的
    },
    "Trend_Follower": { # 15. 跟风党 (什么火玩什么)
        "fav_tags": [], # 靠好评率和热度判断
        "dislike_tags": [],
        "price_sensitive": False
    }
}

# ==========================================
# 2. 定义 Steam Tag ID 到中文名称的映射
# ==========================================
TAG_MAP = {
    # --- 核心分类 ---
    19: "动作", 122: "RPG", 599: "策略", 21: "冒险", 
    1662: "模拟", 597: "休闲", 701: "体育", 699: "竞速", 492: "独立",
    
    # --- 玩法机制 ---
    1774: "射击", 1663: "类Rogue", 1695: "开放世界", 1664: "解谜", 
    1742: "视觉小说", 1669: "大逃杀", 1625: "平台跳跃", 1734: "卡牌构建", 
    1743: "格斗", 1756: "塔防", 1708: "战术", 3859: "多人", 4182: "单人",
    
    # --- 题材/风格 ---
    3942: "科幻", 1684: "奇幻", 1667: "恐怖", 4172: "中世纪", 
    1755: "赛博朋克", 3839: "后末日", 1036: "动漫", 4085: "像素风",
    4175: "剧情丰富", 4667: "抢先体验", 4345: "血腥", 4637: "类魂", 
    1685: "沙盒", 1687: "潜行", 1697: "砍杀", 1716: "节奏", 
    113: "免费开玩", 3878: "竞技"
}

# ==========================================
# 3. 辅助函数
# ==========================================
def clean_price(price_str):
    """
    清洗价格字符串，转换为浮点数
    输入: "¥ 136.00" 或 "Free"
    输出: 136.0 或 0.0
    """
    if pd.isna(price_str) or "免费" in str(price_str) or "Free" in str(price_str):
        return 0.0
    clean = re.sub(r'[^\d.]', '', str(price_str))
    try:
        return float(clean)
    except:
        return 0.0

def generate_interaction_label(row, profile):
    """
    计算【特定用户】对【特定游戏】的喜好
    """
    score = 0.5
    tags = row['tags_list']
    price = row['price']
    
    # 1. 基础好评修正 (跟风党特别看重这个)
    if "好评" in str(row['review_raw']): 
        score += 0.1
        if profile == PROFILES["Trend_Follower"]: score += 0.2 # 跟风党加权
    
    # 2. 兴趣标签匹配
    for tag in tags:
        if tag in profile['fav_tags']: score += 0.25
        if tag in profile['dislike_tags']: score -= 0.4
    
    # 3. 价格敏感度逻辑 (增强版)
    p_sense = profile['price_sensitive']
    
    if p_sense == True: # 普通省钱党
        if price > 100: score -= 0.3
        if price < 50: score += 0.1
    elif p_sense == "Strict": # 白嫖怪
        if price > 0: score = 0 # 哪怕一块钱也不行 (直接置0)
        else: score += 0.4
    elif p_sense == "Inverse": # 土豪
        if price > 200: score += 0.4
        if price < 30: score -= 0.2
    
    # 随机扰动
    score += random.uniform(-0.1, 0.1)
    
    return 1 if score >= 0.65 else 0

# ==========================================
# 4. 主函数：生成 DeepFM 交互数据集
# ==========================================
def generate_deepfm_dataset():
    input_file = "../../data/steam/steam_raw_data.csv"
    output_file = "../../data/steam/deepfm_train_100k.csv"
    
    print("🚀 开始构建 DeepFM 交互数据集...")
    
    if not os.path.exists(input_file):
        print(f"❌ 找不到 {input_file}")
        return

    # 1. 读取基础游戏库 (Item Pool)
    df_items = pd.read_csv(input_file)
    # 预处理 Items
    df_items['tags_list'] = df_items['tags_raw'].apply(lambda x: json.loads(x) if pd.notna(x) else [])
    df_items['price'] = df_items['price_raw'].apply(clean_price)
    
    # 2. 生成虚拟用户群 (User Pool)
    # 生成 1000 个虚拟用户，分配 15 种人设
    virtual_users = []
    profile_names = list(PROFILES.keys()) # 获取所有 15 个名字
    
    for user_id in range(1000):
        # 轮询分配：0->FPS, 1->Casual, ... 14->Trend, 15->FPS...
        p_name = profile_names[user_id % len(profile_names)]
        
        virtual_users.append({
            "user_id": user_id,
            "user_type": p_name, # 这里的 p_name 会存入 CSV，DeepFM 会学到它！
            "profile": PROFILES[p_name]
        })
    
    # 3. 生成交互记录 (Interactions)
    # 让每个用户随机刷 60-100 个游戏
    all_interactions = []
    
    for user in virtual_users:
        # 随机抽取 60-100 个游戏给这个用户看（但不超过可用游戏数）
        max_sample = min(len(df_items), 100)
        min_sample = min(len(df_items), 60)
        sample_size = random.randint(min_sample, max_sample) if min_sample > 0 else len(df_items)
        seen_games = df_items.sample(n=sample_size)
        
        for _, game in seen_games.iterrows():
            # 计算 Label
            label = generate_interaction_label(game, user['profile'])
            
            # 构造一条训练样本
            # ⚠️ 关键：这里既要有 User特征，也要有 Item特征
            sample = {
                # User Features
                "user_id": user['user_id'],
                "user_type": user['user_type'], # 这是一个很强的特征！
                
                # Item Features
                "item_id": game['item_id'],
                "title": game['title'],
                "price": game['price'],
                "tags_list": game['tags_raw'], # 保持原始字符串给后续处理
                "tag_names": ",".join([TAG_MAP.get(i, "") for i in game['tags_list'] if i in TAG_MAP][:3]),
                "cover_url": game['cover_url'],
                
                # Label
                "label": label
            }
            all_interactions.append(sample)
            
    # 4. 保存
    df_train = pd.DataFrame(all_interactions)
    df_train.to_csv(output_file, index=False)
    
    print(f"✅ DeepFM 训练集构建完成！")
    print(f"   🎲 虚拟用户数: {len(virtual_users)}")
    print(f"   🕹️ 基础游戏数: {len(df_items)}")
    print(f"   ⚡ 总交互样本数: {len(df_train)} (Target: ~80k)")
    print(f"   💾 已保存至: {output_file}")
    print(f"   📊 正样本点击率: {df_train['label'].mean():.2%}")

if __name__ == "__main__":
    generate_deepfm_dataset()