import pandas as pd
import json
import ast
import os

# ==========================================
# 1. 完整的 Steam Tag 映射字典 (直接硬编码在这里)
# ==========================================
# 这是基于 Steam 数据库整理的高频标签，涵盖了绝大多数热门 Tag
STEAM_TAG_MAP = {
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

def get_tag_names(tag_id_list):
    """
    输入: [19, 1774, 99999]
    输出: "动作, 射击" (过滤掉不认识的 ID)
    """
    if not isinstance(tag_id_list, list):
        return ""
    
    names = []
    for tid in tag_id_list:
        if tid in STEAM_TAG_MAP:
            names.append(STEAM_TAG_MAP[tid])
    
    return ", ".join(names)

def generate_sft_dataset():
    input_file = "../../data/steam/steam_raw_data.csv"
    output_file = "../../data/steam/steam_sft_train.json"

    print(f"⚙️ [SFT] 开始处理数据，读取: {input_file} ...")
    
    if not os.path.exists(input_file):
        print(f"❌ 错误: 找不到 {input_file}，请先运行 Step 1 Plus 爬虫！")
        return

    df = pd.read_csv(input_file)
    sft_data = []
    
    # 遍历每一行数据
    for index, row in df.iterrows():
        try:
            # 1. 解析基础信息
            title = row['title']
            
            # 2. 解析 Tags (CSV读取后是字符串，需要转回列表)
            # 兼容处理：有些可能是 float('nan')
            tags_raw = row['tags_raw']
            if pd.isna(tags_raw):
                tags_list = []
            elif isinstance(tags_raw, str):
                tags_list = json.loads(tags_raw) # 或者 ast.literal_eval(tags_raw)
            else:
                tags_list = tags_raw
            
            # === 🟢 核心：调用映射字典转中文 ===
            tag_str_cn = get_tag_names(tags_list)
            if not tag_str_cn: 
                tag_str_cn = "未知类型"

            # 3. 解析评论 (Reviews)
            # 我们的爬虫存的是 JSON 字符串 '["评论1", "评论2"]'
            reviews_raw = row['user_reviews']
            if pd.isna(reviews_raw):
                continue
                
            if isinstance(reviews_raw, str):
                reviews_list = json.loads(reviews_raw)
            else:
                reviews_list = reviews_raw

            if not reviews_list:
                continue

            # 4. 构造 SFT 数据对 (一对多)
            # 一个游戏有多条评论，可以生成多条训练数据，极大扩充数据集！
            for review_content in reviews_list:
                # 稍微清洗一下评论，去掉换行符
                clean_review = review_content.replace('\n', ' ').strip()
                
                # 只有当评论长度适中时才要 (太短没信息量，太长容易超 token)
                if 5 < len(clean_review) < 500:
                    sample = {
                        "instruction": f"请以资深玩家的身份，点评一下《{title}》这款游戏。",
                        "input": f"游戏类型标签：{tag_str_cn}",
                        "output": clean_review
                    }
                    sft_data.append(sample)

        except Exception as e:
            # 打印错误但不中断，方便调试
            print(f"⚠️ 跳过第 {index} 行: {e}")
            continue

    # 保存为 JSON (LLaMA-Factory 标准格式)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ SFT 数据集构建完成！")
    print(f"   📊 原始游戏数: {len(df)}")
    print(f"   🚀 生成微调样本数: {len(sft_data)} (一个游戏对应多条评论)")
    print(f"   💾 已保存至: {output_file}")
    
    # 打印一条预览看看效果
    if sft_data:
        print("\n🔎 数据样本预览:")
        print(json.dumps(sft_data[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    generate_sft_dataset()