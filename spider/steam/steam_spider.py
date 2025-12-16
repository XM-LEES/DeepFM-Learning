import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import random

# === 配置 ===
BASE_URL = "https://store.steampowered.com/search/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Cookie": "birthtime=946684801; lastagecheckage=1-0-1900; wants_mature_content=1;"
}

def get_game_reviews(app_id):
    """
    调用 Steam API 获取该游戏最热门的 5 条中文评论
    用于 SFT 微调素材
    """
    # url = f"https://store.steampowered.com/appreviews/{app_id}?json=1&language=schinese&filter=summary&num_per_page=5"
    url = f"https://store.steampowered.com/appreviews/{app_id}?json=1&language=schinese&filter=summary"
    try:
        # 这里不需要 cookie 也能跑，如果报错再加 header
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data['success'] == 1:
                reviews = []
                for r in data['reviews']:
                    # 只要纯文本，去掉太短的垃圾评论
                    content = r['review'].strip()
                    if len(content) > 10: 
                        reviews.append(content)
                return reviews
    except:
        pass
    return []
    

def run_spider(max_pages=5):
    print(f"🚀 [Step 1] 开始爬取原始数据 (Raw Data)...")
    raw_games = []
    
    for page in range(1, max_pages + 1):
        print(f"   正在下载第 {page} 页...")
        params = {"filter": "topsellers", "page": page, "cc": "cn", "l": "schinese"}
        
        try:
            resp = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=15)
            if resp.status_code != 200: continue
            
            soup = BeautifulSoup(resp.text, "html.parser")
            rows = soup.select("#search_resultsRows > a")
            
            for row in rows:
                try:
                    # 获取最原始的数据，不清洗，不打标
                    app_id = row.get("data-ds-appid")
                    if not app_id: continue
                    
                    title = row.select_one(".title").text.strip()
                    
                    # 价格保留原始字符串 (如 "¥ 136.00" 或 "Free")，留给第二步处理
                    price_div = row.select_one(".discount_final_price") or row.select_one(".search_price")
                    price_raw = price_div.text.strip() if price_div else "0"
                    
                    # Tags 保留原始 JSON 列表
                    tag_str = row.get("data-ds-tagids")
                    tags_raw = json.loads(tag_str) if tag_str else []
                    
                    # 图片 URL
                    img_tag = row.select_one("img")
                    img_url = img_tag.get('srcset', '').split(', ')[0].split(' ')[0] or img_tag.get('src')
                    
                    # 好评信息 (用于第二步辅助打标)
                    review_sum = row.select_one(".search_review_summary")
                    review_raw = review_sum['data-tooltip-html'] if review_sum else ""

                    # ===  新增：获取评论数据 ===
                    # 只有当我们需要 SFT 素材时才跑这个，会慢一点点
                    reviews = get_game_reviews(app_id)

                    raw_games.append({
                        "item_id": app_id,
                        "title": title,
                        "price_raw": price_raw,
                        "tags_raw": tags_raw,      # 存为 List
                        "review_raw": review_raw,  # 存原始好评 HTML
                        "cover_url": img_url,
                        "user_reviews": reviews    # 把爬到的评论存成列表
                    })

                    print(f"   已获取: {title} (含 {len(reviews)} 条评论)")
                    
                except Exception as e:
                    continue
            
            # 稍微多睡一会，因为多请求了 API
            time.sleep(random.uniform(3, 6))
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

    # 保存原始数据
    df = pd.DataFrame(raw_games)
    # 强制把 tags 存为字符串形式，避免 CSV 读取歧义
    df['tags_raw'] = df['tags_raw'].apply(json.dumps)
    # 存 review 时防止 CSV 错乱，建议直接存 JSON 格式的字符串
    df['user_reviews'] = df['user_reviews'].apply(json.dumps, ensure_ascii=False)
    df.to_csv("../../data/steam/steam_raw_data.csv", index=False, encoding='utf-8-sig')
    print(f"✅ [Step 1] 完成！原始数据已保存至 '../../data/steam/steam_raw_data.csv' (共 {len(df)} 条)")

if __name__ == "__main__":
    run_spider(max_pages=60) # 建议爬 5 页，约 250 条数据
    # 25 * 60 = 1500 条数据