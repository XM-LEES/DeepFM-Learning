import feedparser
import pandas as pd
import time
import urllib.parse

# === 配置 ===
# 搜索关键词：这里搜索 "Artificial Intelligence" 也就是 cs.AI 类别
SEARCH_QUERY = "cat:cs.AI OR cat:cs.CL OR cat:cs.CV" 
MAX_RESULTS = 500 # 建议 200-500 条
BASE_URL = "http://export.arxiv.org/api/query?"

def fetch_arxiv_raw():
    print(f"🚀 [Step 1] 开始爬取 ArXiv 论文 (Query: {SEARCH_QUERY})...")
    
    # ArXiv API 支持一次性请求大量数据，不需要像 Steam 那样翻页
    # 但为了稳定性，建议每 100 条请求一次
    all_papers = []
    batch_size = 100
    
    for start in range(0, MAX_RESULTS, batch_size):
        print(f"   正在获取第 {start} - {start+batch_size} 条...")
        
        params = {
            "search_query": SEARCH_QUERY,
            "start": start,
            "max_results": batch_size,
            "sortBy": "submittedDate", # 按提交时间倒序 (最新的在前面)
            "sortOrder": "descending"
        }
        
        url = BASE_URL + urllib.parse.urlencode(params)
        
        # 使用 feedparser 解析 XML
        feed = feedparser.parse(url)
        
        if not feed.entries:
            print("   ⚠️ 未获取到数据，可能已达到末尾。")
            break
            
        for entry in feed.entries:
            try:
                # 1. ID 清洗 (http://arxiv.org/abs/2312.00001v1 -> 2312.00001v1)
                paper_id = entry.id.split('/abs/')[-1]
                
                # 2. 提取 PDF 链接
                pdf_url = ""
                for link in entry.links:
                    if link.type == 'application/pdf':
                        pdf_url = link.href
                
                # 3. 提取主分类 (Primary Category)
                primary_cat = entry.arxiv_primary_category['term']
                
                # 4. 提取作者 (List)
                authors = [author.name for author in entry.authors]
                
                all_papers.append({
                    "item_id": paper_id,
                    "title": entry.title.replace('\n', ' '),
                    "abstract": entry.summary.replace('\n', ' '),
                    "authors_raw": authors,       # 存为 List
                    "category": primary_cat,      # Sparse Feature
                    "published": entry.published[:10], # 2024-12-15
                    "pdf_url": pdf_url            # Meta (Dify 跳转用)
                })
            except Exception as e:
                continue
        
        # ⚠️ ArXiv API 规定必须间隔 3 秒
        time.sleep(3)

    # 保存原始数据
    df = pd.DataFrame(all_papers)
    df.to_csv("../../data/arxiv/arxiv_raw_data.csv", index=False, encoding='utf-8-sig')
    print(f"✅ [Step 1] 完成！原始数据已保存至 '../../data/arxiv/arxiv_raw_data.csv' (共 {len(df)} 条)")

if __name__ == "__main__":
    fetch_arxiv_raw()