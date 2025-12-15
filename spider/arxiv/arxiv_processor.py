import pandas as pd
import ast

# === 🎓 模拟科研人员画像 ===
# 场景 A: 专注于大语言模型 (LLM) 和 Agent 的研究生
RESEARCHER_LLM = {
    "name": "Researcher_LLM",
    "interest_keywords": ["LLM", "Large Language Model", "Agent", "RAG", "Prompt", "Transformer", "Generative"],
    "ignore_keywords": ["Image", "Detection", "Segmentation", "3D", "Reinforcement Learning"] # 对纯 CV 不感兴趣
}

# 场景 B: 专注于计算机视觉 (CV) 的研究生
RESEARCHER_CV = {
    "name": "Researcher_CV",
    "interest_keywords": ["Vision", "Image", "Detection", "3D", "Segmentation", "Object", "Diffusion"],
    "ignore_keywords": ["Language", "Text", "NLP", "Audio"]
}

# 场景 C: 网络安全与对抗攻击 (Security & Safety)
# 🎯 特点：喜欢找漏洞、防御、隐私保护，对纯粹的模型架构优化不感兴趣
RESEARCHER_SEC = {
    "name": "Researcher_Security",
    "interest_keywords": ["Adversarial", "Attack", "Defense", "Privacy", "Security", "Backdoor", "Poisoning", "Robustness", "Safety"],
    "ignore_keywords": ["Gaming", "Art", "Music", "Recommendation", "UI/UX"] 
}

# 场景 D: 图神经网络与推荐算法 (Graph & RecSys)
# 🎯 特点：喜欢社交网络、节点分类、推荐系统，这跟纯图像或纯文本处理差别很大
RESEARCHER_GRAPH = {
    "name": "Researcher_Graph",
    "interest_keywords": ["Graph", "GNN", "Node", "Link Prediction", "Social Network", "Recommendation", "Collaborative Filtering", "Contrastive Learning"],
    "ignore_keywords": ["Transformer", "Language", "Robot", "Hardware"] # 稍微排斥纯语言模型，更关注结构化数据
}

# 场景 E: 边缘计算与模型压缩 (Edge AI & Systems)
# 🎯 特点：这是一种“工程派”画像。他们不关心模型有多聪明，只关心模型有多快、多小。
RESEARCHER_SYS = {
    "name": "Researcher_System",
    "interest_keywords": ["Quantization", "Compression", "Pruning", "Edge", "Latency", "Efficient", "Mobile", "Hardware", "FPGA", "Accelerator"],
    "ignore_keywords": ["Theory", "Proof", "Survey", "Ethics"] # 讨厌纯理论证明或伦理讨论
}

# 场景 F: 多模态与AIGC (Multimodal & AIGC)
# 🎯 特点：什么都要——既要图又要文，甚至要视频。这是当下的“缝合怪”热点。
RESEARCHER_MULTI = {
    "name": "Researcher_Multimodal",
    "interest_keywords": ["Multimodal", "Video", "Audio", "Text-to-Image", "Diffusion", "CLIP", "Alignment", "Cross-modal"],
    "ignore_keywords": ["Security", "Encryption", "Database", "Network"] # 对底层设施不感兴趣
}

def format_authors(authors_str):
    """把作者列表转为 'Li et al.' 格式，省空间"""
    try:
        if isinstance(authors_str, str):
            authors = ast.literal_eval(authors_str)
        else:
            authors = authors_str
        
        if not authors: return "Unknown"
        if len(authors) > 1:
            # 取第一个作者的姓 (简单处理)
            first_name = authors[0].split(' ')[-1] 
            return f"{first_name} et al."
        else:
            return authors[0]
    except:
        return "Unknown"

def generate_academic_label(row, profile):
    """
    根据【标题+摘要】的关键词匹配生成 Label
    """
    score = 0.5
    
    # 拼接标题和摘要进行检索
    content = (str(row['title']) + " " + str(row['abstract'])).lower()
    
    # 1. 兴趣词匹配 (命中加分)
    for kw in profile['interest_keywords']:
        if kw.lower() in content:
            score += 0.3
            # 只要命中一个核心词，概率就很大
            break 
            
    # 2. 屏蔽词匹配 (命中减分)
    for kw in profile['ignore_keywords']:
        if kw.lower() in content:
            score -= 0.4
            
    # 3. 引用数/热度模拟 (如果没有真实引用数，可以用发表时间模拟：越新越容易点)
    # 这里简单假设：如果是 cs.AI 或 cs.CL 类别的，基础分高一点
    if row['category'] in ['cs.CL', 'cs.AI'] and "LLM" in profile['name']:
        score += 0.1
        
    return 1 if score >= 0.6 else 0

def process_arxiv(profile):
    print(f"⚙️ [Step 2] 正在为用户 [{profile['name']}] 生成训练数据...")
    
    df = pd.read_csv("../../data/arxiv/arxiv_raw_data.csv")
    
    # 1. 格式化作者 (给 Dify 看的)
    df['display_authors'] = df['authors_raw'].apply(format_authors)
    
    # 2. 生成 Label (给 DeepFM 训练用的)
    df['label'] = df.apply(lambda row: generate_academic_label(row, profile), axis=1)
    
    # 3. 关键：为 DeepFM 准备 Embedding 接口
    # 注意：DeepFM 无法直接训练 string 类型的 abstract。
    # 在这里我们不进行 BERT 转换（太慢），而是保留 text，
    # 并在 Dataset Loader 阶段或者单独脚本里做 text -> vector
    
    # 导出
    output_file = f"../../data/arxiv/train_arxiv_{profile['name']}.csv"
    cols = ['item_id', 'title', 'category', 'abstract', 'display_authors', 'published', 'pdf_url', 'label']
    df[cols].to_csv(output_file, index=False)
    
    print(f"   ✅ 生成完毕: {output_file}")
    print(f"   📊 正样本(感兴趣)比例: {df['label'].mean():.2%}\n")

if __name__ == "__main__":
# 定义一个画像列表
    profiles = [
        RESEARCHER_LLM, 
        RESEARCHER_CV, 
        RESEARCHER_SEC, 
        RESEARCHER_GRAPH, 
        RESEARCHER_SYS,
        RESEARCHER_MULTI
    ]

    # 循环生成所有训练集
    for p in profiles:
        process_arxiv(p)