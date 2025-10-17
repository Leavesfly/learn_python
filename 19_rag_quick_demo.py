# -*- coding: utf-8 -*-
"""
RAG与向量数据演示脚本
"""

from 19_rag_vector_demo import RAGSystem, TFIDFVectorizer, VectorSimilarity

def quick_demo():
    """快速演示RAG系统核心功能"""
    print("🔍 RAG与向量数据快速演示")
    print("=" * 50)
    
    # 创建示例文档
    documents = [
        {
            "id": "python_basics",
            "content": "Python是一种高级编程语言，语法简洁，功能强大，广泛用于数据科学和人工智能开发。",
            "metadata": {"category": "编程语言", "level": "基础"}
        },
        {
            "id": "ml_intro", 
            "content": "机器学习是人工智能的核心分支，通过算法让计算机从数据中自动学习模式和规律。",
            "metadata": {"category": "人工智能", "level": "中级"}
        },
        {
            "id": "deep_learning",
            "content": "深度学习使用多层神经网络模拟人脑处理信息，在图像识别和自然语言处理中表现出色。",
            "metadata": {"category": "人工智能", "level": "高级"}
        },
        {
            "id": "data_science",
            "content": "数据科学结合统计学、计算机科学和领域知识，从海量数据中提取有价值的洞察和知识。",
            "metadata": {"category": "数据科学", "level": "中级"}
        }
    ]
    
    # 创建RAG系统
    print("\n📚 创建RAG系统并添加文档...")
    rag = RAGSystem(vector_dim=64, similarity_threshold=0.05)
    rag.add_documents(documents)
    
    # 显示统计信息
    stats = rag.get_statistics()
    print(f"\n📊 系统统计: {stats}")
    
    # 测试查询
    test_queries = [
        "Python编程语言特点",
        "机器学习算法原理", 
        "神经网络深度学习",
        "数据分析科学方法"
    ]
    
    print("\n🔍 测试查询结果:")
    print("-" * 50)
    
    for query in test_queries:
        print(f"\n查询: '{query}'")
        results = rag.search(query, top_k=2)
        
        if results:
            for i, result in enumerate(results, 1):
                doc = result.document
                print(f"  {i}. [{doc.id}] 相似度: {result.similarity:.4f}")
                print(f"     类别: {doc.metadata.get('category', 'N/A')}")
                print(f"     内容: {doc.content[:60]}...")
        else:
            print("  未找到相关文档")
    
    # 演示向量操作
    print(f"\n🧮 向量操作演示:")
    print("-" * 30)
    
    # 创建简单的向量化器
    vectorizer = TFIDFVectorizer(max_features=20)
    texts = [doc["content"] for doc in documents]
    vectors = vectorizer.fit_transform(texts)
    
    print(f"文档数量: {len(texts)}")
    print(f"向量维度: {len(vectors[0])}")
    print(f"词汇表大小: {vectorizer.tokenizer.vocab_size}")
    
    # 计算文档间相似度
    print(f"\n📏 文档相似度矩阵:")
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            if i <= j:
                sim = VectorSimilarity.cosine_similarity(vectors[i], vectors[j])
                doc_i = documents[i]["id"]
                doc_j = documents[j]["id"]
                print(f"  {doc_i} <-> {doc_j}: {sim:.3f}")
    
    print(f"\n✅ 演示完成！")

if __name__ == "__main__":
    quick_demo()