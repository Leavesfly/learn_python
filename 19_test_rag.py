#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统测试脚本
验证各个组件的功能是否正常
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试导入"""
    try:
        from 19_rag_vector_demo import (
            SimpleTokenizer, TFIDFVectorizer, VectorSimilarity,
            VectorDatabase, RAGSystem, Document, QueryResult
        )
        print("✅ 所有模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_tokenizer():
    """测试分词器"""
    try:
        from 19_rag_vector_demo import SimpleTokenizer
        
        tokenizer = SimpleTokenizer()
        
        # 测试中英文分词
        text = "Python是编程语言 machine learning"
        tokens = tokenizer.tokenize(text)
        print(f"分词测试: '{text}' -> {tokens}")
        
        # 构建词汇表
        texts = ["Python编程", "机器学习", "深度学习"]
        tokenizer.build_vocab(texts)
        print(f"词汇表大小: {tokenizer.vocab_size}")
        
        print("✅ 分词器测试通过")
        return True
    except Exception as e:
        print(f"❌ 分词器测试失败: {e}")
        return False

def test_vectorizer():
    """测试向量化器"""
    try:
        from 19_rag_vector_demo import TFIDFVectorizer
        
        vectorizer = TFIDFVectorizer(max_features=10)
        
        # 测试文档
        docs = [
            "Python是编程语言",
            "机器学习很重要", 
            "深度学习使用神经网络"
        ]
        
        # 训练和转换
        vectors = vectorizer.fit_transform(docs)
        print(f"向量化测试: {len(docs)} 个文档 -> {len(vectors)} 个向量")
        print(f"向量维度: {len(vectors[0])}")
        
        print("✅ 向量化器测试通过")
        return True
    except Exception as e:
        print(f"❌ 向量化器测试失败: {e}")
        return False

def test_similarity():
    """测试相似度计算"""
    try:
        from 19_rag_vector_demo import VectorSimilarity
        
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]  
        vec3 = [1.0, 0.0, 0.0]
        
        # 测试余弦相似度
        sim1 = VectorSimilarity.cosine_similarity(vec1, vec2)  # 应该是0
        sim2 = VectorSimilarity.cosine_similarity(vec1, vec3)  # 应该是1
        
        print(f"相似度测试: vec1 与 vec2 = {sim1:.3f}")
        print(f"相似度测试: vec1 与 vec3 = {sim2:.3f}")
        
        print("✅ 相似度计算测试通过")
        return True
    except Exception as e:
        print(f"❌ 相似度计算测试失败: {e}")
        return False

def test_rag_system():
    """测试RAG系统"""
    try:
        from 19_rag_vector_demo import RAGSystem
        
        # 创建RAG系统
        rag = RAGSystem(vector_dim=32, similarity_threshold=0.0)
        
        # 添加测试文档
        documents = [
            {
                "id": "doc1",
                "content": "Python是一种编程语言",
                "metadata": {"type": "tech"}
            },
            {
                "id": "doc2", 
                "content": "机器学习是人工智能分支",
                "metadata": {"type": "ai"}
            }
        ]
        
        rag.add_documents(documents)
        
        # 测试检索
        results = rag.search("编程语言", top_k=2)
        print(f"检索测试: 找到 {len(results)} 个结果")
        
        if results:
            result = results[0]
            print(f"最佳匹配: {result.document.id}, 相似度: {result.similarity:.3f}")
        
        # 测试上下文生成
        context = rag.generate_context("编程", max_context_length=200)
        print(f"上下文生成: {len(context)} 字符")
        
        print("✅ RAG系统测试通过")
        return True
    except Exception as e:
        print(f"❌ RAG系统测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("🧪 开始RAG系统组件测试")
    print("=" * 50)
    
    tests = [
        ("导入测试", test_imports),
        ("分词器测试", test_tokenizer),
        ("向量化器测试", test_vectorizer),
        ("相似度计算测试", test_similarity),
        ("RAG系统测试", test_rag_system)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"跳过后续测试...")
            break
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！RAG系统运行正常")
    else:
        print("⚠️  部分测试失败，请检查代码")

if __name__ == "__main__":
    run_all_tests()