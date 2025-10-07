#!/usr/bin/env python3
"""
驗證數據不平衡對 GCN Clustering 的影響
"""

import numpy as np
import pickle
from collections import Counter
from sklearn.neighbors import NearestNeighbors

def analyze_knn_graph_imbalance(features_path, labels_path=None, k=80):
    """
    分析 k-NN 圖中的類別分布,驗證不平衡假設
    
    Args:
        features_path: 特徵文件路徑
        labels_path: 標籤文件路徑(如果有ground truth)
        k: k-NN 的 k 值
    """
    print("="*70)
    print("分析 k-NN 圖的類別不平衡問題")
    print("="*70)
    
    # 載入特徵
    features = np.load(features_path)
    print(f"總樣本數: {len(features)}")
    
    # 載入標籤(如果有)
    if labels_path and labels_path.lower() != 'none':
        try:
            labels = np.load(labels_path)
            has_labels = True
            print(f"已載入 ground truth 標籤")
        except:
            has_labels = False
            labels = None
            print("無 ground truth,將進行無監督分析")
    else:
        has_labels = False
        labels = None
        print("無 ground truth,將進行無監督分析")
    
    # 構建 k-NN 圖
    print(f"\n構建 k-NN 圖 (k={k})...")
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(features)
    distances, indices = nbrs.kneighbors(features)
    
    if has_labels:
        # 有標籤:分析每個類別的 k-NN 組成
        analyze_with_labels(labels, indices, k)
    else:
        # 無標籤:通過特徵相似度推測
        analyze_without_labels(features, indices, distances, k)
    
    # 分析特徵空間的密度分布
    analyze_density_distribution(features, distances, k)
    
    return indices, distances

def analyze_with_labels(labels, indices, k):
    """使用 ground truth 分析 k-NN 圖"""
    print("\n" + "="*70)
    print("基於 Ground Truth 的 k-NN 分析")
    print("="*70)
    
    label_counts = Counter(labels)
    unique_labels = sorted(label_counts.keys())
    
    print(f"\n類別分布:")
    for label in unique_labels[:10]:  # 只顯示前10個
        print(f"  類別 {label}: {label_counts[label]} 個樣本")
    if len(unique_labels) > 10:
        print(f"  ... 還有 {len(unique_labels)-10} 個類別")
    
    # 找出最大和最小類別
    largest_class = max(label_counts.items(), key=lambda x: x[1])
    smallest_class = min(label_counts.items(), key=lambda x: x[1])
    
    print(f"\n最大類別: {largest_class[0]} ({largest_class[1]} 個樣本)")
    print(f"最小類別: {smallest_class[0]} ({smallest_class[1]} 個樣本)")
    print(f"不平衡比率: {largest_class[1]/smallest_class[1]:.1f}:1")
    
    # 分析不同大小類別的 k-NN 組成
    print(f"\n" + "="*70)
    print("不同類別的 k-NN 鄰居組成分析")
    print("="*70)
    
    # 選擇幾個代表性類別
    small_classes = [label for label, count in label_counts.items() if count < 50]
    medium_classes = [label for label, count in label_counts.items() if 50 <= count < 200]
    large_classes = [label for label, count in label_counts.items() if count >= 200]
    
    analyze_class_group("小類別 (< 50 樣本)", small_classes, labels, indices, k, label_counts)
    analyze_class_group("中類別 (50-200 樣本)", medium_classes, labels, indices, k, label_counts)
    analyze_class_group("大類別 (>= 200 樣本)", large_classes, labels, indices, k, label_counts)

def analyze_class_group(group_name, class_list, labels, indices, k, label_counts):
    """分析一組類別的 k-NN 組成"""
    if not class_list:
        print(f"\n{group_name}: 無樣本")
        return
    
    print(f"\n{group_name}:")
    print(f"  類別數量: {len(class_list)}")
    
    # 隨機選擇幾個類別詳細分析
    sample_classes = np.random.choice(class_list, min(3, len(class_list)), replace=False)
    
    for target_label in sample_classes:
        # 找出該類別的所有樣本
        class_indices = np.where(labels == target_label)[0]
        
        if len(class_indices) == 0:
            continue
        
        # 分析這些樣本的 k-NN 組成
        neighbor_labels_all = []
        for idx in class_indices:
            # 獲取該樣本的 k 個鄰居(排除自己)
            neighbor_indices = indices[idx, 1:k+1]
            neighbor_labels = labels[neighbor_indices]
            neighbor_labels_all.extend(neighbor_labels)
        
        # 統計鄰居中各類別的佔比
        neighbor_label_counts = Counter(neighbor_labels_all)
        total_neighbors = len(neighbor_labels_all)
        
        same_class_count = neighbor_label_counts[target_label]
        same_class_ratio = same_class_count / total_neighbors
        
        print(f"\n  類別 {target_label} ({label_counts[target_label]} 個樣本):")
        print(f"    k-NN 中同類別佔比: {same_class_ratio*100:.1f}% ({same_class_count}/{total_neighbors})")
        
        # 找出在鄰居中佔比最高的其他類別
        other_class_counts = [(label, count) for label, count in neighbor_label_counts.items() 
                            if label != target_label]
        other_class_counts.sort(key=lambda x: x[1], reverse=True)
        
        print(f"    被入侵最多的類別:")
        for other_label, count in other_class_counts[:3]:
            ratio = count / total_neighbors
            print(f"      類別 {other_label}: {ratio*100:.1f}% "
                  f"({count}/{total_neighbors}, 該類別總共 {label_counts[other_label]} 個樣本)")
        
        # 判斷是否存在嚴重的"入侵"
        if other_class_counts:
            top_invader_label, top_invader_count = other_class_counts[0]
            top_invader_ratio = top_invader_count / total_neighbors
            
            if top_invader_ratio > 0.3:  # 如果某個其他類別佔超過30%
                print(f"    ⚠️  警告: 類別 {top_invader_label} 嚴重入侵了該類別的 k-NN!")
                print(f"       這會導致 GCN 錯誤地連接這兩個類別")

def analyze_without_labels(features, indices, distances, k):
    """無標籤情況下的分析"""
    print("\n" + "="*70)
    print("無監督的 k-NN 密度分析")
    print("="*70)
    
    # 計算每個點的局部密度(k-NN 平均距離)
    avg_distances = np.mean(distances[:, 1:k+1], axis=1)
    
    # 找出密度最高和最低的區域
    density_sorted = np.argsort(avg_distances)
    
    high_density_indices = density_sorted[:100]  # 前100個最密集的點
    low_density_indices = density_sorted[-100:]  # 後100個最稀疏的點
    
    high_density_avg = np.mean(avg_distances[high_density_indices])
    low_density_avg = np.mean(avg_distances[low_density_indices])
    
    print(f"\n高密度區域 (前100個點):")
    print(f"  平均 k-NN 距離: {high_density_avg:.4f}")
    
    print(f"\n低密度區域 (後100個點):")
    print(f"  平均 k-NN 距離: {low_density_avg:.4f}")
    
    print(f"\n密度比率: {low_density_avg/high_density_avg:.2f}:1")
    
    if low_density_avg/high_density_avg > 2:
        print("\n⚠️  警告: 特徵空間中存在明顯的密度不均勻!")
        print("   這通常意味著:")
        print("   1. 某些類別樣本數量遠多於其他類別")
        print("   2. 少數類別的 k-NN 會被多數類別支配")
        print("   3. GCN 容易產生錯誤連接")

def analyze_density_distribution(features, distances, k):
    """分析特徵空間的密度分布"""
    print("\n" + "="*70)
    print("特徵空間密度分布統計")
    print("="*70)
    
    # k-NN 平均距離作為局部密度的指標
    avg_knn_distances = np.mean(distances[:, 1:k+1], axis=1)
    
    print(f"\nk-NN 平均距離統計:")
    print(f"  最小值: {avg_knn_distances.min():.4f}")
    print(f"  25分位: {np.percentile(avg_knn_distances, 25):.4f}")
    print(f"  中位數: {np.median(avg_knn_distances):.4f}")
    print(f"  75分位: {np.percentile(avg_knn_distances, 75):.4f}")
    print(f"  最大值: {avg_knn_distances.max():.4f}")
    print(f"  標準差: {np.std(avg_knn_distances):.4f}")
    
    # 計算變異係數
    cv = np.std(avg_knn_distances) / np.mean(avg_knn_distances)
    print(f"  變異係數: {cv:.4f}")
    
    if cv > 0.5:
        print("\n⚠️  高變異係數表明密度分布很不均勻!")
        print("   這會影響 GCN 的性能")
    
    # 繪製分布直方圖
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.hist(avg_knn_distances, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Average k-NN Distance')
        plt.ylabel('Frequency')
        plt.title('Distribution of Local Density in Feature Space')
        plt.axvline(np.median(avg_knn_distances), color='r', linestyle='--', 
                   label=f'Median: {np.median(avg_knn_distances):.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('knn_density_distribution.png', dpi=150, bbox_inches='tight')
        print(f"\n密度分布圖已保存: knn_density_distribution.png")
        plt.close()
    except:
        print("\n無法繪製分布圖(需要 matplotlib)")

def generate_recommendations(indices, distances, labels=None, k=80):
    """根據分析結果生成建議"""
    print("\n" + "="*70)
    print("針對你的數據的建議")
    print("="*70)
    
    if labels is not None:
        label_counts = Counter(labels)
        largest_class_size = max(label_counts.values())
        smallest_class_size = min(label_counts.values())
        imbalance_ratio = largest_class_size / smallest_class_size
        
        print(f"\n數據不平衡比率: {imbalance_ratio:.1f}:1")
        
        if imbalance_ratio > 20:
            print("\n❌ 嚴重不平衡! GCN 很可能會失敗")
            print("\n強烈建議:")
            print("  1. 使用 Chinese Whispers (已實現)")
            print("  2. 升級到 ArcFace 特徵")
            print("  3. 應用你的 post-processing")
            
        elif imbalance_ratio > 10:
            print("\n⚠️  中度不平衡,GCN 可能表現不佳")
            print("\n建議:")
            print("  1. 嘗試降低 k 值到 20-30")
            print("  2. 使用加權的 k-NN 圖")
            print("  3. 或者使用 Chinese Whispers")
            
        else:
            print("\n✓ 相對平衡,GCN 應該可以工作")
            print("\n建議:")
            print("  1. 使用論文推薦的參數 k=80")
            print("  2. 微調 GCN 的閾值")
    
    # 無標籤情況下的建議
    avg_knn_distances = np.mean(distances[:, 1:k+1], axis=1)
    cv = np.std(avg_knn_distances) / np.mean(avg_knn_distances)
    
    if cv > 0.5:
        print("\n⚠️  特徵空間密度不均勻")
        print("\n建議:")
        print("  1. 使用對不平衡數據更魯棒的方法")
        print("  2. 考慮重新訓練特徵提取器")
        print("  3. 應用特徵空間的歸一化")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='分析 k-NN 圖的不平衡問題')
    parser.add_argument('--features', type=str, required=True,
                       help='特徵文件路徑 (.npy)')
    parser.add_argument('--labels', type=str, default=None,
                       help='標籤文件路徑 (.npy), 可選')
    parser.add_argument('--k', type=int, default=80,
                       help='k-NN 的 k 值 (default: 80)')
    
    args = parser.parse_args()
    
    # 執行分析
    indices, distances = analyze_knn_graph_imbalance(
        args.features, 
        args.labels, 
        args.k
    )
    
    # 生成建議
    if args.labels and args.labels.lower() != 'none':
        labels = np.load(args.labels)
    else:
        labels = None
    
    generate_recommendations(indices, distances, labels, args.k)
    
    print("\n" + "="*70)
    print("分析完成!")
    print("="*70)