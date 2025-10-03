import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class FeatureQualityDiagnostics:
    def __init__(self, features, labels=None, video_name=""):
        """
        Args:
            features: numpy array of shape (N, D), N samples, D dimensions
            labels: numpy array of shape (N,), ground truth labels (optional)
            video_name: string, name of the video for saving plots
        """
        self.features = features
        self.labels = labels
        self.video_name = video_name
        self.has_labels = labels is not None
        
        # Normalize features
        self.features_normalized = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        
        print(f"=== Feature Quality Diagnostics for {video_name} ===")
        print(f"Total samples: {len(features)}")
        print(f"Feature dimension: {features.shape[1]}")
        if self.has_labels:
            print(f"True number of identities: {len(np.unique(labels))}")
            label_counts = Counter(labels)
            print(f"Number of identities: {len(label_counts)}")
            print(f"Samples per identity - Min: {min(label_counts.values())}, Max: {max(label_counts.values())}, Mean: {np.mean(list(label_counts.values())):.1f}")
        print()
    
    def compute_similarity_matrix(self):
        """計算cosine similarity matrix"""
        print("Computing similarity matrix...")
        self.sim_matrix = cosine_similarity(self.features_normalized)
        self.dist_matrix = 1 - self.sim_matrix
        
        # 計算統計信息（排除對角線）
        mask = ~np.eye(self.sim_matrix.shape[0], dtype=bool)
        sim_values = self.sim_matrix[mask]
        
        print(f"Similarity range: [{sim_values.min():.3f}, {sim_values.max():.3f}]")
        print(f"Mean similarity: {sim_values.mean():.3f}")
        print(f"Median similarity: {np.median(sim_values):.3f}")
        print(f"Std similarity: {sim_values.std():.3f}")
        print()
        
    def analyze_similarity_distribution(self):
        """分析similarity分布"""
        print("=== Similarity Distribution Analysis ===")
        
        if not self.has_labels:
            print("No ground truth labels provided. Showing overall distribution only.")
            plt.figure(figsize=(12, 5))
            
            # Overall distribution (排除對角線)
            mask = ~np.eye(self.sim_matrix.shape[0], dtype=bool)
            all_sims = self.sim_matrix[mask]
            
            plt.subplot(1, 2, 1)
            plt.hist(all_sims, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Frequency')
            plt.title(f'Overall Similarity Distribution - {self.video_name}')
            plt.axvline(all_sims.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {all_sims.mean():.3f}')
            plt.axvline(np.median(all_sims), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(all_sims):.3f}')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            # Cumulative distribution
            plt.subplot(1, 2, 2)
            sorted_sims = np.sort(all_sims)
            cumulative = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
            plt.plot(sorted_sims, cumulative, linewidth=2)
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Cumulative Probability')
            plt.title(f'Cumulative Distribution - {self.video_name}')
            plt.grid(True, alpha=0.3)
            
            # 標注關鍵百分位數
            for percentile in [25, 50, 75, 90]:
                value = np.percentile(all_sims, percentile)
                plt.axvline(value, color='red', linestyle=':', alpha=0.5)
                plt.text(value, percentile/100, f'  {percentile}%: {value:.3f}', 
                        rotation=0, verticalalignment='bottom')
            
            plt.tight_layout()
            plt.savefig(f'similarity_dist_{self.video_name}.png', dpi=150)
            plt.show()
            
            # 分析分布特性
            print(f"\nDistribution Statistics:")
            print(f"  25th percentile: {np.percentile(all_sims, 25):.3f}")
            print(f"  50th percentile: {np.percentile(all_sims, 50):.3f}")
            print(f"  75th percentile: {np.percentile(all_sims, 75):.3f}")
            print(f"  90th percentile: {np.percentile(all_sims, 90):.3f}")
            print(f"  95th percentile: {np.percentile(all_sims, 95):.3f}")
            
            # 判斷分布是否過於集中
            high_sim_ratio = (all_sims > 0.7).sum() / len(all_sims)
            print(f"\nPercentage of pairs with similarity > 0.7: {high_sim_ratio*100:.1f}%")
            
            if high_sim_ratio > 0.5:
                print("⚠ WARNING: Too many high-similarity pairs detected!")
                print("  This suggests features are not discriminative enough.")
                print("  → Strongly recommend switching to ArcFace")
            elif all_sims.std() < 0.15:
                print("⚠ WARNING: Very low variance in similarity scores!")
                print("  Features may be too uniform.")
                print("  → Consider using stronger feature extractor")
            
            print()
            return
        
        # 如果有labels，分析positive和negative pairs
        positive_sims = []
        negative_sims = []
        
        print("Analyzing positive and negative pairs...")
        for i in range(len(self.labels)):
            for j in range(i+1, len(self.labels)):
                sim = self.sim_matrix[i, j]
                if self.labels[i] == self.labels[j]:
                    positive_sims.append(sim)
                else:
                    negative_sims.append(sim)
        
        positive_sims = np.array(positive_sims)
        negative_sims = np.array(negative_sims)
        
        print(f"Positive pairs (same identity): {len(positive_sims)}")
        print(f"  Mean: {positive_sims.mean():.3f}")
        print(f"  Std: {positive_sims.std():.3f}")
        print(f"  Min: {positive_sims.min():.3f}")
        print(f"  Max: {positive_sims.max():.3f}")
        print(f"  Median: {np.median(positive_sims):.3f}")
        print()
        
        print(f"Negative pairs (different identity): {len(negative_sims)}")
        print(f"  Mean: {negative_sims.mean():.3f}")
        print(f"  Std: {negative_sims.std():.3f}")
        print(f"  Min: {negative_sims.min():.3f}")
        print(f"  Max: {negative_sims.max():.3f}")
        print(f"  Median: {np.median(negative_sims):.3f}")
        print()
        
        # 計算分離度
        separation = positive_sims.mean() - negative_sims.mean()
        print(f"Separation (pos_mean - neg_mean): {separation:.3f}")
        
        # 計算overlap
        pos_25 = np.percentile(positive_sims, 25)
        neg_75 = np.percentile(negative_sims, 75)
        overlap_pct = (negative_sims > pos_25).sum() / len(negative_sims) * 100
        print(f"Overlap: {overlap_pct:.1f}% of negative pairs > 25th percentile of positive pairs")
        
        # 計算最佳threshold的理論值
        # 使用Otsu方法的概念
        best_threshold = None
        best_separation = 0
        for threshold in np.linspace(negative_sims.max(), positive_sims.min(), 100):
            tp = (positive_sims >= threshold).sum()
            tn = (negative_sims < threshold).sum()
            fp = (negative_sims >= threshold).sum()
            fn = (positive_sims < threshold).sum()
            
            if tp + fp > 0 and tn + fn > 0:
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                if f1 > best_separation:
                    best_separation = f1
                    best_threshold = threshold
        
        if best_threshold is not None:
            print(f"\nOptimal threshold (by F1): {best_threshold:.3f}")
            print(f"  Best F1 score: {best_separation:.3f}")
        print()
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Histogram
        axes[0, 0].hist([positive_sims, negative_sims], bins=50, alpha=0.6, 
                 label=['Positive (same ID)', 'Negative (diff ID)'],
                 color=['green', 'red'])
        axes[0, 0].set_xlabel('Cosine Similarity')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'Similarity Distribution')
        axes[0, 0].legend()
        axes[0, 0].axvline(positive_sims.mean(), color='green', linestyle='--', alpha=0.7, linewidth=2)
        axes[0, 0].axvline(negative_sims.mean(), color='red', linestyle='--', alpha=0.7, linewidth=2)
        if best_threshold is not None:
            axes[0, 0].axvline(best_threshold, color='blue', linestyle=':', linewidth=2, label=f'Optimal: {best_threshold:.3f}')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Box plot
        axes[0, 1].boxplot([positive_sims, negative_sims], labels=['Positive', 'Negative'])
        axes[0, 1].set_ylabel('Cosine Similarity')
        axes[0, 1].set_title('Box Plot Comparison')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Cumulative distribution
        sorted_pos = np.sort(positive_sims)
        sorted_neg = np.sort(negative_sims)
        axes[1, 0].plot(sorted_pos, np.arange(1, len(sorted_pos)+1)/len(sorted_pos), 
                       label='Positive', color='green', linewidth=2)
        axes[1, 0].plot(sorted_neg, np.arange(1, len(sorted_neg)+1)/len(sorted_neg), 
                       label='Negative', color='red', linewidth=2)
        axes[1, 0].set_xlabel('Cosine Similarity')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].set_title('Cumulative Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Density plot
        from scipy import stats
        pos_density = stats.gaussian_kde(positive_sims)
        neg_density = stats.gaussian_kde(negative_sims)
        xs = np.linspace(min(negative_sims.min(), positive_sims.min()), 
                        max(negative_sims.max(), positive_sims.max()), 200)
        axes[1, 1].plot(xs, pos_density(xs), label='Positive', color='green', linewidth=2)
        axes[1, 1].plot(xs, neg_density(xs), label='Negative', color='red', linewidth=2)
        axes[1, 1].fill_between(xs, pos_density(xs), alpha=0.3, color='green')
        axes[1, 1].fill_between(xs, neg_density(xs), alpha=0.3, color='red')
        axes[1, 1].set_xlabel('Cosine Similarity')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Kernel Density Estimation')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Feature Quality Analysis - {self.video_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'similarity_analysis_{self.video_name}.png', dpi=150)
        plt.show()
        
        # 判斷feature quality
        print("=== Feature Quality Assessment ===")
        if separation > 0.15 and overlap_pct < 30:
            print("✓ GOOD: Features show clear separation between identities")
            print("   → Features are discriminative enough")
        elif separation > 0.08 and overlap_pct < 50:
            print("⚠ MODERATE: Features have some discriminative power but overlap significantly")
            print("   → Consider switching to ArcFace for better performance")
        else:
            print("✗ POOR: Features cannot distinguish identities well")
            print("   → Recommendation: Switch to ArcFace or stronger feature extractor")
        print()
    
    def test_clustering_methods(self, n_clusters_list=None):
        """測試不同clustering方法"""
        print("=== Clustering Method Comparison ===")
        
        if n_clusters_list is None:
            if self.has_labels:
                true_k = len(np.unique(self.labels))
                n_clusters_list = [
                    max(2, true_k - 10),
                    max(2, true_k - 5),
                    true_k,
                    true_k + 5,
                    true_k + 10
                ]
            else:
                # 如果沒有labels，測試多個可能的cluster數
                n_clusters_list = [10, 20, 30, 40, 50]
        
        results = []
        
        for n_clusters in n_clusters_list:
            print(f"\nTesting with n_clusters = {n_clusters}")
            
            # 1. Agglomerative Clustering (Average Linkage) - 使用precomputed distance
            try:
                agc_avg = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',  # 修正：使用affinity而不是metric
                    linkage='average'
                )
                labels_agc_avg = agc_avg.fit_predict(self.dist_matrix)
            except Exception as e:
                print(f"  AGC-Average failed: {e}")
                labels_agc_avg = None
            
            # 2. Agglomerative Clustering (Complete Linkage)
            try:
                agc_complete = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',  # 修正：使用affinity而不是metric
                    linkage='complete'
                )
                labels_agc_complete = agc_complete.fit_predict(self.dist_matrix)
            except Exception as e:
                print(f"  AGC-Complete failed: {e}")
                labels_agc_complete = None
            
            # 3. K-Means
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels_kmeans = kmeans.fit_predict(self.features_normalized)
            except Exception as e:
                print(f"  K-Means failed: {e}")
                labels_kmeans = None
            
            # Compute metrics
            methods = {
                'AGC-Average': labels_agc_avg,
                'AGC-Complete': labels_agc_complete,
                'K-Means': labels_kmeans
            }
            
            for method_name, pred_labels in methods.items():
                if pred_labels is None:
                    continue
                    
                try:
                    silhouette = silhouette_score(self.features_normalized, pred_labels, metric='cosine')
                    calinski = calinski_harabasz_score(self.features_normalized, pred_labels)
                    davies = davies_bouldin_score(self.features_normalized, pred_labels)
                    
                    result = {
                        'n_clusters': n_clusters,
                        'method': method_name,
                        'silhouette': silhouette,
                        'calinski_harabasz': calinski,
                        'davies_bouldin': davies
                    }
                    
                    # 如果有ground truth，計算accuracy-related metrics
                    if self.has_labels:
                        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
                        result['ari'] = adjusted_rand_score(self.labels, pred_labels)
                        result['nmi'] = normalized_mutual_info_score(self.labels, pred_labels)
                    
                    results.append(result)
                    
                    print(f"  {method_name:15s} | Silhouette: {silhouette:.3f} | "
                          f"Calinski: {calinski:.1f} | Davies-Bouldin: {davies:.3f}", end='')
                    if self.has_labels:
                        print(f" | ARI: {result['ari']:.3f} | NMI: {result['nmi']:.3f}")
                    else:
                        print()
                except Exception as e:
                    print(f"  {method_name:15s} | Error: {e}")
        
        if not results:
            print("\n⚠ WARNING: All clustering methods failed!")
            print("  This might indicate issues with the data or sklearn version.")
            return []
        
        # 找出最佳結果
        print("\n=== Best Results by Metric ===")
        
        best_silhouette = max(results, key=lambda x: x['silhouette'])
        print(f"Best Silhouette Score: {best_silhouette['silhouette']:.3f}")
        print(f"  → {best_silhouette['method']} with n_clusters={best_silhouette['n_clusters']}")
        
        best_calinski = max(results, key=lambda x: x['calinski_harabasz'])
        print(f"Best Calinski-Harabasz Score: {best_calinski['calinski_harabasz']:.1f}")
        print(f"  → {best_calinski['method']} with n_clusters={best_calinski['n_clusters']}")
        
        best_davies = min(results, key=lambda x: x['davies_bouldin'])
        print(f"Best Davies-Bouldin Score: {best_davies['davies_bouldin']:.3f} (lower is better)")
        print(f"  → {best_davies['method']} with n_clusters={best_davies['n_clusters']}")
        
        if self.has_labels:
            best_ari = max(results, key=lambda x: x['ari'])
            print(f"Best ARI Score: {best_ari['ari']:.3f}")
            print(f"  → {best_ari['method']} with n_clusters={best_ari['n_clusters']}")
        
        # 評估
        print("\n=== Overall Feature Quality Assessment ===")
        best_sil = best_silhouette['silhouette']
        if best_sil > 0.4:
            print("✓ EXCELLENT: Silhouette > 0.4, features are highly discriminative")
            print("   → Your features are good enough, problem likely in GCN or threshold")
        elif best_sil > 0.25:
            print("✓ GOOD: Silhouette > 0.25, features have reasonable discriminative power")
            print("   → Consider fine-tuning GCN parameters or using adaptive threshold")
        elif best_sil > 0.15:
            print("⚠ MODERATE: Silhouette > 0.15, features have limited discriminative power")
            print("   → Recommend switching to ArcFace for better features")
        else:
            print("✗ POOR: Silhouette < 0.15, features cannot separate identities well")
            print("   → MUST switch to ArcFace or retrain feature extractor")
        print()
        
        # 繪製結果比較圖
        self._plot_clustering_comparison(results)
        
        return results
    
    def _plot_clustering_comparison(self, results):
        """繪製clustering結果比較圖"""
        if not results:
            return
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        methods = sorted(set([r['method'] for r in results]))
        n_clusters_vals = sorted(set([r['n_clusters'] for r in results]))
        
        # 為每個method準備數據
        data = {method: {'n_clusters': [], 'silhouette': [], 'calinski': [], 'davies': []} 
                for method in methods}
        
        for r in results:
            data[r['method']]['n_clusters'].append(r['n_clusters'])
            data[r['method']]['silhouette'].append(r['silhouette'])
            data[r['method']]['calinski'].append(r['calinski_harabasz'])
            data[r['method']]['davies'].append(r['davies_bouldin'])
        
        # Plot 1: Silhouette Score
        for method in methods:
            axes[0].plot(data[method]['n_clusters'], data[method]['silhouette'], 
                        marker='o', label=method, linewidth=2)
        axes[0].set_xlabel('Number of Clusters')
        axes[0].set_ylabel('Silhouette Score')
        axes[0].set_title('Silhouette Score vs. K')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Calinski-Harabasz Score
        for method in methods:
            axes[1].plot(data[method]['n_clusters'], data[method]['calinski'], 
                        marker='o', label=method, linewidth=2)
        axes[1].set_xlabel('Number of Clusters')
        axes[1].set_ylabel('Calinski-Harabasz Score')
        axes[1].set_title('Calinski-Harabasz Score vs. K')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Davies-Bouldin Score (lower is better)
        for method in methods:
            axes[2].plot(data[method]['n_clusters'], data[method]['davies'], 
                        marker='o', label=method, linewidth=2)
        axes[2].set_xlabel('Number of Clusters')
        axes[2].set_ylabel('Davies-Bouldin Score')
        axes[2].set_title('Davies-Bouldin Score vs. K (lower is better)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Clustering Performance Comparison - {self.video_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'clustering_comparison_{self.video_name}.png', dpi=150)
        plt.show()
    
    def test_dbscan(self, eps_list=None, min_samples=3):
        """測試DBSCAN (不需要預先指定cluster數)"""
        print("=== DBSCAN Analysis ===")
        
        if eps_list is None:
            # 根據distance matrix自動選擇eps範圍
            distances = self.dist_matrix[np.triu_indices_from(self.dist_matrix, k=1)]
            eps_list = [
                np.percentile(distances, 5),
                np.percentile(distances, 10),
                np.percentile(distances, 15),
                np.percentile(distances, 20),
                np.percentile(distances, 25)
            ]
            print(f"Auto-selected eps values based on distance percentiles:")
            print(f"  {[f'{e:.3f}' for e in eps_list]}")
        
        best_result = None
        best_silhouette = -1
        
        for eps in eps_list:
            try:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
                labels_dbscan = dbscan.fit_predict(self.dist_matrix)
                
                n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
                n_noise = list(labels_dbscan).count(-1)
                
                if n_clusters > 1:
                    # 只對非noise點計算silhouette
                    non_noise_mask = labels_dbscan != -1
                    if non_noise_mask.sum() > n_clusters:
                        silhouette = silhouette_score(
                            self.features_normalized[non_noise_mask],
                            labels_dbscan[non_noise_mask],
                            metric='cosine'
                        )
                    else:
                        silhouette = -1
                else:
                    silhouette = -1
                
                print(f"eps={eps:.3f} | Clusters: {n_clusters:3d} | Noise: {n_noise:4d} ({n_noise/len(self.labels)*100:5.1f}%) | Silhouette: {silhouette:.3f}")
                
                if silhouette > best_silhouette and n_clusters > 1:
                    best_silhouette = silhouette
                    best_result = {
                        'eps': eps,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'silhouette': silhouette,
                        'labels': labels_dbscan
                    }
            except Exception as e:
                print(f"eps={eps:.3f} | Error: {e}")
        
        if best_result:
            print(f"\nBest DBSCAN result:")
            print(f"  eps={best_result['eps']:.3f}")
            print(f"  Found {best_result['n_clusters']} clusters")
            print(f"  {best_result['n_noise']} noise points ({best_result['n_noise']/len(self.labels)*100:.1f}%)")
            print(f"  Silhouette: {best_result['silhouette']:.3f}")
        else:
            print("\n⚠ WARNING: Could not find valid DBSCAN clustering")
            print("  Try adjusting eps_list or min_samples parameters")
        print()
        
        return best_result
    
    def visualize_tsne(self, perplexity=30, n_samples=None):
        """使用t-SNE視覺化features"""
        print("=== t-SNE Visualization ===")
        
        # 如果樣本太多，隨機採樣
        if n_samples and len(self.features) > n_samples:
            indices = np.random.choice(len(self.features), n_samples, replace=False)
            features_vis = self.features_normalized[indices]
            labels_vis = self.labels[indices] if self.has_labels else None
            print(f"Sampling {n_samples} points for visualization")
        else:
            features_vis = self.features_normalized
            labels_vis = self.labels
            n_samples = len(features_vis)
        
        # 調整perplexity（不能大於樣本數-1）
        max_perplexity = min(perplexity, n_samples - 1)
        if max_perplexity < perplexity:
            print(f"Adjusting perplexity from {perplexity} to {max_perplexity} (max allowed)")
            perplexity = max_perplexity
        
        print(f"Running t-SNE with perplexity={perplexity}...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, 
                    metric='cosine', n_iter=1000, verbose=0)
        features_2d = tsne.fit_transform(features_vis)
        
        plt.figure(figsize=(12, 10))
        
        if self.has_labels and labels_vis is not None:
            # 用ground truth labels上色
            unique_labels = np.unique(labels_vis)
            n_colors = len(unique_labels)
            
            # 使用更多顏色
            if n_colors <= 10:
                colors = plt.cm.tab10(np.linspace(0, 1, 10))
            elif n_colors <= 20:
                colors = plt.cm.tab20(np.linspace(0, 1, 20))
            else:
                colors = plt.cm.gist_rainbow(np.linspace(0, 1, n_colors))
            
            for idx, label in enumerate(unique_labels):
                mask = labels_vis == label
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                           c=[colors[idx % len(colors)]], label=f'ID-{label}', 
                           alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
            
            # 如果identity太多，不顯示legend
            if n_colors <= 30:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=max(1, n_colors//15), fontsize=8)
            else:
                plt.text(0.02, 0.98, f'{n_colors} identities (legend omitted)', 
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.title(f't-SNE Visualization (colored by ground truth) - {self.video_name}\n{n_colors} identities, {n_samples} samples')
        else:
            # 沒有labels，只顯示點
            plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.5, s=30, c='blue', edgecolors='black', linewidths=0.5)
            plt.title(f't-SNE Visualization - {self.video_name}\n{n_samples} samples')
        
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        plt.savefig(f'tsne_{self.video_name}.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("t-SNE visualization saved.\n")
    
    def analyze_knn_structure(self, k_values=[5, 10, 20, 40, 80]):
        """分析k-nearest neighbor結構 - 這對理解GCN的IPS構建很重要"""
        print("=== k-NN Structure Analysis ===")
        print("This analysis helps understand if your k1, k2 parameters in GCN are appropriate.\n")
        
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=max(k_values)+1, metric='cosine')
        nbrs.fit(self.features_normalized)
        distances, indices = nbrs.kneighbors(self.features_normalized)
        
        for k in k_values:
            print(f"k = {k}:")
            
            # 計算k-nn的平均distance
            k_distances = distances[:, 1:k+1]  # 排除自己 (index 0)
            mean_dist = k_distances.mean()
            std_dist = k_distances.std()
            
            print(f"  Mean k-NN distance: {mean_dist:.3f} (±{std_dist:.3f})")
            print(f"  Distance range: [{k_distances.min():.3f}, {k_distances.max():.3f}]")
            
            if self.has_labels:
                # 計算k-nn中same identity的比例
                same_id_counts = []
                for i in range(len(self.labels)):
                    neighbor_labels = self.labels[indices[i, 1:k+1]]  # 排除自己
                    same_id_count = (neighbor_labels == self.labels[i]).sum()
                    same_id_counts.append(same_id_count)
                
                same_id_counts = np.array(same_id_counts)
                avg_same_id = same_id_counts.mean()
                
                print(f"  Average same-identity neighbors: {avg_same_id:.1f} / {k} ({avg_same_id/k*100:.1f}%)")
                print(f"  Min: {same_id_counts.min()}, Max: {same_id_counts.max()}")
                
                # 理想情況分析
                if avg_same_id / k > 0.8:
                    print(f"  ✓ Good: Most neighbors are from same identity")
                elif avg_same_id / k > 0.5:
                    print(f"  ⚠ Moderate: About half neighbors are from same identity")
                else:
                    print(f"  ✗ Poor: Most neighbors are from different identities")
                    print(f"     → This suggests features are not discriminative")
            
            print()
        
        # 建議適當的k1值
        if self.has_labels:
            print("=== Recommendations for GCN Parameters ===")
            
            # 找出能包含大部分same-identity的k值
            for k in k_values:
                same_id_counts = []
                for i in range(len(self.labels)):
                    neighbor_labels = self.labels[indices[i, 1:k+1]]
                    same_id_count = (neighbor_labels == self.labels[i]).sum()
                    same_id_counts.append(same_id_count)
                
                avg_same_id_ratio = np.mean(same_id_counts) / k
                
                if avg_same_id_ratio > 0.7:
                    print(f"✓ k1={k} is a good choice (captures {avg_same_id_ratio*100:.1f}% same-identity neighbors)")
                    break
            else:
                print(f"⚠ WARNING: Even with k={max(k_values)}, cannot capture enough same-identity neighbors")
                print(f"   → This suggests features need improvement")
            
            print()
    
    def run_full_diagnostics(self, n_clusters_list=None, visualize=True, analyze_knn=True):
        """運行完整診斷流程"""
        print("=" * 60)
        print("RUNNING FULL FEATURE QUALITY DIAGNOSTICS")
        print("=" * 60)
        print()
        
        # 1. 計算similarity matrix
        self.compute_similarity_matrix()
        
        # 2. 分析similarity分布
        self.analyze_similarity_distribution()
        
        # 3. 分析k-NN結構 (新增)
        if analyze_knn:
            self.analyze_knn_structure()
        
        # 4. 測試clustering方法
        clustering_results = self.test_clustering_methods(n_clusters_list)
        
        # 5. 測試DBSCAN
        dbscan_result = self.test_dbscan()
        
        # 6. t-SNE視覺化
        if visualize:
            # 如果樣本太多，採樣以加速
            n_samples = 1000 if len(self.features) > 1000 else None
            self.visualize_tsne(n_samples=n_samples)
        
        # 7. 總結報告
        self._generate_final_report(clustering_results, dbscan_result)
        
        print("=" * 60)
        print("DIAGNOSTICS COMPLETE")
        print("=" * 60)
        
        return {
            'clustering_results': clustering_results,
            'dbscan_result': dbscan_result
        }
    
    def _generate_final_report(self, clustering_results, dbscan_result):
        """生成最終診斷報告"""
        print("\n" + "=" * 60)
        print("FINAL DIAGNOSTIC REPORT")
        print("=" * 60)
        
        # 1. Feature Quality Summary
        print("\n1. FEATURE QUALITY:")
        mask = ~np.eye(self.sim_matrix.shape[0], dtype=bool)
        all_sims = self.sim_matrix[mask]
        
        print(f"   - Similarity mean: {all_sims.mean():.3f}")
        print(f"   - Similarity std: {all_sims.std():.3f}")
        
        if clustering_results:
            best_sil = max([r['silhouette'] for r in clustering_results])
            print(f"   - Best silhouette score: {best_sil:.3f}")
            
            if best_sil > 0.3:
                quality = "GOOD"
                recommendation = "Features are adequate"
            elif best_sil > 0.15:
                quality = "MODERATE"
                recommendation = "Consider upgrading to ArcFace"
            else:
                quality = "POOR"
                recommendation = "MUST upgrade to ArcFace"
            
            print(f"   - Quality assessment: {quality}")
            print(f"   - Recommendation: {recommendation}")
        
        # 2. Clustering Performance
        print("\n2. CLUSTERING PERFORMANCE:")
        if clustering_results:
            best_result = max(clustering_results, key=lambda x: x['silhouette'])
            print(f"   - Best method: {best_result['method']}")
            print(f"   - Best k: {best_result['n_clusters']}")
            print(f"   - Silhouette: {best_result['silhouette']:.3f}")
            
            if self.has_labels:
                print(f"   - True number of identities: {len(np.unique(self.labels))}")
                if 'ari' in best_result:
                    print(f"   - ARI: {best_result['ari']:.3f}")
                if 'nmi' in best_result:
                    print(f"   - NMI: {best_result['nmi']:.3f}")
        
        if dbscan_result:
            print(f"\n   DBSCAN Results:")
            print(f"   - Found clusters: {dbscan_result['n_clusters']}")
            print(f"   - Noise points: {dbscan_result['n_noise']} ({dbscan_result['n_noise']/len(self.labels)*100:.1f}%)")
            print(f"   - Silhouette: {dbscan_result['silhouette']:.3f}")
        
        # 3. Next Steps
        print("\n3. RECOMMENDED NEXT STEPS:")
        
        if clustering_results:
            best_sil = max([r['silhouette'] for r in clustering_results])
            
            if best_sil < 0.2:
                print("   Step 1: Switch to ArcFace for feature extraction (CRITICAL)")
                print("   Step 2: Re-run this diagnostic with ArcFace features")
                print("   Step 3: If still poor, check face detection quality")
            elif best_sil < 0.3:
                print("   Option A: Switch to ArcFace for better results (RECOMMENDED)")
                print("   Option B: Try simple clustering (AGC or DBSCAN) instead of GCN")
                print("   Option C: Fine-tune GCN with your specific data")
            else:
                print("   Step 1: Your features are good, problem is likely in GCN")
                print("   Step 2: Check GCN score distribution (positive vs negative)")
                print("   Step 3: Adjust IPS parameters (k1, k2) based on k-NN analysis")
                print("   Step 4: Implement adaptive threshold strategy")
        
        # 4. Quick Action Items
        print("\n4. QUICK ACTION CHECKLIST:")
        print("   [ ] Review the similarity distribution plot")
        print("   [ ] Check the t-SNE visualization for cluster separation")
        print("   [ ] Review k-NN analysis for appropriate k1, k2 values")
        
        if clustering_results and max([r['silhouette'] for r in clustering_results]) < 0.25:
            print("   [ ] !!! PRIORITY: Switch to ArcFace feature extractor !!!")
        
        print("\n" + "=" * 60 + "\n")


# ============================================================================
# 使用範例
# ============================================================================

def load_your_data():
    
    print("⚠ Warning: Using dummy data. Replace load_your_data() with your actual data loading logic.")
    features = np.load(r"C:\Users\VIPLAB\Desktop\Yan\gcn_clustering-master\result_s1ep1\gcn_temp\features.npy")
    labels = None  
    
    return features, labels


if __name__ == "__main__":
    print("Loading features...")
    features, labels = load_your_data()
    
    # 創建診斷器
    diagnostics = FeatureQualityDiagnostics(
        features=features,
        labels=labels,  # 如果沒有ground truth，設為None
        video_name="your_video_name"
    )
    
    # 運行完整診斷
    results = diagnostics.run_full_diagnostics(
        n_clusters_list=None,  # 自動選擇，或手動指定如 [10, 20, 30, 40, 50]
        visualize=True,  # 是否做t-SNE視覺化
        analyze_knn=True  # 是否分析k-NN結構（對理解GCN很重要）
    )
    
    print("\n✓ All diagnostics complete!")
    print("  Check the generated plots in the current directory:")
    print("  - similarity_dist_*.png")
    print("  - clustering_comparison_*.png (if multiple methods tested)")
    print("  - tsne_*.png")