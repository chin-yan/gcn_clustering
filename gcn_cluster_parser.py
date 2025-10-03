#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gcn_cluster_parser.py

Parse GCN clustering output (edges and scores) into cluster format
compatible with your existing project structure.
"""

import numpy as np
import networkx as nx
import sys
from collections import defaultdict


def parse_gcn_output_to_clusters(edges_path, scores_path, image_paths, threshold=0.5):
    """
    Convert GCN output (edges and scores) into clusters format
    
    Args:
        edges_path: Path to edges.npy (Nx2 array of node pairs)
        scores_path: Path to scores.npy (N array of link probabilities)
        image_paths: List of image paths corresponding to node indices
        threshold: Minimum score to consider an edge as valid link
        
    Returns:
        clusters: List of lists, where each inner list contains image paths in a cluster
    """
    print(f"Parsing GCN output into clusters...")
    print(f"  Threshold: {threshold}")
    
    # Load GCN output
    edges = np.load(edges_path)
    scores = np.load(scores_path)
    
    print(f"  Total edges: {len(edges)}")
    print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    
    # Filter edges by threshold
    valid_mask = scores > threshold
    valid_edges = edges[valid_mask]
    valid_scores = scores[valid_mask]
    
    print(f"  Valid edges (score > {threshold}): {len(valid_edges)}")
    
    # Build graph from valid edges
    G = nx.Graph()
    
    # Add all nodes
    for i in range(len(image_paths)):
        G.add_node(i)
    
    # Add valid edges
    for (i, j), score in zip(valid_edges, valid_scores):
        G.add_edge(int(i), int(j), weight=float(score))
    
    # Find connected components (these are our clusters)
    connected_components = list(nx.connected_components(G))
    
    print(f"  Found {len(connected_components)} clusters")
    
    # Convert node indices to image paths
    clusters = []
    for component in connected_components:
        cluster = [image_paths[node_idx] for node_idx in component]
        clusters.append(cluster)
    
    # Sort clusters by size (largest first)
    clusters.sort(key=len, reverse=True)
    
    # Print cluster size distribution
    cluster_sizes = [len(c) for c in clusters]
    print(f"  Cluster size range: [{min(cluster_sizes)}, {max(cluster_sizes)}]")
    print(f"  Average cluster size: {np.mean(cluster_sizes):.1f}")

    import matplotlib.pyplot as plt
    plt.hist(scores, bins=50)
    plt.savefig('score_distribution.png')
    print(f"Score statistics:")
    print(f"  Mean: {scores.mean():.3f}")
    print(f"  Std: {scores.std():.3f}")
    print(f"  Median: {np.median(scores):.3f}")
    print(f"  25th percentile: {np.percentile(scores, 25):.3f}")
    print(f"  75th percentile: {np.percentile(scores, 75):.3f}")
    
    return clusters


def run_gcn_clustering_and_parse(features_path, knn_graph_path, image_paths, 
                                 gcn_checkpoint, output_dir, threshold=0.5):
    """
    Complete pipeline: run GCN and parse results into clusters
    
    Args:
        features_path: Path to features.npy
        knn_graph_path: Path to knn_graph.npy
        image_paths: List of image paths
        gcn_checkpoint: Path to GCN checkpoint
        output_dir: Directory where GCN will save edges.npy and scores.npy
        threshold: Score threshold for edge validity
        
    Returns:
        clusters: List of lists containing image paths
    """
    import subprocess
    import os
    
    # Run GCN clustering
    print("Running GCN clustering...")

    gcn_test_script = os.path.abspath(r"C:\Users\VIPLAB\Desktop\Yan\gcn_clustering-master\test.py")
    gcn_checkpoint = os.path.abspath(r"C:\Users\VIPLAB\Desktop\Yan\gcn_clustering-master\logs\logs\best.ckpt")
    output_dir = os.path.abspath(output_dir)
    
    if not os.path.exists(gcn_test_script):
        print(f"Error: test.py not found at {gcn_test_script}")
        return None
    
    if not os.path.exists(gcn_checkpoint):
        print(f"Error: checkpoint not found at {gcn_checkpoint}")
        return None
    
    python_executable = sys.executable

    gcn_cmd = [
        python_executable, gcn_test_script,
        '--val_feat_path', features_path,
        '--val_knn_graph_path', knn_graph_path,
        '--checkpoint', gcn_checkpoint,
        '--k-at-hop', '80', '5',
        '--active_connection', '5'
    ]
    
    # Change to output directory so GCN saves edges.npy and scores.npy there
    original_dir = os.getcwd()
    os.chdir(output_dir)
    
    try:
        result = subprocess.run(gcn_cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print("GCN clustering failed:")
            print(result.stderr)
            return None
        
        print("GCN clustering completed")
        
        # Check if output files exist
        edges_path = 'edges.npy'
        scores_path = 'scores.npy'
        
        if not os.path.exists(edges_path) or not os.path.exists(scores_path):
            print("Error: GCN output files not found")
            return None
        
        # Parse output to clusters
        clusters = parse_gcn_output_to_clusters(
            edges_path=edges_path,
            scores_path=scores_path,
            image_paths=image_paths,
            threshold=threshold
        )
        
        return clusters
        
    finally:
        os.chdir(original_dir)


def adaptive_threshold_clustering(edges_path, scores_path, image_paths, 
                                  min_threshold=0.3, max_threshold=0.8, 
                                  target_cluster_count=None):
    """
    Try multiple thresholds to find the best clustering
    
    Args:
        edges_path: Path to edges.npy
        scores_path: Path to scores.npy
        image_paths: List of image paths
        min_threshold: Minimum threshold to try
        max_threshold: Maximum threshold to try
        target_cluster_count: Optional target number of clusters
        
    Returns:
        Best clusters and the threshold used
    """
    print("Trying adaptive threshold selection...")
    
    thresholds = np.linspace(min_threshold, max_threshold, 10)
    best_clusters = None
    best_threshold = None
    best_score = -float('inf')
    
    for threshold in thresholds:
        clusters = parse_gcn_output_to_clusters(
            edges_path, scores_path, image_paths, threshold
        )
        
        # Simple scoring: prefer moderate number of reasonably-sized clusters
        n_clusters = len(clusters)
        cluster_sizes = [len(c) for c in clusters]
        avg_size = np.mean(cluster_sizes)
        
        # Avoid too many tiny clusters or too few huge clusters
        if avg_size < 3 or n_clusters < 5:
            continue
        
        # Score based on cluster size distribution
        size_std = np.std(cluster_sizes)
        score = n_clusters / (1 + size_std / avg_size)
        
        print(f"  Threshold {threshold:.2f}: {n_clusters} clusters, avg size {avg_size:.1f}, score {score:.2f}")
        
        if target_cluster_count:
            # Prefer result close to target
            score = score - abs(n_clusters - target_cluster_count) * 0.5
        
        if score > best_score:
            best_score = score
            best_clusters = clusters
            best_threshold = threshold
    
    print(f"Selected threshold: {best_threshold:.2f} with {len(best_clusters)} clusters")
    
    return best_clusters, best_threshold