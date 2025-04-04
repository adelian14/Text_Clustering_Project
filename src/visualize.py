import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from metrics import count_examples
import time
def visualize_counts(y_train, y_pred, title, selected_categories):
    cnt_true, cnt_pred = count_examples(y_train, y_pred, selected_categories)
    y_true_plot = []
    y_pred_plot = []
    labels = []
    x_spaces = np.arange(len(selected_categories))
    width = .35
    for cat in selected_categories:
        y_true_plot.append(cnt_true[str(cat)])
        y_pred_plot.append(cnt_pred[str(cat)])
        labels.append(cat)
    plt.figure(figsize=(10, 6))
    plt.bar(x_spaces - width/2, y_true_plot, width, label='True')
    plt.bar(x_spaces + width/2, y_pred_plot, width, label='Predicted')
    plt.xlabel('Categories')
    plt.ylabel('Count')
    plt.xticks(x_spaces,labels, rotation=70)
    plt.title(title)
    plt.legend()
    timestamp = int(time.time()*1000)
    plt.savefig(f"../results/label_distribution_{timestamp}.png")
    
def pca_visualization(X_pca, y_pred, cluster_centers, target_names, title = ''):

    unique_labels = np.unique(y_pred)
    palette = sns.color_palette("hls", len(unique_labels))

    plt.figure(figsize=(10, 10))
    for label, color in zip(unique_labels, palette):
        mask = (y_pred == label)
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    label=str(target_names[label]), color=color, alpha=0.7)

    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='darkred', marker='X', s=200, label="Cluster Centers")

    plt.legend(title="Clusters")
    plt.title(f"PCA Visualization of Clusters\n{title}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    timestamp = int(time.time()*1000)
    plt.savefig(f"../results/PCA_visualization_{timestamp}.png")
    
def tsne_visualization(X_tsne, y_pred, cluster_centers, target_names, title = ''):

    unique_labels = np.unique(y_pred)
    palette = sns.color_palette("hls", len(unique_labels))

    plt.figure(figsize=(10, 10))
    for label, color in zip(unique_labels, palette):
        mask = (y_pred == label)
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    label=str(target_names[label]), color=color, alpha=0.7)

    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='darkred', marker='X', s=200, label="Cluster Centers")

    plt.legend(title="Clusters")
    plt.title(f"t-SNE Visualization of Clusters\n{title}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    timestamp = int(time.time()*1000)
    plt.savefig(f"../results/TSNE_visualization_{timestamp}.png")