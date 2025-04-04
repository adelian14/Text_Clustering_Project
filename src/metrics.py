import numpy as np
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment

def purity_score(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)


def assign_clusters(y_true, y_pred, selected_categories):
    cm = contingency_matrix(y_true, y_pred)
    n_labels, n_clusters = cm.shape
    M = cm.T
    if n_clusters != n_labels:
        raise ValueError(
            f"Number of clusters ({n_clusters}) != number of labels ({n_labels}).\n"
            "A one-to-one assignment is only well-defined if they match.\n"
            "If they differ, you can still do a max bipartite matching but you'll have leftover nodes."
        )
    cost = -M
    row_ind, col_ind = linear_sum_assignment(cost)
    cluster_to_label = {}
    for cluster_id, label_idx in zip(row_ind, col_ind):
        cluster_to_label[cluster_id] = selected_categories[label_idx]

    return cluster_to_label

def count_examples(y_true, y_pred, true_labels):
    pred_labels = assign_clusters(y_true, y_pred, true_labels)
    true_counts = {}
    pred_counts = {}
    for y in y_true:
        if true_labels[y] in true_counts:
          true_counts[str(true_labels[y])] += 1
        else:
          true_counts[str(true_labels[y])] = 1
    for y in y_pred:
        if pred_labels[y] in pred_counts:
          pred_counts[str(pred_labels[y])] += 1
        else:
          pred_counts[str(pred_labels[y])] = 1
    return true_counts, pred_counts