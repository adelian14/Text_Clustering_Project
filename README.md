# 📌 Project: Text Clustering on 20 Newsgroups Dataset

This project explores text clustering using the classic 20 Newsgroups dataset.
The main objective is to group similar news articles using K-Means clustering,
after transforming the raw text data into numerical features using TF-IDF vectorization.

By applying clustering techniques to this dataset, we aim to group articles by topic without using labeled training data,
and then evaluate how well the clustering aligns with the true categories.

## 📚 Dataset
### We use the 20 Newsgroups dataset, which contains roughly 20,000 newsgroup documents evenly divided across 20 different categories. For this project:

- We load all categories (subset='all')

- We remove headers, footers, and quoted replies to clean the data and focus on the main body text

Each document in the dataset is labeled with one of 20 newsgroup topics, and this label is used only for evaluation—not during the clustering process.

## 🛠️ Tech Stack & Libraries
### This project is implemented in Python using the following libraries:

- `scikit-learn` – for clustering, vectorization, evaluation

- `nltk` – for text preprocessing (stopwords, stemming)

- `matplotlib`, `seaborn` – for visualization

- `pandas`, `numpy` – for data manipulation

- `scipy` – for evaluation utilities (e.g. Hungarian algorithm)

## 🔍 Methodology

### 1. Selecting a Subset of Categories

To simplify the clustering task and make the visualizations more interpretable, we randomly select **6 categories** from the 20 available in the dataset. This subset allows us to evaluate clustering performance in a more controlled setting.

```python
n_categories = 6
```

Once selected, we filter the original dataset to include only articles from those categories. The category labels are then remapped to a range of 0 to `n_categories - 1` for consistency.

This step reduces noise and complexity while still preserving the multi-class nature of the clustering task.

### 2. Text Preprocessing

We clean and normalize the text data before vectorization using the following steps:

- Convert all text to lowercase
- Remove:
  - Numbers
  - Punctuation and extra whitespace
  - Long words (25+ characters)
  - Emails, URLs, HTML tags, and non-ASCII characters
- Tokenize and keep only alphabetic words
- Remove English stopwords
- Apply **Porter Stemming** to reduce words to their base form

These steps are implemented in a custom `preprocess()` function and applied to the dataset.

### 3. Train-Test Split

After preprocessing, the dataset is split into **training (80%)** and **testing (20%)** sets. The test set is used for evaluating the generalization of the clustering pipeline.

### 4. Clustering Evaluation

Since clustering is unsupervised, evaluating performance can be tricky. We use a few methods to assess the quality of our clusters:

#### ✅ **Purity Score**
Purity is a simple metric to evaluate how pure each cluster is — i.e., how many of the items in each cluster share the same ground-truth label. It’s calculated using the **contingency matrix** of true vs predicted labels.

> A **contingency matrix** (also known as a confusion matrix for clustering) is a table that shows the frequency distribution of actual class labels vs. predicted clusters. Each cell `[i][j]` represents the number of samples from true class `i` assigned to predicted cluster `j`.

#### 🔁 **Cluster-to-Label Assignment (Hungarian Algorithm)**
Clustering algorithms like K-Means assign arbitrary cluster numbers, which don’t directly correspond to actual class labels. To evaluate properly, we align the predicted clusters to the true labels using the **Hungarian algorithm**.

> The **Hungarian algorithm** solves the assignment problem by finding the most optimal one-to-one mapping between clusters and labels — maximizing overlap in the contingency matrix while minimizing misalignment.

#### 📊 **Label Distribution**
We also log the number of documents in each predicted and actual category to inspect if the clustering is reasonably balanced and aligned.

### 📈 5. Visualizations

To interpret the clustering results more intuitively, we include multiple visualizations:

#### 📊 Category Distribution

We use grouped bar charts to compare the number of samples in each **true category** vs their corresponding **predicted cluster**, allowing us to visually inspect how well the clusters align with ground truth.

```python
# Example function
visualize_counts(y_train, y_pred, title="True vs Predicted Category Counts")
```

#### 🧩 PCA Cluster Plot

We reduce high-dimensional TF-IDF vectors into 2 components using **PCA (Principal Component Analysis)** for a quick visual snapshot of how well-separated the clusters are. Cluster centers are also plotted as red Xs.

```python
pca_visualization(X_pca, y_pred, cluster_centers, target_names)
```

#### 🌀 t-SNE Cluster Plot

We also apply **t-SNE (t-distributed Stochastic Neighbor Embedding)** — a non-linear dimensionality reduction method — to explore the cluster structure in a more organic, layout-preserving way.

```python
tsne_visualization(X_tsne, y_pred, cluster_centers, target_names)
```

Each point in the plots represents a news article, colored by its predicted cluster. These plots help us visually assess whether the model has successfully separated different topics.

### 🧠 6. Feature Extraction & Clustering

#### 🔡 TF-IDF Vectorization

We convert text into numeric vectors using **TF-IDF** and tune the following parameters:

- **N-gram range**: from `(1,1)` up to `(1,3)`
- **Max Features**: `1024`, `2048`, `4096`

This enables exploration of how textual richness and vocabulary size affect clustering quality.

---

### 🔄 7. Dimensionality Reduction

To make clustering more efficient and visualizable, we reduce the high-dimensional TF-IDF vectors into 2 dimensions using:

#### 🌀 t-SNE

- Captures non-linear relationships
- Good for **visual interpretation** but not ideal for generalization

#### 📉 PCA (Principal Component Analysis)

- Linear dimensionality reduction technique
- We retain **90% of the original variance**
- Enables clustering while preserving meaningful structure in the data

For both methods, we apply **K-Means** after reduction and compare clustering results.

---

### 🔧 8. Hyperparameter Tuning & Evaluation

Each vectorization + reduction combo is evaluated using:

- **Silhouette Score** – Measures how similar a point is to its own cluster vs. other clusters
- **Purity Score** – Measures how well the predicted clusters align with actual categories
- **Visualizations** – Show predicted vs true distribution, and cluster scatter plots with centers

```bash
Silhouette score = 0.4307, Purity = 0.6517
```

We then visualize the best configuration using PCA or t-SNE for final inspection.

### 🧪 9. Test Set Evaluation

To evaluate the generalization ability of our clustering pipeline, we apply the trained vectorizer, PCA transformer, and K-Means model to the **test set**.

We use the same evaluation metrics:

- **Silhouette Score**
- **Purity**
- **Visual comparison** between actual and predicted categories

```bash
Silhouette score = 0.0048, Purity = 0.5879
```

Visualizations help us understand how consistent the model’s performance is on unseen data.



