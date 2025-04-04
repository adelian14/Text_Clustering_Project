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
