from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from metrics import purity_score, assign_clusters
from visualize import tsne_visualization, pca_visualization, visualize_counts
from loading import loading, done
from preprocessing import prepare_data

def using_tsne(X_train, y_train, selected_categories):
    loading("Vectorizing")
    vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=4096)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    done(0)
    
    loading('Reducing dimensionality using t-SNE')
    tsne = TSNE(perplexity = 50, random_state=42)
    X_train_tsne = tsne.fit_transform(X_train_vectorized.toarray())
    done(0)
    
    loading('Training using KMeans')
    model = KMeans(n_clusters=len(selected_categories), random_state=42)
    model.fit(X_train_tsne)
    done(0)
    y_pred = model.predict(X_train_tsne)
    silhouette_avg = silhouette_score(X_train_tsne, y_pred)
    purity = purity_score(y_train, y_pred)
    labels = assign_clusters(y_train, y_pred, selected_categories)
    title = f"Silhouette score = {silhouette_avg:.4f}, Purity = {purity:.4f}"
    tsne_visualization(X_train_tsne, y_pred, model.cluster_centers_, labels, title)
    visualize_counts(y_train, y_pred, "(t-SNE) "+title, selected_categories)
    return labels[y_pred[-1]]
    
def using_pca(X_train, y_train, selected_categories):
    
    loading("Vectorizing")
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=2048)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    done(0)
    
    loading('Reducing dimensionality using PCA')
    pca = PCA(n_components=.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_vectorized.toarray())
    done(0)
    
    loading('Training using KMeans')
    model = KMeans(n_clusters=len(selected_categories), random_state=42)
    model.fit(X_train_pca)
    done(0)
    
    y_pred = model.predict(X_train_pca)
    silhouette_avg = silhouette_score(X_train_pca, y_pred)
    purity = purity_score(y_train, y_pred)
    labels = assign_clusters(y_train, y_pred,selected_categories)
    title = f"Silhouette score = {silhouette_avg:.4f}, Purity = {purity:.4f}"
    pca_visualization(X_train_pca, y_pred, model.cluster_centers_, labels, title)
    visualize_counts(y_train, y_pred, "(PCA) "+title, selected_categories)
    return vectorizer, pca, model, labels
    
def predict(X, vectorizer, pca, model, labels):
    X_test = prepare_data(X)
    X_test_vectorized = vectorizer.transform(X_test)
    X_test_pca = pca.transform(X_test_vectorized.toarray())
    y_pred = model.predict(X_test_pca)
    final_labels = []
    for i in y_pred:
        final_labels.append(labels[i])
    return final_labels
    