import numpy as np
import pandas as pd
from loading import loading, done
from fetch_data import fetch_data, get_random_n_categories, get_subset_df
from preprocessing import preprocess
from training import using_tsne, using_pca, predict

loading("Loading data")
news_df, categories = fetch_data()
done()

option = int(input("Enter 1 to get random categories or 2 to get specific categories: "))
if option == 1:
    n = int(input("Enter the number of random categories: "))
    selected_categories = get_random_n_categories(n, news_df, categories)
else:
    print("Available categories:")
    for i, cat in enumerate(categories):
        print(f"{i}: {cat}")
    selected_categories = input("Enter the numbers of the specific categories separated by commas: ").split(',')
    selected_categories = np.array([categories[int(cat.strip())] for cat in selected_categories])

print(f"\nSelected categories:  {', '.join(selected_categories)}\n")
df = get_subset_df(news_df, categories, selected_categories)


loading("Preprocessing data")
df['text'] = df['text'].apply(preprocess)
done()
X_train, y_train = np.array(df['text']), np.array(df['target'])

test_text = """Computer graphics is a field focused on generating and manipulating visual content using computational techniques.
It involves concepts like rasterization, rendering, shading, and texture mapping to create realistic or stylized images.
Technologies such as OpenGL and DirectX enable hardware-accelerated graphics, while transformations, projections,
and matrices support 2D and 3D modeling. Advanced topics include ray tracing,
anti-aliasing, GPU programming, and real-time animation for simulations, video games, and visual effects."""

test_text_processed = preprocess(test_text)
updated_X_train = np.append(X_train, test_text_processed)
y_train_updated = np.append(y_train, 0)

loading('Model training using t-SNE')
predicted_label = using_tsne(updated_X_train, y_train_updated, selected_categories)
done()

loading('Model training using PCA')
vectorizer, pca, model, labels = using_pca(X_train, y_train, selected_categories)
done()
final_labels = predict([test_text], vectorizer, pca, model, labels)

with open(f'../results/predictions_for_{len(selected_categories)}_categories.txt', 'w') as f:
    f.write(f"Predicted category using (Vectorization, t-SNE and KMeans) for \"{test_text}\" is ({predicted_label})\n\n")
    f.write(f"Predicted category using (Vectorization, PCA and KMeans) for \"{test_text}\" is ({final_labels[0]})")

