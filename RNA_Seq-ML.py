import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# --------------------------------------------------------------
# LOAD DATASET

file_path = "GSE159699_summary_count.star.txt.gz"

df = pd.read_csv(
    file_path,
    sep="\t",
    compression="gzip"
)

print("Dataset shape:", df.shape)


# --------------------------------------------------------------
# PREPROCESSING
df = df.set_index("refGene")

# Samples as rows, genes as columns
X = df.T

# Create labels
y = X.index.to_series().apply(
    lambda x: "AD" if x.endswith("-AD") else "Control"
)

print("\nDiagnosis distribution:")
print(y.value_counts())


# Filter low-expression genes
X = X.loc[:, (X >= 10).sum(axis=0) >= 3]

print("\nGenes after filtering:", X.shape[1])


# CPM normalization
X = X.div(X.sum(axis=1), axis=0) * 1000000

# Log transformation
X = np.log2(X + 1)


# --------------------------------------------------------------
# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------------------
# FEATURE SELECTION
selector = SelectKBest(
    f_classif,
    k=min(500, X_train.shape[1])
)

X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

genes = X.columns[selector.get_support()]


# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------------------
# CLASSIFICATION
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = {}

for name, model in models.items():

    if name == "Random Forest":
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)
    results[name] = accuracy

    print("\n==============================")
    print(name)
    print("==============================")

    print("Accuracy:", accuracy * 100, "%")
    print(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["AD", "Control"],
        yticklabels=["AD", "Control"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(name + " - Confusion Matrix")
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)

    plt.savefig(
        "results/" +
        name.lower().replace(" ", "_") +
        "_confusion_matrix.png",
        dpi=300
    )

    plt.show()


# --------------------------------------------------------------
# MODEL COMPARISON
print("\n===== MODEL COMPARISON =====")

for name, accuracy in results.items():
    print(f"{name}: {accuracy * 100:.2f}%")


# --------------------------------------------------------------
# RANDOM FOREST FEATURE IMPORTANCE

rf = models["Random Forest"]

importance = pd.Series(
    rf.feature_importances_,
    index=genes
)

top_features = importance.nlargest(15)

print("\n===== TOP 15 GENES =====")
print(top_features)

plt.figure(figsize=(8, 5))

top_features.sort_values().plot(
    kind="barh"
)

plt.xlabel("Feature Importance")
plt.ylabel("Genes")
plt.title("Top 15 Genes - Random Forest")
plt.tight_layout()

plt.savefig(
    "results/random_forest_feature_importance.png",
    dpi=300
)

plt.show()


# --------------------------------------------------------------
# K-MEANS CLUSTERING
X_all = selector.transform(X)
X_all = scaler.transform(X_all)

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_all)

print("\n===== K-MEANS CLUSTERING =====")
print("Silhouette Score:", silhouette_score(X_all, clusters))


# --------------------------------------------------------------
# PCA VISUALIZATION
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_all)

plt.figure(figsize=(7, 5))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="tab10",
    s=50
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clustering of RNA-Seq Samples")
plt.colorbar(label="Cluster")
plt.tight_layout()

plt.savefig(
    "results/kmeans_pca.png",
    dpi=300
)

plt.show()


# --------------------------------------------------------------
# CLUSTER VS DIAGNOSIS
comparison = pd.crosstab(
    clusters,
    y,
    normalize="index"
) * 100

print("\n===== CLUSTER VS DIAGNOSIS (%) =====")
print(comparison.round(2))

print("\n===== ANALYSIS COMPLETE =====")