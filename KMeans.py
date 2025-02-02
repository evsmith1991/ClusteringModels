# STEPS:
# 1. Extract necessary cols for now. Ignore direct investment behaviors
# 2. Run PCA Dimensionality Reduction
# 3. Run Clustering Algo - KMeans
# 4. Adjust number of clusters as needed
# 5. Investigate findings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
% matplotlib inline

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#Preprocessing
df = datasets["KmeansDataset"]
datasets["KmeansDataset"].describe()
df['income_numeric'].fillna(df['income_numeric'].mode()[0], inplace=True)
df['platform_aum'].fillna(df['platform_aum'].mode()[0], inplace=True)
df.fillna(0, inplace=True)

z_scores = np.abs(stats.zscore(df[['income_numeric', 'platform_aum']]))

threshold = 3
df = df[(z_scores < threshold).all(axis=1)]

print(f"Number of rows after removing outliers: {df.shape[0]}")

#K-Means Clustering
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

features = ['age', 'income_numeric', 'platform_aum', 'titan_join_age', 
            'logins_in_past_year', 'recurring_yn', 'deposit_withdrawal_ratio', 
            'risk_score', 'net_transfers']

# normalise data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

kmeans = KMeans(n_clusters=6, random_state=42)
df['kmeans_cluster'] = kmeans.fit_predict(scaled_features)

# silhouette score
sil_score_kmeans = silhouette_score(scaled_features, df['kmeans_cluster'])
print('Silhouette Score for K-Means:', sil_score_kmeans)

# pairplot visualizaitons
sns.pairplot(df[features + ['kmeans_cluster']], hue='kmeans_cluster', palette='coolwarm')
plt.show()

# analyze the cluster centroids
kmeans_centroids = pd.DataFrame(kmeans.cluster_centers_, columns=features)
print("K-Means Cluster Centroids:")
print(kmeans_centroids)

# elbow Method
inertia = []
for n in range(1, 10):
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 10), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

sil_score_kmeans = silhouette_score(scaled_features, df['kmeans_cluster'])
print('Silhouette Score for K-Means:', sil_score_kmeans)

# centroids of the clusters
print("Cluster centroids:")
centroids = kmeans.cluster_centers_
print(pd.DataFrame(centroids, columns=features))

# summary stats for each cluster (mean)
cluster_summary = df.groupby('kmeans_cluster')[features].mean()
print("Summary statistics (mean) for each cluster:")
print(cluster_summary)

# summary stats for each cluster (median)
cluster_median = df.groupby('kmeans_cluster')[features].median()
print("Summary statistics (median) for each cluster:")
print(cluster_median)

# Plotting metrics by cluster
metrics = ['age', 'income_numeric', 'platform_aum', 'titan_join_age', 
            'logins_in_past_year', 'recurring_yn', 'deposit_withdrawal_ratio', 
            'risk_score', 'net_transfers']

for metric in metrics:
    plt.figure(figsize=(8, 6))
    sns.barplot(x=cluster_summary.index, y=cluster_summary[metric], palette='viridis')
    plt.title(f'{metric.capitalize()} by Cluster')
    plt.xlabel('kmeans_cluster')
    plt.ylabel(metric.capitalize())
    plt.show()

# Comparing Income by Cluster
fig, ax1 = plt.subplots(figsize=(12, 7))

ax1.bar(cluster_summary.index - 0.2, cluster_summary['income_numeric'], width=0.2, label='Income Numeric', color='blue', alpha=0.7)
ax1.bar(cluster_summary.index, cluster_summary['platform_aum'], width=0.2, label='Platform AUM', color='green', alpha=0.7)
ax1.set_xlabel('kmeans_cluster')
ax1.set_ylabel('Income Numeric & Platform AUM')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.bar(cluster_summary.index + 0.2, cluster_summary['age'], width=0.2, label='Age', color='red', alpha=0.7)
ax2.set_ylabel('Age', color='red')
ax2.tick_params(axis='y', labelcolor='red')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.title('Comparison of Income, Platform AUM, and Age Across Clusters')
fig.tight_layout()
plt.show()

# Distributions by cluster
for metric in metrics:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='kmeans_cluster', y=metric, data=df, palette='coolwarm')
    plt.title(f'Distribution of {metric.capitalize()} Across Clusters')
    plt.xlabel('kmeans_cluster')
    plt.ylabel(metric.capitalize())
    plt.show()
