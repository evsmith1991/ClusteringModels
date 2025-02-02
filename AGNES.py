# Agglomerative Nesting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
% matplotlib inline

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Preprocessing
df = datasets["AgglomerativeClustering"]
datasets["AgglomerativeClustering"].describe()
# df['income_numeric'].fillna(df['income_numeric'].mode()[0], inplace=True)
df['platform_aum'].fillna(df['platform_aum'].mode()[0], inplace=True)
df.fillna(0, inplace=True)

z_scores = np.abs(stats.zscore(df[['platform_aum', 'logins_in_past_year', 'deposit_withdrawal_ratio', 'risk_score', 'net_transfers']]))

# threshold = 3
# df = df[(z_scores < threshold).all(axis=1)]

print(f"Number of rows after removing outliers: {df.shape[0]}")

# AGNES Cluster visualization
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

features = ['platform_aum',
            'logins_in_past_year', 'recurring_yn', 'deposit_withdrawal_ratio', 
            'risk_score', 'net_transfers']

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

pca = PCA(n_components=3)
pca_features = pca.fit_transform(scaled_features)

agg_clustering = AgglomerativeClustering(n_clusters=3) # update number to get k number of clusters
df['agg_cluster'] = agg_clustering.fit_predict(pca_features)

sil_score_agg = silhouette_score(pca_features, df['agg_cluster'])
print('Silhouette Score for Agglomerative Clustering with PCA (No Outliers):', sil_score_agg)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance explained:", np.sum(pca.explained_variance_ratio_))

scat = ax.scatter(pca_features[:, 0], pca_features[:, 1], pca_features[:, 2], 
                  c=df['agg_cluster'], cmap='coolwarm', marker='o')

ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
ax.set_title('Agglomerative Clusters Visualized with PCA (3D)')

plt.colorbar(scat, label='Cluster')
plt.show()

for cluster in sorted(df['agg_cluster'].unique()):
    cluster_data = df[df['agg_cluster'] == cluster]
    print(f'Cluster {cluster}:')
    print('mean:', cluster_data[features].mean())
    print('median:', cluster_data[features].median())
    print('\n')

plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=df['agg_cluster'], palette='coolwarm', s=50)
plt.title('Agglomerative Clustering Results (PCA Reduced, No Outliers)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()


# Metric Distributions by Cluster
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='agg_cluster', y=feature, data=df, palette='coolwarm')
    plt.title(f'Distribution of {feature} by Cluster')
    plt.show()

# Num users per cluster
cluster_sizes = df['agg_cluster'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values, palette='coolwarm')
plt.title('Number of Users in Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Number of Users')
plt.show()

# Metric Distribution per cluster
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='agg_cluster', y=feature, data=df, palette='coolwarm')
    plt.title(f'{feature} Distribution by Cluster')
    plt.show()

# Find cluster for specific user
def find_user_cluster(df, user_id):
    user_row = df[df['userid'] == user_id]
    
    if not user_row.empty:
        cluster = user_row['agg_cluster'].values[0]
        return 'User', user_id, 'belongs to Cluster', cluster
    else:
        return 'User', user_id, 'not found in the dataset.'

user_id = '5283ccec-e968-4d7c-93e4-84d3d1ae9b92' 
result = find_user_cluster(df, user_id)
print(result)

# List cluster by userid
filtered_df = df[df['is_reserve_user'] == 1]
# Set pandas to display all rows
pd.set_option('display.max_rows', None)
# Print the userids and agg_cluster columns
print(filtered_df[['userid', 'agg_cluster']])
