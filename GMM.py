import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

df = datasets["KmeansDataset"]

features = ['age', 'income_numeric', 'platform_aum', 'titan_join_age', 
            'logins_in_past_year', 'recurring_yn', 'deposit_withdrawal_ratio', 
            'risk_score', 'net_transfers']

# normalise data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# gmm
# gmm = GaussianMixture(n_components=6, random_state=42)
# df['gmm_cluster'] = gmm.fit_predict(scaled_features)

# silhouette score
# sil_score_gmm = silhouette_score(scaled_features, df['gmm_cluster'])
# print(f'Silhouette Score for GMM: {sil_score_gmm}')

# pairplot visualizaitons
# sns.pairplot(df[features + ['gmm_cluster']], hue='gmm_cluster', palette='coolwarm')
# plt.show()

# analyze the cluster centroids
# gmm_means = pd.DataFrame(gmm.means_, columns=features)
# print("GMM Cluster Means:")
# print(gmm_means)

# generate sil scores to find optimal clusters number
for n_clusters in range(6, 11):
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    df[f'gmm_cluster_{n_clusters}'] = gmm.fit_predict(scaled_features)
    sil_score = silhouette_score(scaled_features, df[f'gmm_cluster_{n_clusters}'])
    print('Silhouette Score for', n_clusters, 'clusters:', sil_score)
