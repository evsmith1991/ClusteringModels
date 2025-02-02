import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

features = ['age', 'income_numeric', 'platform_aum', 'titan_join_age', 
            'logins_in_past_year', 'recurring_yn', 'deposit_withdrawal_ratio', 
            'risk_score', 'net_transfers']

# normalise data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# diff eps vals and min_samples vals
for eps in [1.0, 1.25, 1.5, 1.75, 2]:
    dbscan = DBSCAN(eps=eps, min_samples=20) 
    df['dbscan_cluster'] = dbscan.fit_predict(scaled_features)

    n_clusters = len(set(df['dbscan_cluster'])) - (1 if -1 in df['dbscan_cluster'] else 0)
    
    if n_clusters > 1:
        sil_score_dbscan = silhouette_score(scaled_features, df['dbscan_cluster'])
        print(f'EPS: {eps} | Number of Clusters: {n_clusters} | Silhouette Score: {sil_score_dbscan}')
    else:
        print(f'EPS: {eps} | Less than 2 clusters found, skipping silhouette score.')
