import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(r"C:\Users\tahseen\OneDrive\Desktop\Prodigy InfoTech Internship\PRODIGY_ML_02-main\PRODIGY_ML_02-main\Clustered Customers.csv")

print(data.head())

sns.set(style="whitegrid")

# 1. CustomerID Distribution (as it's unique, just showing range)
plt.figure(figsize=(8, 6))
plt.plot(data['CustomerID'], marker='o', linestyle='--', color='b')
plt.title('CustomerID Distribution')
plt.xlabel('Index')
plt.ylabel('CustomerID')
plt.show()

# 2. Gender Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', data=data, hue='Gender', palette='coolwarm', legend=False)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# 3. Age Distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['Age'], bins=10, kde=True, color='green')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# 4. Annual Income Distribution (k$)
plt.figure(figsize=(8, 6))
sns.histplot(data['Annual Income (k$)'], bins=10, kde=True, color='purple')
plt.title('Annual Income Distribution (k$)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Count')
plt.show()

# 5. Spending Score Distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['Spending Score (1-100)'], bins=10, kde=True, color='orange')
plt.title('Spending Score Distribution')
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Count')
plt.show()

X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method to Determine Optimal Number of Clusters')
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

data['Cluster'] = clusters

plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', marker='o')
plt.title('Customer Segments')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.colorbar()
plt.show()

data.to_csv('Clustered Customers.csv', index=False)

print(data.head())