import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from kneed import KneeLocator

dataset = pd.read_csv("Wuzzuf_Jobs.csv")

#Q1
dataset["factorizedYearsExp"] = pd.factorize(dataset["YearsExp"])[0]
print(dataset['factorizedYearsExp'])
#**************************************************************************
#Q2
# encodeing
dataset['Title'] = pd.factorize(dataset['Title'])[0]
dataset['Company'] = pd.factorize(dataset['Company'])[0]
x = dataset.iloc[:, [0,1]].values
print(x)

wcss = []
for i in range(1, 21):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
# Showing Elbow Point
plt.plot(range(1, 21), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
# Choosing the elpow point

k1 = KneeLocator(range(1, 21), wcss, curve="convex", direction="decreasing")
print(f"best number of clusters is {k1.elbow}")

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters=k1.elbow, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(x)

# # Visualising the clusters
for i in range(k1.elbow):
    colors = ['red', 'blue', 'green', 'yellow', 'black', 'pink']
    plt.scatter(x[y_kmeans == i, 0], x[y_kmeans == i, 1], s=100, c=colors[i], label=f'Cluster{i+1}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='olive', label='Centroids')
plt.title('Clusters of Jobs')
plt.xlabel('Jobs')
plt.ylabel('Companies')
plt.legend()
plt.show()