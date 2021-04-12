import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA


with open(r'C:\code\kmeans\xploreelasticutv-10000.json ', 'r') as f:
    data = json.load(f)

min_timestamps = []
hours_timestamps = []
for line in data:
    hour_timestamp = ""
    min_timestamp = ""

    for i in range(2):
        hour_timestamp += str(line)[150 + i]
        min_timestamp += str(line)[153 + i]

    if hour_timestamp[0] == '0':
        hours_timestamps.append(int(hour_timestamp[1:]))
    else:
        hours_timestamps.append(int(hour_timestamp))
    min_timestamps.append(int(min_timestamp))

hp = np.array(hours_timestamps)
mp = np.array(min_timestamps)

hours_min_timestamps = np.vstack((hp, mp)).T

# pca = PCA(n_components=2).fit_transform(hours_min_timestamps)
kmeans = KMeans(init="k-means++", n_clusters=10, n_init=4)
kmeans.fit(hours_min_timestamps)


h = 1     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = hp.min() - 1, hp.max() + 1
y_min, y_max = mp.min() - 1, mp.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation="nearest",
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           aspect="auto", origin="lower")

plt.plot(hours_timestamps, min_timestamps, 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=100, linewidths=3,
            color="b", zorder=10)
plt.title("K-means clustering on timestamps extracted from" + "\n" + "xploreelasticutv-10000 JSON log file")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
