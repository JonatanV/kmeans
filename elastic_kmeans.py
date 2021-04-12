import matplotlib.pyplot as plt
import json
import datetime
import time
from sklearn.cluster import KMeans
import numpy as np

with open(r'C:\code\kmeans\xploreelasticutv-1000.json ', 'r') as f:
    data = json.load(f)

timestamps = np.array([])
new_timestamps = []
unix_timestamp = float(0.0)
for line in data:
    timestamp = ""

    for i in range(24):
        timestamp += str(line)[139 + i]
    utc_timestamp = datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')
    unix_timestamp = time.mktime(utc_timestamp.timetuple())

    timestamps = np.append(timestamps, unix_timestamp, axis=None).reshape(-1, 1)
new_timestamps = np.append(new_timestamps, timestamps, axis=None)

timestamps2D = new_timestamps.reshape(100, 10)

kmeans = KMeans(init="k-means++", n_clusters=10, n_init=4)
kmeans.fit(np.concatenate((timestamps, timestamps), axis=1))


h = 100     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = timestamps.min() - 1, timestamps.max() + 1
y_min, y_max = timestamps.min() - 1, timestamps.max() + 1
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

plt.plot(timestamps, timestamps, 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids, centroids, marker="x", s=100, linewidths=3,
            color="b", zorder=10)
plt.title("K-means clustering on timestamps extracted from" + "\n" + "xploreelasticutv-1000 JSON log file")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
