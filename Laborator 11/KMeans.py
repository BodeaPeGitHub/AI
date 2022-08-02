import pandas as pd
import numpy as np


class KMeans:

    def __init__(self, k):
        self.__k = k

    @staticmethod
    def calculate_cost(X, centroids, cluster):
        sum = 0
        for i, val in enumerate(X):
            sum += np.sqrt(
                (centroids[int(cluster[i]), 0] - val[0]) ** 2 + (centroids[int(cluster[i]), 1] - val[1]) ** 2)
        return sum

    def fit(self, inputs: np.array):
        diff = True
        cluster = np.zeros(inputs.shape[0])
        centroids = inputs.sample(n=self.__k).values
        while diff:
            for i, row in enumerate(inputs):
                minim_distance = float('inf')
                for idx, centroid in enumerate(centroids):
                    distance = np.sqrt(np.sum((centroid - row) ** 2, axis=1))
                    minim_distance, cluster[i] = (distance, idx) if minim_distance > distance else (minim_distance, cluster[i])
            new_centroids = pd.DataFrame(inputs).groupby(by=cluster).mean().values
            diff = np.count_nonzero(centroids - new_centroids) == 0
            centroids = new_centroids
        return centroids, cluster
