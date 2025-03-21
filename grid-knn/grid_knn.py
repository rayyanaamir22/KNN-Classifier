"""
GridKNN model.
"""

# frameworks
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from sklearn.metrics.pairwise import euclidean_distances

# utils
from .box import Box

class GridKNN:
    """
    kNN model that employs the grid-hashing approach.
    """
    def __init__(self, k: int, box_size: float) -> None:
        """
        Initialize the GridKNN model.

        Parameters:
        - k: number of nearest neighbors to find.
        - box_size: size of each box in the grid.
        """
        self.k = k
        self.box_size = box_size
        self.grid_hashmap = defaultdict(list)  # stores each of the boxes

    def fit(self, data: np.ndarray) -> None:
        """
        Fit the model to the dataset.

        Parameters:
        - data: ndarray of shape (n_samples, n_features).
        """
        self.data = data
        self.build_grid()
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the model to the dataset.

        Parameters:
        - data: DataFrame of shape (n_samples, n_features).
        """
        self.data = data.values
        self.build_grid()

    def build_grid(self) -> None:
        """
        Partition the data into a grid and store points in the hashmap.
        """
        raise NotImplementedError

    def __get_box(point: np.ndarray) -> Box:
        """
        Get the box for a given point.
        """
        raise NotImplementedError

    def __get_surrounding_boxes(self, box: Box) -> deque[Box]:
        """
        Get a queue representing the boxes surrounding this one.
        """
        raise NotImplementedError
    
    def __get_k_nearest_neighbours(self, point: np.ndarray) -> list[np.ndarray]:
        """
        Get the k nearest neighbours of this point
        """
        raise NotImplementedError

    def __classify_point(self, point: np.ndarray, nearest_neighbours: list[np.ndarray]) -> float:
        """
        Classify a point as the mode of its k nearest neighbours.
        """
        raise NotImplementedError

    def predict(self, point: np.ndarray) -> float:
        """
        Classify a point as the mode of its k nearest neighbours.

        Parameters:
        - point (ndarray): the point to be classified

        Returns:
        The classification of the point as a float
        """
        
        raise NotImplementedError

# Example usage
if __name__ == "__main__":
    # example data
    data = np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8], [8, 9]])
    query_point = [2.5, 3.5]
    k = 2
    box_size = 2

    # initialize and query the model
    model = GridKNN(data, k, box_size)
    neighbors, distances = model.query(query_point)

    print("Query Point:", query_point)
    print("Neighbors:\n", neighbors)
    print("Distances:", distances)