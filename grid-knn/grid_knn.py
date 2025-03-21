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
    # attr types
    k: int  # nearest neighbours to consider
    n: int  # dimensionality of dataset
    box_size: float  # size of each box in the partition
    grid_hashmap: defaultdict[tuple[float], Box]  # stores each of the boxes

    def __init__(self, k: int, box_size: float) -> None:
        """
        Initialize the GridKNN model.

        Parameters:
        - k: number of nearest neighbors to find.
        - box_size: size of each box in the grid.
        """
        self.k = k
        self.n = None  # no dataset yet
        self.box_size = box_size
        self.grid_hashmap = defaultdict(Box)

    def fit(self, data: np.ndarray) -> None:
        """
        Fit the model to the dataset.

        Parameters:
        - data: ndarray of shape (n_samples, n_features).
        """
        self.data = data
        self.n = self.data.ndim
        self._build_grid()
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the model to the dataset.

        Parameters:
        - data: DataFrame of shape (n_samples, n_features).
        """
        self.data = data.values
        self.data.ndim
        self._build_grid()

    def _build_grid(self) -> None:
        """
        Partition the data into a grid and store points in the hashmap.
        """
        # get the outermost point (furthest from origin, by euclidean distance)
        distances = np.linalg.norm(self.data, axis=1)
        farthest_point = self.data[np.argmax(distances)]

        #  number of partitions per dimension
        num_boxes = np.ceil(farthest_point / self.box_size).astype(int)
        grid_range = [np.arange(0, (num_boxes[i] + 1) * self.box_size, self.box_size) for i in range(self.n)]

        # create boxes partitioning the entire space up to the furthest point, inclusive
        for outer_corner in np.array(np.meshgrid(*grid_range)).T.reshape(-1, self.n):
            outer_corner_key = tuple(outer_corner)
            self.grid_hashmap[outer_corner_key] = Box(outer_corner, self.box_size)

    def _compute_outer_corner(self, point: np.ndarray) -> np.ndarray:
        """
        Get the outer corner of the box this point is in.
        """
        return np.ceil(point / self.box_size) * self.box_size

    def _get_box(self, point: np.ndarray) -> Box:
        """
        Get the box containing the given point.
        """
        # get the outer corner of this point's box
        outer_corner = self._compute_outer_corner(point)
        outer_corner_key = tuple(outer_corner)
        return self.grid_hashmap[outer_corner_key]

    def _get_surrounding_boxes(self, boxes: deque[Box]) -> deque[Box]:
        """
        Get a queue representing the boxes surrounding the given ones.
        """
        raise NotImplementedError
    
    def _get_k_nearest_neighbours(self, point: np.ndarray) -> np.ndarray:
        """
        Get the k nearest neighbours of this point
        """
        # get the box that the point is within
        box_containing_point = self._get_box(point)
        # if it has atleast k points (besides the query point), classify based on them
        if box_containing_point.num_samples > self.k:
            # TODO: classify point with the samples in it's own box
            pass
        # otherwise, check surrounding boxes until the k nearest neighbours are identified
        else:
            # TODO: recurse into the boxes surrounding our current ones
            boxes = deque([box_containing_point])  # initialize as deque for BFS
            pass
        raise NotImplementedError

    def _mode_class(self, nearest_neighbours: np.ndarray) -> float:
        """
        Get the mode classification of the k nearest neighbours.
        """
        # TODO: get the mode classification of the nearest neighbours
        raise NotImplementedError

    def predict(self, point: np.ndarray) -> float:
        """
        Classify a point as the mode of its k nearest neighbours.

        Parameters:
        - point (ndarray): the point to be classified

        Returns:
        The classification of the point as a float
        """
        nearest_neighbours = self._get_k_nearest_neighbours(point)
        return self._mode_class(nearest_neighbours)

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