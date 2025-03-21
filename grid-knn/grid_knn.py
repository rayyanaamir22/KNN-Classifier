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
        self.grid_hashmap = defaultdict(list)

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
        # TODO: get the outermost point (furthest from origin, by euclidean distance)
        pass

        # TODO: create boxes partitioning the space up to the furthest point, inclusive
        pass

        # TODO: insert each point into their corresponding box
        pass
        
        raise NotImplementedError

    def _get_box(self, point: np.ndarray) -> Box:
        """
        Get the box containing the given point.
        """
        # TODO: find the interval the point is within for each dimension
        box_intervals_containing_point = None
        box_outer_corner = None
        # get the index for the grid_hashmap (coords of outermost corner)
        outer_corner_key = tuple(box_outer_corner)
        # query the grid map to retrieve the box with this outer corner
        return self.grid_hashmap[outer_corner_key]

    def _get_surrounding_boxes(self, boxes: deque[Box]) -> deque[Box]:
        """
        Get a queue representing the boxes surrounding this one.
        """
        # TODO: get the indices of surrounding boxes' outer corners
        pass
        # TODO: query the grid map for the boxes and return in a deque
        # (the queue is convenient for BFS expansion in _get_k_nearest_neighbours)
        pass
        raise NotImplementedError
    
    def _get_k_nearest_neighbours(self, point: np.ndarray) -> list[np.ndarray]:
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

    def _mode_class(self, nearest_neighbours: list[np.ndarray]) -> float:
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