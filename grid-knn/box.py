"""
Box (hypercube) class to be used in the GridKNN's partitioning algorithm.
"""

# frameworks
import numpy as np

class Box:
    """
    Box (hypercube) class
    """
    # attr types
    data: list[np.ndarray]
    num_samples: int  # number of data points within this box
    inner_corner: np.ndarray
    box_size: float

    def __init__(self, data: list[np.ndarray], inner_corner: np.ndarray, box_size: float) -> None:
        """
        Box (hypercube) constructor

        Parameters:
        - data (list[ndarray]): data points inside this box
        - inner_corner (ndarray): the corner of this box that is closest (by euclidean distance) to the origin
        """
        self.data = data
        self.num_samples = len(data)
        self.inner_corner = inner_corner
        self.box_size = box_size
    
    def __repr__(self) -> str:
        """
        String representation of the box
        """
        return f"Box(inner_corner={self.inner_corner}, box_size={self.box_size}, num_samples={self.num_samples})"