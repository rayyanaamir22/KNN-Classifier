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
    outer_corner: np.ndarray
    box_size: float

    def __init__(self, data: list[np.ndarray], outer_corner: np.ndarray, box_size: float) -> None:
        """
        Box (hypercube) constructor

        Parameters:
        - data (list[ndarray]): data points inside this box
        - outer_corner (ndarray): the corner of this box that is furthest (by euclidean distance) from the origin
        """
        self.data = data
        self.num_samples = len(data)
        self.outer_corner = outer_corner
        self.box_size = box_size
    
    def __repr__(self) -> str:
        """
        String representation of the box
        """
        return f"Box(outer_corner={self.outer_corner}, box_size={self.box_size}, num_samples={self.num_samples})"