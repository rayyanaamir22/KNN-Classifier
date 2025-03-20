from typing import Union
from scipy.spatial.distance import euclidean as euc


class KNN_Classifier:
    """
    Custom K-Nearest Neighbour Classifier Model.
    """
    def __init__(self) -> None:
        """
        Constructor method.
        """
        self.fitted = False

    def closest(self, row) -> None:
        """
        Find the closest neighbours.
        """
        shortestDistance = euc(row, self.X_train[0])
        shortestIndex = 0
        # Find shortest distance
        for i in range(1, len(self.X_train)):
            distance = euc(row, self.X_train[i])
            if (distance < shortestDistance):
                shortestDistance = distance
                shortestIndex = i
        return self.y_train[shortestIndex]


    def fit(self, X_train, y_train) -> None:
        """
        Fit the model to a training set.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.fitted = True

    def predict(self, X_test) -> Union[list[Union[float, int, str]], None]:
        """
        Return predictions for y for a given set of X. If model isn't fitted, return None.
        """
        if self.fitted:
            predictions = []
            for row in X_test:
                label = self.closest(row)
                predictions.append(label)
            return predictions
        else:
            print("Model needs to be fitted first!")