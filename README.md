# KNN-Classifier

Various implementations of the k-Nearest Neighbour model, supporting datasets in the form of `np.array` and `pd.DataFrame` objects.

The base model is inspired by [this](https://www.youtube.com/watch?v=AoeEHqVSNOw&t=44s) video.

## base-knn
Vanilla implementation of the kNN algorithm, with `fit` and `predict` methods.

## k-d-tree-knn
kNN model augmented with a k-d tree divide-and-conquer approach to reduce the search space exponentially. Efficient for tasks with low to medium dimensionality.

## grid-knn
Novel kNN approach that partitions the search space $\mathbb{R}^n$ into (roughly) equal sized [hypercubes](https://en.wikipedia.org/wiki/Hypercube), and hashes into the box containing the inputted data point, upon which the base kNN algorithm is performed. If k neighbours are not present, it recursively expands into the surrounding boxes (enter. [Curse of Dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)). A depth-limit can be enforced on the recursive expansion, effectively making `grid-knn` an approximation algorithm.

Very efficient for tasks with low dimensionality and dense data, but suffers a lot from curse of dimensionality and sparser data.