from scipy.spatial.distance import cdist

class NearestNeighborDistanceMetric:
    def __init__(self, metric, matching_threshold, budget=None):
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}
        self.metric = metric

    def distance(self, features1, features2):
        if self.metric == "cosine":
            return cdist(features1, features2, metric="cosine")
        else:
            raise ValueError("Invalid metric")
