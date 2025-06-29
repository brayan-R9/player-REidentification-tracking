from .nn_matching import NearestNeighborDistanceMetric
from .tracker import Tracker
from .detection import Detection

class DeepSort:
    def __init__(self, max_cosine_distance=0.5, nn_budget=100):
        self.metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(self.metric)

    def update_tracks(self, detections):
        self.tracker.predict()
        self.tracker.update(detections)

    def get_tracks(self):
        return self.tracker.tracks
