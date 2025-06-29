# tracker.py

import numpy as np
from .kalman_filter import KalmanFilter, convert_x_to_bbox

from .linear_assignment import linear_assignment

class Track:
    def __init__(self, bbox, track_id):
        self.kf = KalmanFilter()
        self.kf.initiate(bbox)
        self.track_id = track_id
        self.time_since_update = 0

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1

    def update(self, bbox):
        self.kf.update(bbox)
        self.time_since_update = 0

    def to_tlbr(self):
        return convert_x_to_bbox(self.kf.get_state())

class Tracker:
    def __init__(self, max_age=30, iou_threshold=0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        for track in self.tracks:
            track.predict()

        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append(Track(det, self.next_id))
                self.next_id += 1
            return

        matches, unmatched_tracks, unmatched_detections = linear_assignment(
            self.tracks, detections, self.iou_threshold
        )

        for t, d in matches:
            self.tracks[t].update(detections[d])

        for t in unmatched_tracks:
            self.tracks[t].time_since_update += 1

        for d in unmatched_detections:
            self.tracks.append(Track(detections[d], self.next_id))
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
