# linear_assignment.py

import numpy as np
from scipy.optimize import linear_sum_assignment
from .kalman_filter import convert_x_to_bbox

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
              (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o

def linear_assignment(tracks, detections, iou_threshold):
    if len(tracks) == 0 or len(detections) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))

    iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for t, track in enumerate(tracks):
        track_bbox = convert_x_to_bbox(track.kf.get_state())
        for d, det in enumerate(detections):
            iou_matrix[t, d] = iou(track_bbox, det)

    matched_indices = np.where(iou_matrix > iou_threshold)
    matches = []
    matched_tracks, matched_detections = set(), set()

    for t, d in zip(*matched_indices):
        if t not in matched_tracks and d not in matched_detections:
            matches.append((t, d))
            matched_tracks.add(t)
            matched_detections.add(d)

    unmatched_tracks = list(set(range(len(tracks))) - matched_tracks)
    unmatched_detections = list(set(range(len(detections))) - matched_detections)

    return matches, unmatched_tracks, unmatched_detections
