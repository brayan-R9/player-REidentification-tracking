# kalman_filter.py

import numpy as np

class KalmanFilter:
    def __init__(self):
        self._ndim = 4
        self._dt = 1.

        self._motion_mat = np.eye(8)
        for i in range(4):
            self._motion_mat[i, i + 4] = self._dt

        self._update_mat = np.eye(4, 8)

        self.mean = None
        self.covariance = None

    def initiate(self, measurement):
        # measurement: [x1, y1, x2, y2]
        cx, cy, s, r = convert_bbox_to_z(measurement)
        mean_pos = np.array([cx, cy, s, r])
        mean_vel = np.zeros(4)
        self.mean = np.r_[mean_pos, mean_vel]
        self.covariance = np.eye(8)

    def predict(self):
        self.mean = np.dot(self._motion_mat, self.mean)
        self.covariance = np.dot(
            np.dot(self._motion_mat, self.covariance),
            self._motion_mat.T
        ) + np.eye(8) * 0.01

    def update(self, measurement):
        cx, cy, s, r = convert_bbox_to_z(measurement)
        z = np.array([cx, cy, s, r])

        projected_mean = np.dot(self._update_mat, self.mean)
        projected_cov = np.dot(
            np.dot(self._update_mat, self.covariance),
            self._update_mat.T
        ) + np.eye(4) * 0.1

        K = np.dot(
            np.dot(self.covariance, self._update_mat.T),
            np.linalg.inv(projected_cov)
        )

        innovation = z - projected_mean
        self.mean += np.dot(K, innovation)
        self.covariance = np.dot(
            np.eye(8) - np.dot(K, self._update_mat),
            self.covariance
        )

    def get_state(self):
        return self.mean[:4]

def convert_bbox_to_z(bbox):
    """
    Convert [x1,y1,x2,y2] to [cx,cy,s,r]
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.
    cy = y1 + h / 2.
    s = w * h
    r = w / float(h)
    return cx, cy, s, r

def convert_x_to_bbox(x):
    """
    Convert [cx, cy, s, r] to [x1, y1, x2, y2]
    """
    cx, cy, s, r = x
    w = np.sqrt(s * r)
    h = s / w
    x1 = cx - w / 2.
    y1 = cy - h / 2.
    x2 = cx + w / 2.
    y2 = cy + h / 2.
    return np.array([x1, y1, x2, y2])
