class Track:
    def __init__(self, detection, track_id, kf):
        self.track_id = track_id
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.features = [detection.feature]
        self.kf = kf
        self.mean, self.covariance = self.kf.initiate(self.bbox)
        self.time_since_update = 0
        self.hit_streak = 0
        self.age = 0

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.bbox)
        self.features.append(detection.feature)
        self.time_since_update = 0
        self.hit_streak += 1

    def mark_missed(self):
        self.time_since_update += 1

    def is_deleted(self):
        return self.time_since_update > 10

    def is_confirmed(self):
        return self.hit_streak >= 1

    def to_tlwh(self):
        x, y, a, h = self.mean[:4]
        w = a * h
        return [x - w/2, y - h/2, w, h]
