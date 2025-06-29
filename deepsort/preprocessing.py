import numpy as np


def non_max_suppression(boxes, max_bbox_overlap, scores=None):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2 if scores is None else scores

    idxs = np.argsort(idxs)

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        suppress = [last]

        for pos in idxs[:-1]:
            xx1 = max(x1[last], x1[pos])
            yy1 = max(y1[last], y1[pos])
            xx2 = min(x2[last], x2[pos])
            yy2 = min(y2[last], y2[pos])

            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            overlap = float(w * h) / area[pos]

            if overlap > max_bbox_overlap:
                suppress.append(pos)

        idxs = np.delete(idxs, suppress)

    return pick
