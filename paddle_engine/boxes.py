"""Axis-aligned boxes as [x0, y0, x1, y1] in work-image pixels."""
import numpy as np


def height(b):
    return b[3] - b[1]


def from_polygon(poly):
    p = np.asarray(poly)
    return [int(p[:, 0].min()), int(p[:, 1].min()), int(p[:, 0].max()), int(p[:, 1].max())]


def row_overlap(a, b):
    """Shared height as a share of the shorter box: 1 on the same row, <= 0 on different rows."""
    return (min(a[3], b[3]) - max(a[1], b[1])) / max(1, min(height(a), height(b)))


def gap(a, b):
    """Horizontal gap between the boxes; negative when they overlap."""
    return max(a[0], b[0]) - min(a[2], b[2])


def union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def intersects(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def merge_overlapping(boxes):
    """Two detections of one line that overlap side by side become one: the detector cuts a
    headline amount in two colours ("750" black + "EGP" orange) into overlapping boxes, the
    second starting inside the last digit ("0 EGP")."""
    boxes = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if row_overlap(boxes[i], boxes[j]) >= 0.5 and gap(boxes[i], boxes[j]) < 0:
                    boxes[i] = union(boxes[i], boxes.pop(j))
                    merged = True
                    break
            if merged:
                break
    return boxes


def crop(img, b, pad):
    h, w = img.shape[:2]
    return img[max(0, b[1] - pad):min(h, b[3] + pad), max(0, b[0] - pad):min(w, b[2] + pad)]
