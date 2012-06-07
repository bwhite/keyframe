import imfeat
import numpy as np
import keyframe
import impoint
import cv2


class SURF(keyframe.BaseKeyframer):

    def __init__(self, max_diff=5, **kw):
        super(SURF, self).__init__(max_diff=max_diff, **kw)
        self._surf = impoint.SURF()

    def feat_func(self, frame):
        frame = cv2.resize(frame, (256, 256))
        return self._surf(frame)

    def diff_func(self, points0, points1):
        return len(self._surf.match(points0, points1))
