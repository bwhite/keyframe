import imfeat
import numpy as np
import keyframe
import impoint


class SURF(keyframe.BaseKeyframer):

    def __init__(self, max_diff=5, **kw):
        super(SURF, self).__init__([('opencv', 'gray', 8)], max_diff=max_diff, **kw)
        self._surf = impoint.SURF()

    def feat_func(self, frame):
        return self._surf(frame)

    def diff_func(self, points0, points1):
        return len(self._surf.match(points0, points1))
