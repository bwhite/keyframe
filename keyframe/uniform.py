import imfeat
import numpy as np
import keyframe
import impoint


class Uniform(keyframe.BaseKeyframer):

    def __init__(self, **kw):
        super(Uniform, self).__init__([('opencv', 'bgr', 8), ('opencv', 'gray', 8)],
                                      min_diff=float('-inf'), **kw)

    def feat_func(self, frame):
        return 0

    def diff_func(self, points0, points1):
        return 0
