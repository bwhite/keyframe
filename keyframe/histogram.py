import imfeat
import numpy as np
import keyframe


class Histogram(keyframe.BaseKeyframer):

    def __init__(self, min_diff=10, **kw):
        super(Histogram, self).__init__([('opencv', 'gray', 8)], min_diff=min_diff, **kw)
        self._feat = imfeat.Histogram('gray')

    def feat_func(self, frame):
        width, height = frame.width, frame.height
        cgr = imfeat.CoordGeneratorRect
        out = []
        for block, trans in imfeat.BlockGenerator(frame, cgr, output_size=(50, 50), step_delta=(50, 50)):
            out.append(self._feat(block))
        return np.hstack(out)

    def diff_func(self, vec0, vec1):
        return np.sum(np.abs(vec0 - vec1))
