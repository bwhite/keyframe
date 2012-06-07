import imfeat
import numpy as np
import keyframe


class Meta(keyframe.BaseKeyframer):

    def __init__(self, kfs, min_diff=1, **kw):
        super(Meta, self).__init__(min_diff=min_diff, **kw)
        self._kfs = kfs

    def feat_func(self, frame):
        return [x.feat_func(frame) for x in self._kfs]

    def diff_func(self, vec0, vec1):
        scores = [k.diff_func(x, y) for k, x, y in zip(self._kfs, vec0, vec1)]
        print 'scores', scores
        outs = [int(k._min_diff <= score or score <= k._max_diff) for k, score in zip(self._kfs, scores)]
        print 'outs', outs
        return sum(outs)
