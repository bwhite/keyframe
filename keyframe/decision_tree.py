import imfeat
import numpy as np
import keyframe
import impoint
import glob
import cv
import os
import vidfeat
import random
import classipy
import cPickle as pickle


class DecisionTree(keyframe.BaseKeyframer):

    def __init__(self, min_diff=.8, **kw):
        super(DecisionTree, self).__init__([('opencv', 'bgr', 8)], min_diff=min_diff, **kw)
        self._surf = impoint.SURF()
        self._feat = imfeat.Histogram('lab', num_bins=8)
        self.rfc = None

    def train(self, gt_path, video_path, hard_negatives=False):
        if not hard_negatives:
            self.pos_label_values = []
            self.neg_label_values = []
        label_values = []
        for vid_num, gt_fn in enumerate(glob.glob(gt_path + '/*.txt')):
            print('GT:[%s, %s]' % (vid_num, gt_fn))
            with open(gt_fn) as fp:
                keyframes = set([int(line) for line in fp.read().strip().split()])
            event, video_name = gt_fn[:-4].split('/')[-1].split('_')
            prev_frame = None
            pos_diffs = {}
            neg_diffs = {}
            min_frame_dist = 120
            last_neg = -min_frame_dist + 1
            try:
                os.makedirs('out_frames/all')
                os.makedirs('out_frames/pos')
                os.makedirs('out_frames/neg')
            except OSError:
                pass
            fn = '%s/%s/%s-compressed.avi' % (video_path, event, video_name)
            if not os.path.exists(fn):
                continue
            for frame_num, frame_time, frame in vidfeat.convert_video_ffmpeg(fn, ('frameiter', self.MODES)):
                if 0 and keyframes:
                    cv.SaveImage('out_frames/all/frame%.8d-%.8d-%.8d.jpg' % (vid_num, frame_time * 100, frame_num), frame)
                if prev_frame:
                    if frame_num in keyframes and not hard_negatives:
                        if prev_frame:
                            cv.SaveImage('out_frames/pos/frame%.8d-%.8d-.jpg' % (vid_num, frame_num - 1), prev_frame)
                            cv.SaveImage('out_frames/pos/frame%.8d-%.8d+.jpg' % (vid_num, frame_num), frame)
                        diff = self._diff_func(self.feat_func(prev_frame), self.feat_func(frame))
                        print(diff)
                        pos_diffs[frame_num] = diff
                    elif not keyframes and frame_num - last_neg > min_frame_dist:
                        print('Making neg[%d]' % frame_num)
                        # Must be 60 frames away from a previous neg
                        cur_feat = self.feat_func(frame)
                        prev_feat = self.feat_func(prev_frame)
                        if hard_negatives:
                            score = self.diff_func(prev_feat, cur_feat)
                            print('Hard Neg Score[%f]' % score)
                            if self._min_diff <= score or score <= self._max_diff:
                                diff = self._diff_func(prev_feat, cur_feat)
                                print(diff)
                                self.neg_label_values.append((0, diff))
                                cv.SaveImage('out_frames/neg/frame%.8d-%.8d-.jpg' % (vid_num, frame_num - 1), prev_frame)
                                cv.SaveImage('out_frames/neg/frame%.8d-%.8d+.jpg' % (vid_num, frame_num), frame)
                        else:
                            diff = self._diff_func(prev_feat, cur_feat)
                            print(diff)
                            neg_diffs[frame_num] = diff
                        last_neg = frame_num
                prev_frame = frame
            if not hard_negatives:
                self.neg_label_values += [(0, x) for x in neg_diffs.values()]
            self.pos_label_values += [(1, x) for x in pos_diffs.values()]
        print('NumPos[%d] NumNeg[%d]' % (len(self.pos_label_values), len(self.neg_label_values)))
        label_values = self.pos_label_values + self.neg_label_values
        with open('lv.pkl', 'w') as fp:
            pickle.dump(label_values, fp, -1)
        print('NumPos[%d] NumNeg[%d]' % (len(self.pos_label_values), len(self.neg_label_values)))
        values = np.vstack(zip(*label_values)[1])
        dims = zip(np.min(values, 0), np.max(values, 0))
        self.rfc = classipy.RandomForestClassifier(classipy.rand_forest.VectorFeatureFactory(dims, np.zeros(len(dims)), 100), num_feat=1000, tree_depth=4)
        for _ in range(5):
            self.rfc.train(label_values, replace=False)

    def save(self, fn='out.pkl'):
        with open(fn, 'w') as fp:
            pickle.dump(self.rfc.trees_ser, fp, -1)

    def load(self, fn='out.pkl'):
        with open(fn) as fp:
            self.rfc = classipy.RandomForestClassifier(classipy.rand_forest.VectorFeatureFactory(), trees_ser=pickle.load(fp))

    def _hist_feat_func(self, frame):
        width, height = frame.width, frame.height
        cgr = imfeat.CoordGeneratorRect
        out = []
        for block, trans in imfeat.BlockGenerator(frame, cgr, output_size=(height / 4, width / 4), step_delta=(height / 4, width / 4)):
            out.append(self._feat(block))
        out.append(out[0] + out[1] + out[4] + out[5])
        out.append(out[2] + out[3] + out[6] + out[7])
        out.append(out[8] + out[9] + out[12] + out[13])
        out.append(out[10] + out[11] + out[14] + out[15])
        out = [out[16] + out[17] + out[18] + out[19]]
        return out

    def feat_func(self, frame):
        gray_frame = imfeat.convert_image(frame, [('opencv', 'gray', 8)])
        #gray_frame_f = imfeat.convert_image(frame, [('opencv', 'gray', 32)])
        s = self._surf(gray_frame)
        h = self._hist_feat_func(frame)
        return {'surf': s, 'hist': h}

    def _diff_func(self, points0, points1):
        num_match = np.asfarray([len(self._surf.match(points0['surf'], points1['surf'])) / (len(points0['surf']) * len(points1['surf']) + .00000001)])
        hist_diff_sums = np.array([np.sum(np.abs(x - y)) for x, y in zip(points0['hist'], points1['hist'])])
        return np.hstack([num_match, hist_diff_sums])

    def diff_func(self, points0, points1):
        preds = sorted([x[::-1] for x in self.rfc.predict(self._diff_func(points0, points1))])
        a = preds[1][1]
        return a


if __name__ == '__main__':
    a = DecisionTree()
    a.train('../utils/gt', '../utils/events', hard_negatives=False)
    a.save()
    #a.load()
    a.train('../utils/gt', '../utils/events', hard_negatives=True)
    a.save('out2.pkl')
