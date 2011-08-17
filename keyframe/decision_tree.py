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

    def __init__(self, min_diff=.5, **kw):
        super(DecisionTree, self).__init__([('opencv', 'gray', 8)], min_diff=min_diff, **kw)
        self._surf = impoint.SURF()
        self._feat = imfeat.Histogram('gray', num_bins=4, norm=False)
        self.rfc = None

    def train(self, gt_path, video_path, hard_negatives=False):
        pos_label_values = []
        neg_label_values = []
        label_values = []
        for vid_num, gt_fn in enumerate(glob.glob(gt_path + '/*.txt')):
            print('GT:[%s, %s]' % (vid_num, gt_fn))
            with open(gt_fn) as fp:
                keyframes = set([int(line) for line in fp.read().strip().split()])
                keyframes_np = np.array(list(keyframes))
            print(keyframes)
            event, video_name = gt_fn[:-4].split('/')[-1].split('_')
            prev_feat = None
            prev_frame = None
            pos_diffs = {}
            neg_diffs = {}
            last_hard_neg = -100000
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
                cur_feat = self.feat_func(frame)
                if 0 and keyframes:
                    cv.SaveImage('out_frames/all/frame%.8d-%.8d-%.8d.jpg' % (vid_num, frame_time * 100, frame_num), frame)
                if prev_feat:
                    if frame_num in keyframes:
                        if prev_frame:
                            cv.SaveImage('out_frames/pos/frame%.8d-%.8d-.jpg' % (vid_num, frame_num - 1), prev_frame)
                            cv.SaveImage('out_frames/pos/frame%.8d-%.8d+.jpg' % (vid_num, frame_num), frame)
                        diff = self._diff_func(prev_feat, cur_feat)
                        pos_diffs[frame_num] = diff
                    elif np.all(np.abs(keyframes_np - frame_num) > 60):  # Must be 60 frames away from a positive
                        if hard_negatives:
                            if frame_num - last_hard_neg > 60:
                                score = self.diff_func(prev_feat, cur_feat)
                                if self._min_diff <= score or score <= self._max_diff:
                                    diff = self._diff_func(prev_feat, cur_feat)
                                    neg_label_values.append((0, diff))
                                    last_hard_neg = frame_num
                                    #cv.SaveImage('out_frames/neg/frame%.8d-%.8d-.jpg' % (vid_num, frame_num - 1), prev_frame)
                                    #cv.SaveImage('out_frames/neg/frame%.8d-%.8d+.jpg' % (vid_num, frame_num), frame)
                        else:
                            diff = self._diff_func(prev_feat, cur_feat)
                            neg_diffs[frame_num] = diff
                prev_feat = cur_feat
                prev_frame = frame
            if not hard_negatives:
                cur_num_neg = min(len(neg_diffs), len(pos_label_values) * 10)
                neg_label_values += [(0, x) for x in random.sample(neg_diffs.values(), cur_num_neg)]
            pos_label_values += [(1, x) for x in pos_diffs.values()]
        num_samples = min(len(pos_label_values), len(neg_label_values))
        print('Before Prune: NumPos[%d] NumNeg[%d]' % (len(pos_label_values), len(neg_label_values)))
        pos_label_values = random.sample(pos_label_values, num_samples)
        neg_label_values = random.sample(neg_label_values, min(num_samples * 10, len(neg_label_values)))
        label_values = pos_label_values + neg_label_values
        with open('lv.pkl', 'w') as fp:
            pickle.dump(label_values, fp, -1)
        print('NumPos[%d] NumNeg[%d]' % (len(pos_label_values), len(neg_label_values)))
        values = np.vstack(zip(*label_values)[1])
        dims = zip(np.min(values, 0), np.max(values, 0))
        self.rfc = classipy.RandomForestClassifier(classipy.rand_forest.VectorFeatureFactory(dims, np.zeros(len(dims)), 100), num_feat=1000, tree_depth=3)
        self.rfc.train(label_values)

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
        return np.hstack(out)

    def feat_func(self, frame):
        gray_frame = imfeat.convert_image(frame, [('opencv', 'gray', 8)])
        return {'surf': self._surf(gray_frame), 'hist': self._hist_feat_func(gray_frame)}

    def _diff_func(self, points0, points1):
        num_match = np.asfarray([len(self._surf.match(points0['surf'], points1['surf']))])
        hist_diff = np.abs(points0['hist'] - points1['hist'])
        hist_diff_med = np.median(hist_diff)
        hist_diff_sum = np.sum(hist_diff)
        eps = .000001
        hist_norm_diff = np.abs(points0['hist'] / (np.sum(points0['hist']) + eps) - points1['hist'] / (np.sum(points1['hist']) + eps))
        hist_norm_diff_med = np.median(hist_norm_diff)
        hist_norm_diff_sum = np.sum(hist_norm_diff)
        return np.hstack([num_match, hist_diff_sum, hist_norm_diff_sum, hist_diff_med, hist_norm_diff_med])

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
