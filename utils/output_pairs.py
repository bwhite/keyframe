import keyframe
import cv
import cv2
import vidfeat
import matplotlib
matplotlib.use('agg')
import pylab
import os
import numpy as np


def plot_matches(image, matches, points0, points1, max_feat_width, color=(0, 255, 0)):
    if image.width > max_feat_width:
            height = int((max_feat_width / float(image.width)) * image.height)
            image = cv.fromarray(cv2.resize(np.asarray(cv.GetMat(image)), (max_feat_width, height)))
    for match in matches:
        point0 = points0[match[0]]
        point1 = points1[match[1]]
        cv.Line(image, (int(point0['x']), int(point0['y'])), (int(point1['x']), int(point1['y'])), color=color)
    return image


def main(path, output_dir):
    try:
        os.makedirs(output_dir + '/surf')
    except OSError:
        pass
    #kf = keyframe.Meta([keyframe.SURF(max_diff=15), keyframe.Histogram(min_diff=5)], min_diff=2)
    kf = keyframe.DecisionTree(verbose=True, min_diff=.5, min_interval=.5)
    kf.load('out2.pkl')
    prev_frame = None
    show_surf = 1
    for (frame_num, frame_time, frame), iskeyframe in kf(vidfeat.convert_video_ffmpeg(path, ('frameiter', kf.MODES))):
        #cv.SaveImage(output_dir + '/all/%.8d-_%f.jpg' % (frame_num, frame_time), frame)
        if prev_frame:
            if iskeyframe:
                cv.SaveImage(output_dir + '/%.8d-_%f_%f.jpg' % (frame_num - 1, frame_time, kf.scores[-1]), prev_frame)
                cv.SaveImage(output_dir + '/%.8d+_%f_%f.jpg' % (frame_num, frame_time, kf.scores[-1]), frame)
            if show_surf:
                prev_frame = plot_matches(prev_frame, kf.surf_debug['matches'], kf.surf_debug['points0'], kf.surf_debug['points1'], max_feat_width=kf.max_feat_width)
                cv.SaveImage(output_dir + '/surf/%.8d+_%f_%f.jpg' % (frame_num, frame_time, kf.scores[-1]), prev_frame)
        prev_frame = frame
    pylab.figure(1)
    pylab.clf()
    pylab.plot(kf.get_scores())
    pylab.savefig('matches_fig.png')


def _parser():
    import argparse
    parser = argparse.ArgumentParser('Output pairs bordering on keyframes')
    parser.add_argument('input_video', help='Input video path')
    parser.add_argument('output_dir', help='Output dir')
    return parser.parse_args()

if __name__ == '__main__':
    args = _parser()
    main(args.input_video, args.output_dir)
