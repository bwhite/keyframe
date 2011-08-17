import keyframe
import pyffmpeg
import cv
import vidfeat
import matplotlib
matplotlib.use('agg')
import pylab
import os


def main(path, output_dir):
    try:
        os.makedirs(output_dir)
    except OSError:
        pass
    if 1:
        #kf = keyframe.Meta([keyframe.SURF(max_diff=15), keyframe.Histogram(min_diff=5)], min_diff=2)
        kf = keyframe.DecisionTree()
        kf.load('out2.pkl')
        #stream = pyffmpeg.VideoStream()
        #stream.open(path)
        prev_frame = None
        for (frame_num, frame_time, frame), iskeyframe in kf(vidfeat.convert_video_ffmpeg(path, ('frameiter', kf.MODES))):
            if prev_frame:
                if iskeyframe:
                    cv.SaveImage(output_dir + '/%.8d-.jpg' % (frame_num - 1), prev_frame)
                    cv.SaveImage(output_dir + '/%.8d+.jpg' % frame_num, frame)
            prev_frame = frame

        pylab.figure(1)
        pylab.clf()
        pylab.plot(-kf.get_scores())
        pylab.savefig('matches_fig.png')
    else:
        kf = keyframe.Histogram()
        stream = pyffmpeg.VideoStream()
        stream.open(path)
        prev_frame = None
        for (frame_num, frame_time, frame), iskeyframe in kf(vidfeat.convert_video(stream,
                                                                                   ('frameiter', kf.MODES))):
            if prev_frame:
                if iskeyframe:
                    cv.SaveImage(output_dir + '/%.8d-.jpg' % (frame_num - 1), prev_frame)
                    cv.SaveImage(output_dir + '/%.8d+.jpg' % frame_num, frame)
            prev_frame = frame

        pylab.figure(1)
        pylab.clf()
        pylab.plot(kf.get_scores())
        pylab.savefig('scores_fig.png')


def _parser():
    import argparse
    parser = argparse.ArgumentParser('Output pairs bordering on keyframes')
    parser.add_argument('input_video', help='Input video path')
    parser.add_argument('output_dir', help='Output dir')
    return parser.parse_args()

if __name__ == '__main__':
    args = _parser()
    main(args.input_video, args.output_dir)
