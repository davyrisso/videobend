import itertools
import sys

import numpy
import cv2

from ..utils.frame import Frame
from ..utils.optical_flow import OpticalFlow, OpticalFlowGenerator
from ..utils.video import VideoReader, VideoWriter


def GenerateFrames(
        input_video,
        start_frame, end_frame,
        effect_start_frame, effect_end_frame,
        frame_blend_weight=0.0):
    """Applies the effect and returns a generator over the generated frames."""

    # Unaffected frames before effect.
    frames_before = input_video.GetStream().ReadFrames(
        start_frame=start_frame, end_frame=effect_start_frame)

    # Unaffected frames after effect.
    frames_after = input_video.GetStream().ReadFrames(
        start_frame=effect_end_frame, end_frame=end_frame)

    # Affected frames.
    frames = input_video.GetStream().ReadFrames(
        start_frame=effect_start_frame, end_frame=effect_end_frame)

    def GeneratedFrames():
        image_buffer = None

        for frame, optical_flow in OpticalFlowGenerator(frames).GenerateFlow():
            if image_buffer is None:
                image_buffer = frame.pixels.copy()

            cv2.remap(
                src=image_buffer,
                dst=image_buffer,
                map1=optical_flow.motion_vectors_x,
                map2=optical_flow.motion_vectors_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_DEFAULT)

            if frame_blend_weight > 0:
                cv2.remap(
                    src=frame.pixels,
                    dst=frame.pixels,
                    map1=optical_flow.motion_vectors_x,
                    map2=optical_flow.motion_vectors_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_DEFAULT)

                cv2.addWeighted(
                    src1=image_buffer,
                    src2=frame.pixels,
                    dst=image_buffer,
                    alpha=1 - frame_blend_weight,
                    beta=frame_blend_weight,
                    gamma=0)

            yield Frame(image_buffer, frame.position)

    return itertools.chain(frames_before, GeneratedFrames(), frames_after)


def main(input_video_1_path, input_video_2_path, output_video_path,
         start_frame=0, end_frame=None,
         effect_start_frame=0, effect_end_frame=None,
         motion_start_frame=0, motion_end_frame=None,
         frame_blend_weight=0,
         preview=False):
    input_video = VideoReader(file_path=input_video_path)
    motion_video = VideoReader(file_path=motion_video_path)

    output_video = VideoWriter.FromReader(
        input_video, file_path=output_video_path)

    if end_frame is None:
        end_frame = input_video.frame_count

    if effect_end_frame is None:
        effect_end_frame = input_video.frame_count

    if motion_end_frame is None:
        motion_end_frame = motion_video.frame_count

    effect_start_frame = max(effect_start_frame, start_frame)
    effect_end_frame = min(effect_end_frame, end_frame)

    print('\n'.join(['Starting motionmosh effect with parameters:',
                     ' - input: %s' % input_video_path,
                     ' - output: %s' % output_video_path,
                     ' - video:',
                     '   - start frame: %d' % start_frame,
                     '   - end frame: %d' % end_frame,
                     ' - effect:',
                     '   - start frame: %d' % effect_start_frame,
                     '   - effect end frame: %d' % effect_end_frame,
                     '\n']))

    print('\n'.join(['Input video info:',
                     ' - size: %dx%d' % input_video.frame_size,
                     ' - fps: %d' % input_video.fps,
                     ' - frame count: %d' % input_video.frame_count,
                     ' - codec code: %s' % input_video.fourcc,
                     '\n']))

    print('Generating frames...')

    # Gets a generator of transformed frames.
    generated_frames = GenerateFrames(
        input_video,
        start_frame=start_frame,
        end_frame=end_frame,
        effect_start_frame=effect_start_frame,
        effect_end_frame=effect_end_frame,
        frame_blend_weight=frame_blend_weight)

    for frame in output_video.WriteFrames(generated_frames):
        sys.stdout.write(
            'Progress: %d/%d (%.2f%%) \r' %
            (frame.position[0], end_frame,
             float(frame.position[0]) / float(end_frame) * 100))
        sys.stdout.flush()

        if preview:
            cv2.imshow('Preview', frame.pixels)

    print('\nDone writing %s.' % output_video_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=('Uses Optical Flow to mix the motion of two videos.'))

    parser.add_argument(
        'input_video_1_path',
        type=str,
        help='Path of the first input video')

    parser.add_argument(
        'input_video_2_path',
        type=str,
        help='Path of the second input video')

    parser.add_argument(
        'output_video_path',
        type=str,
        help='Output video file path')

    parser.add_argument(
        '-s_1', '--input_1_start_frame',
        metavar='<frame>',
        type=int,
        default=0,
        help='Start frame for input video 1. Default: 0 (first frame)')

    parser.add_argument(
        '-e_1', '--input_1_end_frame',
        metavar='<frame>',
        type=int,
        default=None,
        help='End frame for input video 1. Default: last frame.')

    parser.add_argument(
        '-s_2', '--input_2_start_frame',
        metavar='<frame>',
        type=int,
        default=0,
        help='Start frame for input video 2. Default: 0 (first frame)'
    )

    parser.add_argument(
        '-e_2', '--input_2_end_frame',
        metavar='<frame>',
        type=int,
        default=0,
        help='End frame for input video 2. Default: last frame')

    parser.add_argument(
        '-m_w_1', '--input_1_motion_weight',
        metavar='<0.0-1.0>',
        type=float,
        default=0.0,
        help='Weight of the motion for the first input video')

    parser.add_argument(
        '-m_w_2', '--input_2_motion_weight',
        metavar='<0.0-1.0>',
        type=float,
        default=0.0,
        help='Weight of the motion for the second input video')

    parser.add_argument(
        '--preview',
        action='count',
        default=0,
        help='Shows a preview of the generated frames during computation.')

    args = parser.parse_args()

    main(
        args.input_video_path,
        args.output_video_path,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        effect_start_frame=args.effect_start_frame,
        effect_end_frame=args.effect_end_frame,
        frame_blend_weight=args.frame_blend_weight,
        preview=bool(args.preview))
