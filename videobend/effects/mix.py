import itertools
import sys

import numpy
import cv2

from ..utils.constants import BORDER_MODES, DEFAULT_BORDER_VALUE, INTERPOLATION_METHODS
from ..utils.frame import Frame
from ..utils.optical_flow import OpticalFlow, OpticalFlowGenerator
from ..utils.video import VideoReader, VideoWriter


def GenerateFrames(
    input_video_1, input_video_2,
    input_1_start_frame,
        input_1_end_frame,
        input_2_start_frame,
        input_2_end_frame,
        input_1_motion_weight,
        input_1_motion_multiplier,
        input_1_motion_threshold,
        input_2_motion_weight,
        input_2_motion_multiplier,
        input_2_motion_threshold,
        input_1_frame_blend_weight,
        input_2_frame_blend_weight,
        interpolation_method,
        border_mode):
    """Applies the effect and returns a generator over the generated frames."""

    input_1_frames = input_video_1.GetStream().ReadFrames(
        start_frame=input_1_start_frame, end_frame=input_1_end_frame)

    input_2_frames = input_video_2.GetStream().ReadFrames(
        start_frame=input_2_start_frame, end_frame=input_2_end_frame)

    input_1_optical_flows = OpticalFlowGenerator(input_1_frames).GenerateFlow()
    input_2_optical_flows = OpticalFlowGenerator(input_2_frames).GenerateFlow()

    def GeneratedFrames():
        image_buffer = None

        for input_1_frame, input_1_optical_flow in input_1_optical_flows:
            input_2_frame, input_2_optical_flow = input_2_optical_flows.next()

            if image_buffer is None:
                image_buffer = input_1_frame.pixels.copy()

            total_flow = numpy.add(
                ((input_1_optical_flow.flow * (
                  abs(input_1_optical_flow.flow) >= input_1_motion_threshold)) *
                 input_1_motion_multiplier * input_1_motion_weight),
                ((input_2_optical_flow.flow * (
                  abs(input_2_optical_flow.flow) >= input_2_motion_threshold)) *
                 input_2_motion_multiplier * input_2_motion_weight))

            remap_vectors = OpticalFlow.GetRemapVectors(total_flow)

            cv2.remap(
                src=image_buffer,
                dst=image_buffer,
                map1=remap_vectors[0],
                map2=remap_vectors[1],
                interpolation=interpolation_method,
                borderMode=border_mode)

            if input_1_frame_blend_weight > 0:
                input_1_remap_vectors = OpticalFlow.GetRemapVectors(
                    input_1_optical_flow.flow,
                    multiplier_x=input_1_motion_multiplier,
                    multiplier_y=input_1_motion_multiplier)

                cv2.remap(
                    src=input_1_frame.pixels,
                    dst=input_1_frame.pixels,
                    map1=input_1_remap_vectors[0],
                    map2=input_1_remap_vectors[1],
                    interpolation=interpolation_method,
                    borderMode=border_mode)

                cv2.addWeighted(
                    src1=image_buffer,
                    src2=input_1_frame.pixels,
                    dst=image_buffer,
                    alpha=1 - input_1_frame_blend_weight,
                    beta=input_1_frame_blend_weight,
                    gamma=0)

            if input_2_frame_blend_weight > 0:
                input_2_remap_vectors = OpticalFlow.GetRemapVectors(
                    input_2_optical_flow.flow,
                    multiplier_x=input_2_motion_multiplier,
                    multiplier_y=input_2_motion_multiplier)

                cv2.remap(
                    src=input_2_frame.pixels,
                    dst=input_2_frame.pixels,
                    map1=input_2_remap_vectors[0],
                    map2=input_2_remap_vectors[1],
                    interpolation=interpolation_method,
                    borderMode=border_mode)

                cv2.addWeighted(
                    src1=image_buffer,
                    src2=input_2_frame.pixels,
                    dst=image_buffer,
                    alpha=1 - input_2_frame_blend_weight,
                    beta=input_2_frame_blend_weight,
                    gamma=0)

            yield Frame(image_buffer, input_1_frame.position)

    return itertools.chain(GeneratedFrames())


def main(input_video_1_path,
         input_video_2_path,
         output_video_path,
         input_1_start_frame=0,
         input_1_end_frame=None,
         input_2_start_frame=0,
         input_2_end_frame=None,
         input_1_motion_weight=1.0,
         input_1_motion_multiplier=1.0,
         input_1_motion_threshold=0.0,
         input_2_motion_weight=1.0,
         input_2_motion_multiplier=1.0,
         input_2_motion_threshold=0.0,
         input_1_frame_blend_weight=0.0,
         input_2_frame_blend_weight=0.0,
         interpolation_method=0,
         border_mode=1,
         preview=False):
    input_video_1 = VideoReader(file_path=input_video_1_path)
    input_video_2 = VideoReader(file_path=input_video_2_path)

    output_video = VideoWriter.FromReader(
        input_video_1, file_path=output_video_path)

    if input_1_end_frame is None:
        input_1_end_frame = input_video_1.frame_count

    if input_2_end_frame is None:
        input_2_end_frame = input_video_2.frame_count

    # Gets a generator of transformed frames.
    print('Writing frames to %s...' % output_video_path)
    generated_frames = GenerateFrames(
        input_video_1=input_video_1,
        input_video_2=input_video_2,
        input_1_start_frame=input_1_start_frame,
        input_1_end_frame=input_1_end_frame,
        input_2_start_frame=input_2_start_frame,
        input_2_end_frame=input_2_end_frame,
        input_1_motion_weight=input_1_motion_weight,
        input_1_motion_multiplier=input_1_motion_multiplier,
        input_1_motion_threshold=input_1_motion_threshold,
        input_2_motion_weight=input_2_motion_weight,
        input_2_motion_multiplier=input_2_motion_multiplier,
        input_2_motion_threshold=input_2_motion_threshold,
        input_1_frame_blend_weight=input_1_frame_blend_weight,
        input_2_frame_blend_weight=input_2_frame_blend_weight,
        interpolation_method=interpolation_method,
        border_mode=border_mode
    )

    for frame in output_video.WriteFrames(generated_frames):
        sys.stdout.write(
            'Progress: %d/%d (%.2f%%) \r' %
            (frame.position[0], input_1_end_frame,
             float(frame.position[0]) / float(input_1_end_frame) * 100))
        sys.stdout.flush()

        if preview:
            cv2.imshow('Preview', frame.pixels)

    print('\nDone writing %s.' % output_video_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=('Uses Optical Flow to mix the motion of two videos.'))

    parser.add_argument(
        '-i_1', '--input_video_1_path',
        required=True,
        type=str,
        help='Path of the first input video')

    parser.add_argument(
        '-i_2', '--input_video_2_path',
        required=True,
        type=str,
        help='Path of the second input video')

    parser.add_argument(
        '-o', '--output_video_path',
        required=True,
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
        default=None,
        help='End frame for input video 2. Default: last frame')

    parser.add_argument(
        '-m_w_1', '--input_1_motion_weight',
        metavar='<0.0-1.0>',
        type=float,
        default=1.0,
        help='Weight of the motion for the first input video')

    parser.add_argument(
        '-m_m_1', '--input_1_motion_multiplier',
        metavar='<0.0-1.0>',
        type=float,
        default=1.0,
        help='Multiplier for the motion for the first input video')

    parser.add_argument(
        '-m_t_1', '--input_1_motion_threshold',
        metavar='<0.0-1.0>',
        type=float,
        default=0.0,
        help='Threshold for the motion for the first input video')

    parser.add_argument(
        '-m_w_2', '--input_2_motion_weight',
        metavar='<0.0-1.0>',
        type=float,
        default=1.0,
        help='Weight of the motion for the second input video')

    parser.add_argument(
        '-m_m_2', '--input_2_motion_multiplier',
        metavar='<0.0-1.0>',
        type=float,
        default=1.0,
        help='Multiplier for the motion for the second input video')

    parser.add_argument(
        '-m_t_2', '--input_2_motion_threshold',
        metavar='<0.0-1.0>',
        type=float,
        default=0.0,
        help='Threshold for the motion for the second input video')

    parser.add_argument(
        '-f_b_w_1', '--input_1_frame_blend_weight',
        metavar='<0.0-1.0>',
        type=float,
        default=0.0,
        help=('Blend weight for the transformed frame in the final image. ' +
              'Used to preserve detail from the frames throuhougt the video. ' +
              'If > 0 the transformed frames will be blended with the result ' +
              'images, otherwise only the first frame\'s pixels will be used.'))

    parser.add_argument(
        '-f_b_w_2', '--input_2_frame_blend_weight',
        metavar='<0.0-1.0>',
        type=float,
        default=0.0,
        help=('Blend weight for the transformed frame in the final image. ' +
              'Used to preserve detail from the frames throuhougt the video. ' +
              'If > 0 the transformed frames will be blended with the result ' +
              'images, otherwise only the first frame\'s pixels will be used.'))

    parser.add_argument(
        '--interpolation_method',
        metavar='<0-4>',
        type=int,
        default=0,
        help=(
            'Interpolation method. Values: %s. Default: 0 (%s).' % (
                INTERPOLATION_METHODS, INTERPOLATION_METHODS[0])))

    parser.add_argument(
        '--border_mode',
        metavar='<0-16>',
        type=int,
        default=1,
        help=(
            'Border mode. Values: %s. Default: 1 (%s).' % (
                BORDER_MODES, BORDER_MODES[1])))

    parser.add_argument(
        '--preview',
        action='count',
        default=0,
        help='Shows a preview of the generated frames during computation.')

    args = parser.parse_args()

    main(
        input_video_1_path=args.input_video_1_path,
        input_video_2_path=args.input_video_2_path,
        output_video_path=args.output_video_path,
        input_1_start_frame=args.input_1_start_frame,
        input_1_end_frame=args.input_1_end_frame,
        input_2_start_frame=args.input_2_start_frame,
        input_2_end_frame=args.input_2_end_frame,
        input_1_motion_weight=args.input_1_motion_weight,
        input_1_motion_multiplier=args.input_1_motion_multiplier,
        input_1_motion_threshold=args.input_1_motion_threshold,
        input_2_motion_weight=args.input_2_motion_weight,
        input_2_motion_multiplier=args.input_2_motion_multiplier,
        input_2_motion_threshold=args.input_2_motion_threshold,
        input_1_frame_blend_weight=args.input_1_frame_blend_weight,
        input_2_frame_blend_weight=args.input_2_frame_blend_weight,
        interpolation_method=args.interpolation_method,
        border_mode=args.border_mode,
        preview=bool(args.preview))
