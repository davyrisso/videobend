import itertools
import sys

import numpy
import cv2

from ..utils.constants import BORDER_MODES, DEFAULT_BORDER_VALUE, INTERPOLATION_METHODS
from ..utils.frame import Frame
from ..utils.optical_flow import OpticalFlow, OpticalFlowGenerator
from ..utils.video import VideoReader, VideoWriter


def GenerateFrames(
        input_video,
        start_frame, end_frame,
        repeat_effect_start_frame, repeat_effect_end_frame,
        repeat_effect_motion_multiplier_x, repeat_effect_motion_multiplier_y,
        repeat_effect_motion_threshold_x, repeat_effect_motion_threshold_y,
        mosh_effect_start_frame, mosh_effect_end_frame,
        mosh_effect_motion_multiplier_x, mosh_effect_motion_multiplier_y,
        mosh_effect_motion_threshold_x, mosh_effect_motion_threshold_y,
        frame_blend_weight, interpolation_method, border_mode):
    """Applies the effect and returns a generator over the generated frames."""

    effects_start_frame = min(
        repeat_effect_start_frame, mosh_effect_start_frame)
    effects_end_frame = max(repeat_effect_end_frame, mosh_effect_end_frame)

    # Unaffected frames before effect.
    frames_before = input_video.GetStream().ReadFrames(
        start_frame=start_frame, end_frame=effects_start_frame)

    # Unaffected frames after effect.
    frames_after = input_video.GetStream().ReadFrames(
        start_frame=effects_end_frame, end_frame=end_frame)

    # Affected frames.
    frames = input_video.GetStream().ReadFrames(
        start_frame=effects_start_frame, end_frame=effects_end_frame)

    def GeneratedFrames():
        image_buffer = None
        repeat_remap_vectors = None

        for frame, optical_flow in OpticalFlowGenerator(frames).GenerateFlow():
            if image_buffer is None:
                image_buffer = frame.pixels.copy()

            # Repeat effect remap vectors calculation.
            if frame.position_frames == repeat_effect_start_frame + 1:
                repeat_remap_vectors = optical_flow.GetRemapVectors(
                    optical_flow.flow,
                    multiplier_x=repeat_effect_motion_multiplier_x,
                    multiplier_y=repeat_effect_motion_multiplier_y,
                    threshold_x=repeat_effect_motion_threshold_x,
                    threshold_y=repeat_effect_motion_threshold_y)

            # Repeat effect.
            if (frame.position_frames > repeat_effect_start_frame + 1 and
                    frame.position_frames <= repeat_effect_end_frame):
                cv2.remap(
                    src=image_buffer,
                    dst=image_buffer,
                    map1=repeat_remap_vectors[0],
                    map2=repeat_remap_vectors[1],
                    interpolation=interpolation_method,
                    borderMode=border_mode,
                    borderValue=DEFAULT_BORDER_VALUE)

            # Mosh effect.
            if (frame.position_frames >= mosh_effect_start_frame and
                    frame.position_frames <= mosh_effect_end_frame):
                remap_vectors = optical_flow.GetRemapVectors(
                    optical_flow.flow,
                    multiplier_x=mosh_effect_motion_multiplier_x,
                    multiplier_y=mosh_effect_motion_multiplier_y,
                    threshold_x=mosh_effect_motion_threshold_x,
                    threshold_y=mosh_effect_motion_threshold_y)

                cv2.remap(
                    src=image_buffer,
                    dst=image_buffer,
                    map1=remap_vectors[0],
                    map2=remap_vectors[1],
                    interpolation=interpolation_method,
                    borderMode=border_mode,
                    borderValue=DEFAULT_BORDER_VALUE)

            # Frame blending.
            if frame_blend_weight > 0:
                cv2.remap(
                    src=frame.pixels,
                    dst=frame.pixels,
                    map1=remap_vectors[0],
                    map2=remap_vectors[1],
                    interpolation=interpolation_method,
                    borderMode=border_mode,
                    borderValue=DEFAULT_BORDER_VALUE)
                cv2.addWeighted(
                    src1=image_buffer,
                    src2=frame.pixels,
                    dst=image_buffer,
                    alpha=1 - frame_blend_weight,
                    beta=frame_blend_weight,
                    gamma=0)

            yield Frame(image_buffer, frame.position)

    return itertools.chain(frames_before, GeneratedFrames(), frames_after)


def main(input_video_path, output_video_path, start_frame=0, end_frame=None,
         repeat_effect_start_frame=0, repeat_effect_end_frame=None,
         repeat_effect_motion_threshold=0,
         repeat_effect_motion_multiplier_x=1.0,
         repeat_effect_motion_multiplier_y=1.0,
         mosh_effect_start_frame=0, mosh_effect_end_frame=None,
         mosh_effect_motion_threshold=0.0,
         mosh_effect_motion_multiplier_x=1.0,
         mosh_effect_motion_multiplier_y=1.0,
         frame_blend_weight=0.0,
         interpolation_method=0, border_mode=1, preview=False):
    input_video = VideoReader(file_path=input_video_path)
    output_video = VideoWriter.FromReader(
        input_video, file_path=output_video_path)

    if end_frame is None:
        end_frame = input_video.frame_count

    if repeat_effect_end_frame is None:
        repeat_effect_end_frame = input_video.frame_count

    if mosh_effect_end_frame is None:
        mosh_effect_end_frame = input_video.frame_count

    repeat_effect_start_frame = max(repeat_effect_start_frame, start_frame)
    mosh_effect_end_frame = max(mosh_effect_end_frame, start_frame)

    repeat_effect_start_frame = min(repeat_effect_start_frame, end_frame)
    mosh_effect_end_frame = min(mosh_effect_end_frame, end_frame)

    print('\n'.join(['Starting motionrepeat effect with parameters:',
                     ' - input: %s' % input_video_path,
                     ' - output: %s' % output_video_path,
                     ' - video:',
                     '   - start frame: %d' % start_frame,
                     '   - end frame: %d' % end_frame,
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
        repeat_effect_start_frame=repeat_effect_start_frame,
        repeat_effect_end_frame=repeat_effect_end_frame,
        repeat_effect_motion_multiplier_x=repeat_effect_motion_multiplier_x,
        repeat_effect_motion_multiplier_y=repeat_effect_motion_multiplier_y,
        repeat_effect_motion_threshold_x=repeat_effect_motion_threshold,
        repeat_effect_motion_threshold_y=repeat_effect_motion_threshold,
        mosh_effect_start_frame=mosh_effect_start_frame,
        mosh_effect_end_frame=mosh_effect_end_frame,
        mosh_effect_motion_multiplier_x=mosh_effect_motion_multiplier_x,
        mosh_effect_motion_multiplier_y=mosh_effect_motion_multiplier_y,
        mosh_effect_motion_threshold_x=mosh_effect_motion_threshold,
        mosh_effect_motion_threshold_y=mosh_effect_motion_threshold,
        frame_blend_weight=frame_blend_weight,
        interpolation_method=interpolation_method,
        border_mode=border_mode)

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
        description=('Uses Optical Flow to reproduce a motion repeat effect.'))

    parser.add_argument(
        '-i', '--input_video_path',
        required=True,
        type=str,
        help='Input video path')

    parser.add_argument(
        '-o', '--output_video_path',
        required=True,
        type=str,
        help='Output video file path')

    parser.add_argument(
        '-s', '--start_frame',
        metavar='<frame>',
        type=int,
        default=0,
        help='Output ideo start frame number. Default: 0 (first frame)')

    parser.add_argument(
        '-e', '--end_frame',
        metavar='<frame>',
        type=int,
        default=None,
        help='Output video end frame number. Default: last frame.')

    parser.add_argument(
        '-repeat_start', '--repeat_effect_start_frame',
        metavar='<frame>',
        type=int,
        default=0,
        help='Repeat effect start frame number. Default: 0 (first frame)')

    parser.add_argument(
        '-repeat_end', '--repeat_effect_end_frame',
        metavar='<frame>',
        type=int,
        default=None,
        help='Repeat effect end frame number. Default: last frame.')

    parser.add_argument(
        '-repeat_m_threshold', '--repeat_effect_motion_threshold',
        metavar='<pixel_threshold>',
        type=float,
        default=None,
        help=('Motion pixel threshold for the repeat effect.')
    )

    parser.add_argument(
        '-repeat_m_mult_x', '--repeat_effect_motion_multiplier_x',
        metavar='<multiplier>',
        type=float,
        default=1.0,
        help=('Motion multiplier for the repeat effect along the x axis.')
    )

    parser.add_argument(
        '-repeat_m_mult_y', '--repeat_effect_motion_multiplier_y',
        metavar='<multiplier>',
        type=float,
        default=1.0,
        help=('Motion multiplier for the repeat effect along the y axis.')
    )

    parser.add_argument(
        '-repeat_m_mult', '--repeat_effect_motion_multiplier',
        metavar='<multiplier>',
        type=float,
        default=1.0,
        help=('Global motion multiplier for the repeat effect.')
    )

    parser.add_argument(
        '-mosh_start', '--mosh_effect_start_frame',
        metavar='<frame>',
        type=int,
        default=0,
        help='Mosh effect start frame number. Default: 0 (first frame)')

    parser.add_argument(
        '-mosh_end', '--mosh_effect_end_frame',
        metavar='<frame>',
        type=int,
        default=None,
        help='Mosh effect end frame number. Default: last frame.')

    parser.add_argument(
        '-mosh_m_threshold', '--mosh_effect_motion_threshold',
        metavar='<pixel_threshold>',
        type=float,
        default=None,
        help=('Motion pixel threshold for the mosh effect.')
    )

    parser.add_argument(
        '-mosh_m_mult_x', '--mosh_effect_motion_multiplier_x',
        metavar='<multiplier>',
        type=float,
        default=1.0,
        help=('Motion multiplier for the mosh effect along the x axis.')
    )

    parser.add_argument(
        '-mosh_m_mult_y', '--mosh_effect_motion_multiplier_y',
        metavar='<multiplier>',
        type=float,
        default=1.0,
        help=('Motion multiplier for the mosh effect along the y axis.')
    )

    parser.add_argument(
        '-mosh_m_mult', '--mosh_effect_motion_multiplier',
        metavar='<multiplier>',
        type=float,
        default=1.0,
        help=('Global motion multiplier for the mosh effect.')
    )

    parser.add_argument(
        '--frame_blend_weight',
        metavar='<0.0-1.0>',
        type=float,
        default=0.0,
        help=('Blend weight for the transformed frame in the final image. ' +
              'Used to preserve detail from the frames throuhougt the video. ' +
              'If > 0 the transformed frames will be blended with the result ' +
              'images, otherwise only the first frame\'s pixels will be used.'))

    parser.add_argument(
        '-m_mult', '--motion_multiplier',
        metavar='<multiplier>',
        type=float,
        default=1.0,
        help='Multiplier to apply to the motion vectors.'
    )

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
        args.input_video_path,
        args.output_video_path,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        # Repeat effect args.
        repeat_effect_start_frame=args.repeat_effect_start_frame,
        repeat_effect_end_frame=args.repeat_effect_end_frame,
        repeat_effect_motion_threshold=args.repeat_effect_motion_threshold,
        repeat_effect_motion_multiplier_x=(
            args.repeat_effect_motion_multiplier_x *
            args.repeat_effect_motion_multiplier),
        repeat_effect_motion_multiplier_y=(
            args.repeat_effect_motion_multiplier_y *
            args.repeat_effect_motion_multiplier),
        # Mosh effect args.
        mosh_effect_start_frame=args.mosh_effect_start_frame,
        mosh_effect_end_frame=args.mosh_effect_end_frame,
        mosh_effect_motion_threshold=args.mosh_effect_motion_threshold,
        mosh_effect_motion_multiplier_x=(
            args.mosh_effect_motion_multiplier_x *
            args.mosh_effect_motion_multiplier),
        mosh_effect_motion_multiplier_y=(
            args.mosh_effect_motion_multiplier_y *
            args.mosh_effect_motion_multiplier),

        frame_blend_weight=args.frame_blend_weight,

        interpolation_method=args.interpolation_method,
        border_mode=args.border_mode,

        preview=bool(args.preview))
