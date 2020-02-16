import cv2
import numpy

# Default codec for video output, as a 4 character code.
# (see http://www.fourcc.org/codecs.php)
DEFAULT_FOURCC = cv2.VideoWriter_fourcc(*'XVID')


class VideoReader():
    """A convenience wrapper around cv2.VideoCapture."""

    def __init__(self, file_path):
        self.__file_path = file_path

        # Creates a capture to read video information, then releases it.
        video_capture = cv2.VideoCapture(file_path)
        self.__frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.__frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.__frame_heigth = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.__fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        # Format code for the video.
        self.__fourcc = int(video_capture.get(cv2.CAP_PROP_FOURCC))
        video_capture.release()

    @property
    def file_path(self):
        return self.__file_path

    @property
    def frame_count(self):
        return self.__frame_count

    @property
    def frame_width(self):
        return self.__frame_width

    @property
    def frame_height(self):
        return self.__frame_heigth

    @property
    def frame_size(self):
        return (self.frame_width, self.frame_height)

    @property
    def fps(self):
        return self.__fps

    @property
    def fourcc(self):
        return self.__fourcc

    @property
    def length_seconds(self):
        return float(self.__frame_count) / float(self.__fps)

    def ReadFrames(self, start_frame=0, end_frame=None, interactive=False):
        """Returns a generator over the video frames.

        Args:
            - start_frame: int. Frame number to start reading at.
            - end_frame: int. Frame number to end reading at. None=last frame.
            - interactive: bool. Set to true to enable frame visualization.

        Yields:
            A tuple of (frame, position).
            frame is a numpy array containing RGB information for each pixel.
            position is a tuple of (position_frame, position_millisecond),
            indicating the read position in frame count and milliseconds.
        """
        video_capture = cv2.VideoCapture(self.file_path)

        while video_capture.isOpened():
            retrieved, frame = video_capture.read()
            position_frames = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            position_milliseconds = video_capture.get(cv2.CAP_PROP_POS_MSEC)

            if position_frames < start_frame:
                continue

            if end_frame is not None and position_frames > end_frame:
                return

            if frame is not None:
                yield (frame, (position_frames, position_milliseconds))
            else:
                video_capture.release()
                return

            if interactive:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return

    def ReadFrameAt(self, frame_position):
        if frame_position < 0:
            raise Exception('Invalid frame position: Must be > 0')
        if frame_position > self.frame_count:
            raise Exception(
                'Invalid frame position: Max value = %d' %
                self.frame_count)

        video_capture = cv2.VideoCapture(self.file_path)
        for _ in range(frame_position):
            video_capture.read()
        _, frame = video_capture.read()
        video_capture.release()

        return frame


class VideoWriter():
    """A convenience wrapper around cv2.VideoWriter."""

    def __init__(self, file_path, frame_size, fps, fourcc=DEFAULT_FOURCC):
        self.__file_path = file_path
        self.__frame_size = frame_size
        self.__fps = fps
        self.__fourcc = fourcc

    @property
    def file_path(self):
        return self.__file_path

    @property
    def frame_size(self):
        return self.__frame_size

    @property
    def frame_width(self):
        return self.frame_size[0]

    @property
    def frame_height(self):
        return self.frame_size[1]

    @property
    def fps(self):
        return self.__fps

    @property
    def fourcc(self):
        return self.__fourcc

    @classmethod
    def FromReader(cls, video_reader, file_path):
        """Creates a VideoWriter with the same properties (size, fps, etc.) as
        the passed VideoReader."""
        return VideoWriter(
            file_path, frame_size=video_reader.frame_size, fps=video_reader.fps,
            fourcc=DEFAULT_FOURCC)

    def WriteFrame(self, frame):
        pass

    def WriteFrames(self, frames):
        video_writer = cv2.VideoWriter(
            filename=self.file_path, fourcc=self.fourcc, fps=self.fps,
            frameSize=self.frame_size)

        for frame, position in frames:
            video_writer.write(frame)
            yield frame, position

        video_writer.release()
