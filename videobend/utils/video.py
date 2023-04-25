import os
import queue
import threading
import time

import numpy
import cv2

from ..utils.frame import Frame

VIDEO_READER_DEFAULT_QUEUE_SIZE = 128
# Default codec for video writer, as a 4 character code.
# (see http://www.fourcc.org/codecs.php)
VIDEO_WRITER_DEFAULT_FOURCC = int(cv2.VideoWriter_fourcc(*'mp4v'))
VIDEO_WRITER_DEFAULT_API_PREFERENCE = int(cv2.CAP_FFMPEG)


class VideoStream():
    """A class that provides a stream of frames over a video file.

    Uses threading to ensure all frames are captured on time and increase read
    performances.
    """

    def __init__(self, file_path, queue_size=VIDEO_READER_DEFAULT_QUEUE_SIZE):
        """Constructor.

        Args:
            file_path: A string. Input video file path.
            queue_size: An int. Max size for the internal frames queue.
        """
        self.__file_path = file_path
        self.__queue_size = queue_size

        self.__video_capture = cv2.VideoCapture(file_path)

        self.__capture_queue = queue.Queue(maxsize=queue_size)
        self.__capture_stopped = False
        self.__capture_thread = threading.Thread(target=self.__Update, args=())
        self.__capture_thread.daemon = True

    @property
    def has_more_frames(self):
        """Returns true if there are more frames in the stream.

        We add a retry mechanism in case frames have not yet been added to the
        queue.
        """
        num_tries = 0
        max_num_tries = 5

        while (self.__capture_queue.qsize() == 0 and
               not self.__capture_stopped and
               num_tries < max_num_tries):
            time.sleep(.1)
            num_tries += 1

        return self.__capture_queue.qsize() > 0

    @property
    def is_running(self):
        return self.has_more_frames or not self.__capture_stopped

    def Start(self):
        """Starts the capture thread."""
        self.__capture_thread.start()

    def Stop(self):
        """Stops the capture thread."""
        self.__capture_stopped = True
        self.__capture_thread.join()

    def ReadFrame(self):
        """Reads the next frame in the queue."""
        return self.__capture_queue.get()

    def ReadFrames(self, start_frame=0, end_frame=None):
        """Returns a generator over captured frames."""
        if end_frame is None:
            end_frame = self.__video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

        # Starts the capture thread and yields frames from the capture queue.
        self.Start()
        while self.has_more_frames:
            frame = self.ReadFrame()
            # Skips frames before and after specified start and end frames.
            if frame.position[0] < start_frame or frame.position[0] > end_frame:
                continue
            yield frame
            cv2.waitKey(1)

    def __Update(self):
        """Capture thread loop. Retrieves frames and puts them in the queue.
        """
        while True:
            if self.__capture_stopped:
                break

            if not self.__capture_queue.full():
                retrieved, frame_pixels = self.__video_capture.read()

                # Retrieved is a flag provided by VideoCapture that indicates
                # if the frame was correctly retrieved (i.e with no errors).
                # TODO(davyrisso): Handle errors instead of just stopping.
                if not retrieved:
                    self.__capture_stopped = True

                # Reads frame position information from the internal
                # VideoCapture object.
                frame_position_frames = self.__video_capture.get(
                    cv2.CAP_PROP_POS_FRAMES)
                frame_position_milliseconds = self.__video_capture.get(
                    cv2.CAP_PROP_POS_MSEC)

                # Adds the frame in the queue if it is not empty:
                # VideoCapture returns an empty frame after the capture has
                # finished.
                if frame_pixels is not None:
                    self.__capture_queue.put(
                        Frame(
                            frame_pixels,
                            (frame_position_frames,
                             frame_position_milliseconds)))
                else:
                    self.__capture_stopped = True


class VideoReader():
    """A class that provides VideoStreams over a video or image sequence."""

    def __init__(
            self, file_path, queue_size=VIDEO_READER_DEFAULT_QUEUE_SIZE):
        """Constructor for a VideoReader.

        Args:
          - file_path: string. Path to the video file to read.
                       Note: This can also be an image sequence pattern
                      (e.g: img_%02d.jpg => img_01.jpg, img_02.jpg...)
          - queue_size: int. Size of the frame queue (queue) size.
        """
        self.__file_path = file_path
        self.__queue_size = queue_size
        self.__video_capture = cv2.VideoCapture(file_path)

    @property
    def file_path(self):
        return self.__file_path

    @property
    def frame_count(self):
        return int(self.__video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def frame_size(self):
        return (
            int(self.__video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.__video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    @property
    def fps(self):
        return int(self.__video_capture.get(cv2.CAP_PROP_FPS))

    @property
    def fourcc(self):
        return int(self.__video_capture.get(cv2.CAP_PROP_FOURCC))

    def GetStream(self):
        return VideoStream(self.file_path, self.__queue_size)


class VideoWriter():
    """A convenience wrapper around cv2.VideoWriter."""

    def __init__(
            self, file_path, frame_size, fps,
            fourcc=VIDEO_WRITER_DEFAULT_FOURCC):
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
        the passed VideoReader.

        Args:
            video_reader: A VideoReader object.
            file_path: Output video file for the new VideoWriter object.

        Returns:
            A new VideoWriter object with the same properties as the passed
            VideoReader object.
        """
        return VideoWriter(
            file_path,
            frame_size=video_reader.frame_size,
            fps=video_reader.fps,
            fourcc=VIDEO_WRITER_DEFAULT_FOURCC)

    def WriteFrames(self, frames):
        """Writes the passed frames to an output video file.

        Also returns a generator of written frames for chaining.

        Args:
            frames: An iterable of utils.Frame.

        Yields:
            frame: The written utils.Frames.
        """
        # Creates the output directory if it does not exist.
        directory = os.path.split(self.file_path)[0]
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        video_writer = cv2.VideoWriter(
            filename=self.file_path,
            fourcc=self.fourcc,
            fps=self.fps,
            frameSize=(self.frame_width, self.frame_height),
            apiPreference=VIDEO_WRITER_DEFAULT_API_PREFERENCE)

        for frame in frames:
            video_writer.write(frame.pixels)
            yield frame

        video_writer.release()
