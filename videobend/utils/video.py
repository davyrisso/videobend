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

    def __init__(self, file_path, queue_size=VIDEO_READER_DEFAULT_QUEUE_SIZE):
        self.__file_path = file_path
        self.__queue_size = queue_size
        self.__video_capture = cv2.VideoCapture(file_path)

        self.__capture_queue = queue.Queue(maxsize=queue_size)
        self.__capture_stopped = False
        self.__capture_thread = threading.Thread(target=self.__Update, args=())
        self.__capture_thread.daemon = True

    @property
    def has_more_frames(self):
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
        self.__capture_thread.start()

    def Stop(self):
        self.__capture_stopped = True
        self.__capture_thread.join()

    def ReadFrame(self):
        return self.__capture_queue.get()

    def ReadFrames(self, start_frame=0, end_frame=None):
        if end_frame is None:
            end_frame = self.__video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

        self.Start()
        while self.has_more_frames:
            frame = self.ReadFrame()
            if frame.position[0] < start_frame or frame.position[0] > end_frame:
                continue
            yield frame
            cv2.waitKey(1)

    def __Update(self):
        while True:
            if self.__capture_stopped:
                break

            if not self.__capture_queue.full():
                retrieved, frame_pixels = self.__video_capture.read()

                if not retrieved:
                    self.__capture_stopped = True

                frame_position_frames = self.__video_capture.get(
                    cv2.CAP_PROP_POS_FRAMES)
                frame_position_milliseconds = self.__video_capture.get(
                    cv2.CAP_PROP_POS_MSEC)

                if frame_pixels is not None:
                    self.__capture_queue.put(
                        Frame(
                            frame_pixels,
                            (frame_position_frames,
                             frame_position_milliseconds)))
                else:
                    self.__capture_stopped = True


class VideoReader():

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
        return self.__video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

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
        the passed VideoReader."""
        return VideoWriter(
            file_path,
            frame_size=video_reader.frame_size,
            fps=video_reader.fps,
            fourcc=VIDEO_WRITER_DEFAULT_FOURCC)

    def WriteFrames(self, frames):
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
