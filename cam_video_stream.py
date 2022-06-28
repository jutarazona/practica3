# import the necessary packages
import datetime

import cv2
from threading import Thread


class CamVideoStream:

    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        frameWidth = 1280
        frameHeight = 720
        self.stream = cv2.VideoCapture(src)
        self.stream.set(3, frameWidth)
        self.stream.set(4, frameHeight)
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
