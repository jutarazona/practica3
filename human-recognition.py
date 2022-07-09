import math
import sys
import time
import cv2
import argparse

import numpy as np

from fps import FPS

from cam_video_stream import CamVideoStream

# source venv/bin/activate

# Set True to enter debug mode
debug_mode = False
# Sending metrics every X frame
frames_to_send_metrics = 10

# Other global variables
start_time = time.perf_counter()

last_people_value = 0

detected_id = []

offset = 25
maxContourn = 1500

faceClassif = cv2.CascadeClassifier('Cascade files-20220628/haarcascade_frontalface_default.xml')


class DNI_detection:
    width = 0
    height = 0
    scale = 0.00392
    conf_threshold = 0.6
    nms_threshold = 0.3
    area_threshold = 600
    outputlayers = []
    net = []
    fps = []
    people2 = 0

    def __init__(self):
        # read pre-trained model and config file
        self.net = cv2.dnn.readNet("yolov3_training_last_project.weights",
                                   "yolov3-test.cfg")
        layer_names = self.net.getLayerNames()
        self.outputlayers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.frame = None
        self.grayScaleFrame = None
        self.fps = FPS().start()

    # function to draw bounding box on the detected object with class name
    def draw_bounding_box(self, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = 'CARD'
        color = (225, 225, 0)

        cv2.rectangle(self.frame, (x, y), (x_plus_w, y_plus_h), color, 2)

        cv2.putText(self.frame, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def detect(self, video_fps):
        people = 0
        if self.frame is not None and self.width == 0:
            self.width = self.frame.shape[1]
            self.height = self.frame.shape[0]

        if self.frame is None:
            return

        if self.fps._numFrames % frames_to_send_metrics == 0:
            blob = cv2.dnn.blobFromImage(self.frame, self.scale, (416, 416), (0, 0, 0), True, crop=False)

            # set input blob for the network
            self.net.setInput(blob)

            # run inference through the network
            # and gather predictions from output layers
            outs = self.net.forward(self.outputlayers)

            # initialization
            class_ids = []
            confidences = []
            boxes = []

            # for each detetion from each output layer
            # get the confidence, class id, bounding box params
            # and ignore weak detections (confidence < 0.5)
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    # print(scores)
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > self.conf_threshold:
                        center_x = int(detection[0] * self.width)
                        center_y = int(detection[1] * self.height)
                        w = int(detection[2] * self.width)
                        h = int(detection[3] * self.height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            # apply non-max suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
            if not indices.__len__() == 0:
                people = indices.size
            # go through the detections remaining
            # after nms and draw bounding box
            if indices.__len__() > 0:
                for i in indices:
                    ii = i
                    i = i[0]
                    box = boxes[i]
                    x = int(box[0])
                    y = int(box[1])
                    w = int(box[2])
                    h = int(box[3])

                    detected_id.append(self.frame.copy()[y - offset:y + h + offset, x - offset:x + w + offset])

                    self.draw_bounding_box(class_ids[i], confidences[i], x, y,
                                           x + w,
                                           y + h)

            self.people2 = people
            # print(people)
            self.fps.stop()
            cv2.putText(self.frame, "FPS:" + str(round(self.fps.fps(), 2)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (125, 125, 0), 2)
            cv2.putText(self.frame, "Detecting ID CARD", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 0), 2)

            # display output image
            cv2.imshow("object detection", self.frame)

    def segmentateCard(self):
        global maxContourn
        from library import detect_blur
        from library import perspective_correction, extraccion_MRZ, ocr, text_detection
        if detected_id.__len__() > 0:
            array = detected_id.pop()
            cv2.normalize(array, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            if not detect_blur(array, 1000):
                try:
                    gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

                except cv2.error as error:
                    return
                DNI = perspective_correction(gray, array)
                if DNI is False:
                    return

                faces = faceClassif.detectMultiScale(DNI,
                                                     scaleFactor=1.1,
                                                     minNeighbors=5,
                                                     minSize=(30, 30),
                                                     maxSize=(200, 200))
                text = 'Faces'
                for (x, y, w, h) in faces:
                    try:
                        cv2.destroyWindow(text)
                    except:
                        pass
                    DNI_copy = DNI.copy()
                    cv2.rectangle(DNI_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.imshow(text, DNI_copy)

                # back dni
                mzr = False
                text = 'MRZ'
                if len(faces) == 0:
                    mzr = extraccion_MRZ(DNI)
                    if mzr is not False:
                        try:
                            cv2.destroyWindow(text)
                        except:
                            pass
                        cv2.imshow(text, mzr)
                        # ocr(mzr)
                        # text_detection(DNI)
                else:
                    text = "Text Detection"
                    text_segmentation = text_detection(DNI)
                    # ocr(DNI)
                    try:
                        cv2.destroyWindow(text)
                    except:
                        pass
                    cv2.imshow(text, text_segmentation)

    def detectByCamera(self):
        self.fps.update()
        cvs = CamVideoStream(src=1).start()
        # loop over some frames
        while True:
            self.frame = cvs.read()
            self.detect(False)
            self.fps.update()
            self.segmentateCard()
            key = cv2.waitKey(20)
            if key == 27:
                break

        cv2.destroyAllWindows()
        cvs.stop()

    def humanDetector(self, args):
        print('[INFO] Opening Web Cam.')
        self.detectByCamera()


def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="path to Video File ")
    arg_parse.add_argument("-c", "--camera", default=True, help="Set true if you want to use the camera.")
    return vars(arg_parse.parse_args())


if __name__ == "__main__":
    args = argsParser()
    DNI = DNI_detection()
    DNI.humanDetector(args)
