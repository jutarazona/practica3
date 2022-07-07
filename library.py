# import the necessary packages
import imutils
import numpy as np
from imutils import paths
import argparse
import cv2


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def detect_blur(image, threshold):
    img = image.copy()
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        return True
    fm = variance_of_laplacian(gray)
    text = "Not Blurry"
    # if the focus measure is less than the supplied threshold,
    # then the image should be considered "blurry"
    if fm < threshold:
        text = "Blurry"
        cv2.putText(img, "{}: {:.2f}".format(text, fm), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.imshow("Image", img)
        return True
    # show the image
    cv2.putText(img, "{}: {:.2f}".format(text, fm), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imshow("Image", img)
    return False


def perspective_correction(image):
    kernel = np.full((3, 3), -1 / 9)
    kernel[2, 2] = 8 / 9

    img = cv2.GaussianBlur(image.copy(), (9, 9), 1)
    '''
    _, edge = cv2.threshold(img, 150, 255, cv2.THRESH_OTSU)
    edge = cv2.dilate(edge, kernel, iterations=2)
    edge = cv2.erode(edge, kernel, iterations=2)

    edge = cv2.erode(edge, kernel, iterations=2)
    edge = cv2.dilate(edge, kernel, iterations=2)
    '''
    # edge = cv2.Canny(img, 75, 150)
    # edge = cv2.Canny(img, 30, 125)
    # edge = auto_canny(img)
    _, edge = cv2.threshold(img, 50, 150, cv2.THRESH_OTSU)
    edge = auto_canny(edge)
    cv2.imshow("image from autocany", edge)

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        # approximate the contourç
        epsilon = 0.01 * cv2.arcLength(c, True)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if 3 < len(approx) < 7 and cv2.contourArea(c) > 1000:
            cv2.drawContours(image, [c], -1, (255, 0, 0), thickness=5)
            cv2.imshow("image", image)
            cv2.waitKey()
            # if our approximated contour has four points, then we
        # can assume that we have found our screen


def auto_canny(image, sigma=0.7):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    # return the edged image
    return cv2.Canny(image, lower, upper)


def auto_canny2(image):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    sigma = 0
    while sigma < 1.1:
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
        text = "sigma: " + str(sigma)
        cv2.imshow(text, edged)
        sigma += 0.1
        cv2.waitKey()
    # return the edged image
    return edged
