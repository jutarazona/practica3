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


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


# function to transform image to four points
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)

    # # multiply the rectangle by the original ratio
    # rect *= ratio

    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = int(max(int(widthA), int(widthB)))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = int(max(int(heightA), int(heightB)))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def perspective_correction(image, array):
    DNI = None
    kernel = np.full((3, 3), -1 / 9)
    kernel[2, 2] = 8 / 9

    img = cv2.GaussianBlur(image.copy(), (9, 9), 1)

    img = cv2.dilate(img, kernel, iterations=2)
    img = cv2.erode(img, kernel, iterations=2)

    img = cv2.erode(img, kernel, iterations=2)
    img = cv2.dilate(img, kernel, iterations=2)

    # cv2.imshow("after stuffs", img)

    # edge = cv2.Canny(img, 75, 150)
    # edge = cv2.Canny(img, 30, 125)
    # edge = auto_canny(img)
    _, edge = cv2.threshold(img.copy(), 0, 255, cv2.THRESH_OTSU)
    # cv2.imshow("image from threshold", edge)
    edge = auto_canny(edge)
    # cv2.imshow("image from autocany", edge)

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        # approximate the contour
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        text = 'Warped DNI'
        if 3 < len(approx) < 5 and cv2.contourArea(c) > 2000:
            x, y, w, h = cv2.boundingRect(c)
            cv2.drawContours(image, [c], -1, (255, 0, 0), thickness=5)
            print(x, y, w, h)

            # cv2.imshow("image", image)
            # if our approximated contour has four points, then we
            # can assume that we have found our screen

            pts = approx.reshape(4, 2)
            rect = order_points(pts)
            print(rect)
            DNI = four_point_transform(array.copy(), pts)

            #cv2.imshow(text, DNI)

    if DNI is None:
        return False

    return DNI


def extraccion_MRZ(image):
    blur = cv2.GaussianBlur(image.copy(), (75, 75), 1)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    # Dilate to combine adjacent text contours
    #cv2.imshow("t", thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    erode = cv2.erode(thresh, kernel, iterations=3)
    dilate = cv2.dilate(erode, kernel, iterations=1)
    #cv2.imshow("dilate", erode)
    #cv2.waitKey()

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(erode, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    ROI = []
    ROInumber = 0
    for c in cnts:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        ratio = float(w / h)
        epsilon = 0.01 * cv2.arcLength(c, True)

        approx = cv2.approxPolyDP(c, epsilon, True)
        text = 'Roi'
        if 3 < len(approx) < 12 and area > 7000:
            # cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)
            cv2.drawContours(image, [c], -1, (255, 0, 0), thickness=5)
            ROI = image[y:y + h, x:x + w]
            ROInumber += 1
            cv2.imshow(text, ROI)
            cv2.waitKey(1000)
            cv2.destroyWindow(text)

    if ROInumber == 1:
        return ROI
    return False


def auto_canny(image, sigma=0.7):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    # return the edged image
    return cv2.Canny(image, lower, upper)


def ocr(img):
    pass