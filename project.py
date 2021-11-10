import cv2
import numpy as np
import time
from tkinter import Tk, simpledialog


cam = cv2.VideoCapture(0)
openingKernel = np.ones((5, 5))

lowerLimit = [100, 105, 50]
upperLimit = [120, 255, 255]

lowerLimitYellow = np.array([30, 84, 133])
upperLimitYellow = np.array([91, 255, 216])

root = Tk()

x1 = 0
y1 = 0

x1Yellow = 0
y1Yellow = 0

canvas = None

penImage = cv2.imread("./pen.jpg")
eraserImage = cv2.imread("./eraser.png")
saveImage = cv2.imread("./save.png")
penImage = cv2.resize(penImage, (50, 50))
eraserImage = cv2.resize(eraserImage, (50, 50))
saveImage = cv2.resize(saveImage, (50, 50))

lower = np.array(lowerLimit, dtype="uint8")
upper = np.array(upperLimit, dtype="uint8")

switch = "Pen"

lastSwitch = time.time()
lastSave = time.time()

noiseth = 600
backgroundThreshold = 800

backgroundobject = cv2.createBackgroundSubtractorMOG2(detectShadows=False)


def lowerLimitH(x):
    lower[0] = x


def lowerLimitS(x):
    lower[1] = x


def lowerLimitV(x):
    lower[2] = x


def upperLimitH(x):
    upper[0] = x


def upperLimitS(x):
    upper[1] = x


def upperLimitV(x):
    upper[2] = x


def lowerYellowLimitH(x):
    lowerLimitYellow[0] = x


def lowerYellowLimitS(x):
    lowerLimitYellow[1] = x


def lowerYellowLimitV(x):
    lowerLimitYellow[2] = x


def upperYellowLimitH(x):
    upperLimitYellow[0] = x


def upperYellowLimitS(x):
    upperLimitYellow[1] = x


def upperYellowLimitV(x):
    upperLimitYellow[2] = x


cv2.namedWindow("filter color object")
cv2.createTrackbar('Lower limit H', 'filter color object',
                   lower[0], 180, lowerLimitH)
cv2.createTrackbar('Lower limit S', 'filter color object',
                   lower[1], 255, lowerLimitS)
cv2.createTrackbar('Lower limit V', 'filter color object',
                   lower[2], 255, lowerLimitV)

cv2.createTrackbar('Upper limit H', 'filter color object',
                   upper[0], 180, upperLimitH)
cv2.createTrackbar('Upper limit S', 'filter color object',
                   upper[1], 255, upperLimitS)
cv2.createTrackbar('Upper limit V', 'filter color object',
                   upper[2], 255, upperLimitV)


cv2.namedWindow('yellow color')
cv2.createTrackbar('Lower limit H', 'yellow color',
                   lowerLimitYellow[0], 180, lowerYellowLimitH)
cv2.createTrackbar('Lower limit S', 'yellow color',
                   lowerLimitYellow[1], 255, lowerYellowLimitS)
cv2.createTrackbar('Lower limit V', 'yellow color',
                   lowerLimitYellow[2], 255, lowerYellowLimitV)

cv2.createTrackbar('Upper limit H', 'yellow color',
                   upperLimitYellow[0], 180, upperYellowLimitH)
cv2.createTrackbar('Upper limit S', 'yellow color',
                   upperLimitYellow[1], 255, upperYellowLimitS)
cv2.createTrackbar('Upper limit V', 'yellow color',
                   upperLimitYellow[2], 255, upperYellowLimitV)

while True:
    _, frame = cam.read()
    frame = cv2.resize(frame, (640, 480), fx=0, fy=0,
                       interpolation=cv2.INTER_CUBIC)
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if canvas is None:
        canvas = np.zeros_like(frame)

    #Write or eraser
    top_left = frame[0:50, 0:50]
    fgmask = backgroundobject.apply(top_left)
    mask = cv2.inRange(hsv, lower, upper)

    switch_thresh = np.sum(fgmask == 255)

    if switch_thresh > backgroundThreshold and (time.time() - lastSwitch) > 1:
        print('switch')
        lastSwitch = time.time()
        if switch == "Pen":
            switch = "Eraser"
        else:
            switch = "Pen"

    # save section
    topX1 = frame.shape[1]-50
    topX2 = frame.shape[1]
    frame[0:50, topX1:topX2] = saveImage

    """ maskMorphology = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, openingKernel, iterations=6) """
    mask = cv2.erode(mask, openingKernel, iterations=4)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, openingKernel, iterations=1)

    cv2.imshow("mask", mask)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > noiseth:
        c = max(contours, key=cv2.contourArea)
        x2, y2, w, h = cv2.boundingRect(c)
        if(x1 == 0 and y1 == 0):
            x1, y1 = x2, y2
        else:
            if switch == "Pen":
                canvas = cv2.line(canvas, (x1, y1),
                                  (x2, y2), [255, 255, 0], 5)
            else:
                canvas = cv2.circle(canvas, (x2, y2), 20, (0, 0, 0), -1)

        if x2 > topX1+25 and x2 < topX2 and y2 > 0 and y2 < 50 and (time.time() - lastSave) > 1:
            lastSave = time.time()
            canvasCopy = canvas.copy()
            canvasCopy = cv2.putText(canvasCopy, 'Image save', (0, 200),
                                     cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1, cv2.LINE_AA)
            _, mask = cv2.threshold(cv2.cvtColor(
                canvasCopy, cv2.COLOR_BGR2GRAY), 20,  255, cv2.THRESH_BINARY)
            foreground = cv2.bitwise_and(canvasCopy, canvasCopy, mask=mask)
            background = cv2.bitwise_and(
                frame, frame, mask=cv2.bitwise_not(mask))
            cv2.imshow("filter color object", cv2.add(foreground, background))
            print('Save')
            filename = simpledialog.askstring(
                "Save image", "Enter the name of the image")
            cv2.imwrite('./paints/'+filename+".jpg", canvas)

        x1, y1 = x2, y2
        #cv2.rectangle(frame, (x2, y2), (x2+w, y2+h), (0, 25, 255), 2)
    else:
        x1, y1 = 0, 0

    # Yellow
    maskYellow = cv2.inRange(hsv, lowerLimitYellow, upperLimitYellow)
    cv2.imshow("yellow color", maskYellow)

    maskYellow = cv2.erode(maskYellow, openingKernel, iterations=4)
    maskYellow = cv2.morphologyEx(
        maskYellow, cv2.MORPH_CLOSE, openingKernel, iterations=1)
    contoursYellow, hierarchyYellow = cv2.findContours(
        maskYellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contoursYellow and cv2.contourArea(max(contoursYellow, key=cv2.contourArea)) > noiseth:
        cYellow = max(contoursYellow, key=cv2.contourArea)
        x2Yellow, y2Yellow, wYellow, hYellow = cv2.boundingRect(cYellow)
        if(x1Yellow == 0 and y1Yellow == 0):
            x1Yellow, y1Yellow = x2Yellow, y2Yellow
        else:
            if switch == "Pen":
                canvas = cv2.line(canvas, (x1Yellow, y1Yellow),
                                  (x2Yellow, y2Yellow), [60, 100, 60], 5)
            else:
                canvas = cv2.circle(
                    canvas, (x2Yellow, y2Yellow), 20, (0, 0, 0), -1)
        x1Yellow, y1Yellow = x2Yellow, y2Yellow

    else:
        x1Yellow, y1Yellow = 0, 0

    _, mask = cv2.threshold(cv2.cvtColor(
        canvas, cv2.COLOR_BGR2GRAY), 20,  255, cv2.THRESH_BINARY)
    foreground = cv2.bitwise_and(canvas, canvas, mask=mask)
    background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
    frame = cv2.add(foreground, background)

    if switch != 'Pen':
        cv2.circle(frame, (x1, y1), 20, (255, 255, 255), -1)
        cv2.circle(frame, (x1Yellow, y1Yellow), 20, (255, 255, 255), -1)
        frame[0: 50, 0: 50] = eraserImage
    else:
        frame[0: 50, 0: 50] = penImage

    cv2.imshow("filter color object", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
