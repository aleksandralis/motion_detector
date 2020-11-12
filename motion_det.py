# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np

# TODO LIST
""" 
1. Remove argparse, it only occludes clear view
2. Add __main__ for running things like initialization and release of resources
3. Put main processing loop inside function
4. Imutils seems unnecessary - it can be easily replaced with pure cv2
5. Stitch all screens together, side by side - it'd make your work much easier
6. Do not process every frame, you can take a snapshot every 0.5s or even 1s
7. You can replace many of opencv ops with pure numpy ops (imho they are cleaner and more intuitive to use), e.g.
		frameDelta = cv2.absdiff(res1, gray) -> frameDelta = np.abs(res1 - gray)
8. Add meaningful names and be consistent w.r.t syntax - dont mix first_frame with firstFrame
9. accumulatedWeighted can be easily replaced with simple average of circular buffer - it would be easier to control
	number of elements in it and so the overall time span of lookup window
"""

# construct the argument parser and parse the arguments
# writer = cv2.VideoWriter("output.avi",
# cv2.VideoWriter_fourcc(*"MJPG"), 30,(375,500))

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=2000, help="minimum area size")
args = vars(ap.parse_args())
# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
# otherwise, we are reading from a video file
else:
	vs = cv2.VideoCapture(args["video"])
# initialize the first frame in the video stream
firstFrame = None
avg1 = None
draw_frame = None
# loop over the frames of the video
writer = cv2.VideoWriter('outp2.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 10, (500, 375))

while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	frame = vs.read()
	frame = frame if args.get("video", None) is None else frame[1]
	text = "Unoccupied"
	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if frame is None:
		break
	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.imshow("input gray", gray)


	gray = cv2.GaussianBlur(gray, (21, 21), 0)
	cv2.imshow("input blurred", gray)
	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		avg1 = np.float32(firstFrame)
		cv2.accumulateWeighted(gray, avg1, 0.06)
		continue

	cv2.accumulateWeighted(gray, avg1, 0.06)

	res1 = cv2.convertScaleAbs(avg1)

	# compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(res1, gray)

	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, (50,50), iterations=4)
	cv2.imshow(" dill", thresh)
	print(thresh.shape)
	print(thresh.dtype, thresh.min(), thresh.max())
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < args["min_area"]:
			continue
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "Occupied"


	# draw the text and timestamp on the frame
	draw_frame = np.copy(frame)
	cv2.putText(draw_frame, "Room Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(draw_frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		(10, draw_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
	# show the frame and record if the user presses a key
	frameDelta = np.stack([frameDelta, frameDelta, frameDelta], axis=-1)
	writer.write(frameDelta)

	cv2.imshow("Security Feed", draw_frame)
	cv2.imshow("Thresh", thresh)
	# print(frameDelta.shape)
	# print(frameDelta.dtype, frameDelta.min(), frameDelta.max())

	cv2.imshow("Frame Delta", frameDelta)

	key = cv2.waitKey(1) & 0xFF
	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break
# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
writer.release()