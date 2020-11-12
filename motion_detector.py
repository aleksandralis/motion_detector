import cv2
import numpy as np

# TODO LIST
""" 
6. Do not process every frame, you can take a snapshot every 0.5s or even 1s
7. You can replace many of opencv ops with pure numpy ops (imho they are cleaner and more intuitive to use), e.g.
		frameDelta = cv2.absdiff(res1, gray) -> frameDelta = np.abs(res1 - gray)
9. accumulatedWeighted can be easily replaced with simple average of circular buffer - it would be easier to control
	number of elements in it and so the overall time span of lookup window
"""


def engrave_prediction(img, is_occupied):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    text = 'occupied' if is_occupied else 'not occupied'
    text_color = GREEN if is_occupied else RED
    cv2.putText(img, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
    return img


def combine_imgs(img_text_pairs):
    WHITE = (255, 255, 255)

    single_h, single_w = img_text_pairs[0][0].shape[0], img_text_pairs[0][0].shape[1]

    imgs = []
    for i, pair in enumerate(img_text_pairs):
        temp_img = np.copy(pair[0])
        # if gray image without channels dimension
        if temp_img.ndim < 3:
            temp_img = np.stack([temp_img] * 3, axis=-1)
        # if gray image with channels dimension
        elif temp_img.shape[-1] < 3:
            temp_img = np.tile(temp_img, axis=-1)

        cv2.putText(temp_img, pair[1], (20, single_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE)
        imgs.append(temp_img)

    output_img = np.concatenate(imgs, axis=1)
    return output_img


def run(video_capture, width, height, blur_kernel, accumulation_weight, low_thresh, high_thresh, dilation_kernel,
        min_area):
    first_run = True

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print('Stream broken')
            return

        frame = cv2.resize(frame, (width, height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        if first_run:
            avg_accumulator = np.copy(gray_blurred).astype(np.float32)
            first_run = False

        cv2.accumulateWeighted(np.float32(gray_blurred), avg_accumulator, accumulation_weight)
        scaled_abs = cv2.convertScaleAbs(avg_accumulator)  # TODO WHAT IT DOES?

        # compute the absolute difference between the current frame and
        frame_delta = cv2.absdiff(scaled_abs, gray_blurred)
        binary_img = cv2.threshold(frame_delta, low_thresh, high_thresh, cv2.THRESH_BINARY)[1]
        dilated_img = cv2.dilate(binary_img, (dilation_kernel, dilation_kernel), iterations=4)

        contours, _ = cv2.findContours(dilated_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)

            # draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        frame = engrave_prediction(frame, is_occupied=len(contours) > 0)
        output_img = combine_imgs([
            (frame, 'out'),
            (frame_delta, 'frame delta'),
            (dilated_img, 'dilated'),
        ])
        cv2.imshow("debug", output_img)
        cv2.waitKey(30)


if __name__ == '__main__':
    # PARAMS
    MIN_AREA = 2000
    WIDTH = 500
    HEIGHT = 375
    BLUR_KERNEL = 21
    ACCUMULATION_WEIGHT = 0.06
    LOW_THRESH = 25
    HIGH_THRESH = 255
    DILATION_KERNEL = 50

    # initialization
    video_capture = cv2.VideoCapture(0)  # 0 for default camera

    # main loop
    run(video_capture, WIDTH, HEIGHT, BLUR_KERNEL, ACCUMULATION_WEIGHT, LOW_THRESH, HIGH_THRESH,
        DILATION_KERNEL, MIN_AREA)

    # release resources
    cv2.destroyAllWindows()
    video_capture.release()
