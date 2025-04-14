import cv2
import numpy as np

pid_state = {
    "previous_error": 0.0,
    "integral": 0.0,
}

Kp = 0.015
Ki = 0.001
Kd = 0.005

MAX_THROTTLE = 0.5
MIN_THROTTLE = 0.2
THROTTLE_SCALE = 1.0

RECOVERY_MODE = False
RECOVERY_COUNT = 0
RECOVERY_MAX = 20

def find_lane_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gauss = cv2.GaussianBlur(gray, (11, 11), 0)

    avg_brightness = np.mean(gray)
    thresh_low = max(120, avg_brightness * 0.6)
    thresh_high = min(220, avg_brightness * 1.2)
    
    img_canny = cv2.Canny(img_gauss, thresh_low, thresh_high)
    
    kernel = np.ones((3, 3), np.uint8)
    img_canny = cv2.dilate(img_canny, kernel, iterations=1)
    
    return img_canny

def birdview_transform(img):
    IMAGE_H = 480
    IMAGE_W = 640
    src = np.float32([
        [0, IMAGE_H],
        [IMAGE_W, IMAGE_H],
        [IMAGE_W * 0.2, IMAGE_H * 0.3],
        [IMAGE_W * 0.8, IMAGE_H * 0.3]
    ])
    
    dst = np.float32([
        [IMAGE_W * 0.15, IMAGE_H],
        [IMAGE_W * 0.85, IMAGE_H],
        [IMAGE_W * 0.15, 0],
        [IMAGE_W * 0.85, 0]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H))
    return warped_img

def find_left_right_points(image, draw=None):
    im_height, im_width = image.shape[:2]
    
    scan_lines = [int(im_height * p) for p in [0.9, 0.85, 0.8, 0.75, 0.7, 0.65]]
    center = im_width // 2
    lane_width = 100
    
    left_points = []
    right_points = []
    
    for y in scan_lines:
        interested_line = image[y, :]
        for x in range(center, 0, -1):
            if interested_line[x] > 0:
                left_points.append((x, y))
                break
                
        for x in range(center + 1, im_width):
            if interested_line[x] > 0:
                right_points.append((x, y))
                break
    
    left_point = -1
    right_point = -1
    
    if left_points:
        left_point = left_points[0][0]
    if right_points:
        right_point = right_points[0][0]
    if left_point != -1 and right_point == -1:
        right_point = left_point + lane_width
    if right_point != -1 and left_point == -1:
        left_point = right_point - lane_width
    if left_point == -1 and right_point == -1:
        global RECOVERY_MODE
        RECOVERY_MODE = True
    else:
        RECOVERY_MODE = False
    
    if draw is not None:
        for y in scan_lines:
            cv2.line(draw, (0, y), (im_width, y), (0, 0, 255), 1)
            
        if left_point != -1:
            cv2.circle(draw, (left_point, scan_lines[0]), 7, (255, 255, 0), -1)
            
        if right_point != -1:
            cv2.circle(draw, (right_point, scan_lines[0]), 7, (0, 255, 0), -1)
            
        if left_points and right_points:
            left_lane_points = np.array(left_points, dtype=np.int32)
            right_lane_points = np.array(right_points, dtype=np.int32)
            
            cv2.polylines(draw, [left_lane_points], False, (255, 255, 0), 2)
            cv2.polylines(draw, [right_lane_points], False, (0, 255, 0), 2)
            
            center_lane_points = []
            for i in range(min(len(left_points), len(right_points))):
                lx, ly = left_points[i]
                rx, ry = right_points[i]
                cx = (lx + rx) // 2
                cy = ly
                center_lane_points.append((cx, cy))
            if center_lane_points:
                center_lane = np.array(center_lane_points, dtype=np.int32)
                cv2.polylines(draw, [center_lane], False, (0, 0, 255), 2)

    return left_point, right_point

def calculate_control_signal(img, draw=None):
    global pid_state, Kp, Ki, Kd, MAX_THROTTLE, MIN_THROTTLE, THROTTLE_SCALE
    global RECOVERY_MODE, RECOVERY_COUNT

    img_lines = find_lane_lines(img)
    img_birdview = birdview_transform(img_lines)
    
    if draw is not None:
        draw[:, :] = birdview_transform(draw)
        
    left_point, right_point = find_left_right_points(img_birdview, draw=draw)

    throttle = MAX_THROTTLE
    steering_angle = 0
    im_center = img.shape[1] // 2

    if RECOVERY_MODE:
        throttle = -0.5
        steering_angle = 1.0 if RECOVERY_COUNT % 40 < 20 else -1.0
        RECOVERY_COUNT += 1
        if RECOVERY_COUNT > RECOVERY_MAX:
            RECOVERY_COUNT = 0
    elif left_point != -1 and right_point != -1:
        center_point = (right_point + left_point) // 2
        error = im_center - center_point

        P = Kp * error
        pid_state["integral"] += error
        I = Ki * pid_state["integral"]
        derivative = error - pid_state["previous_error"]
        D = Kd * derivative
        pid_state["previous_error"] = error

        steering_angle = -(P + I + D)
        steering_angle = max(min(steering_angle, 1.0), -1.0)

        abs_steering = abs(steering_angle)
        throttle_reduction = THROTTLE_SCALE * abs_steering
        throttle = MAX_THROTTLE * (1 - throttle_reduction)
        throttle = max(throttle, MIN_THROTTLE)

        RECOVERY_COUNT = 0

    return throttle, steering_angle