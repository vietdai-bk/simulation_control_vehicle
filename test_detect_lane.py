import cv2
import numpy as np

def find_lane_lines(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, (11, 11), 0)
    img_lane = gauss
    thress_low = 150
    thress_high = 200
    img_lane = cv2.Canny(gauss, threshold1=thress_low, threshold2=thress_high)
    return img_lane

def birdview_transform(img):
    IMAGE_H = 480
    IMAGE_W = 640
    src = np.float32([[0, IMAGE_H], [640, IMAGE_H], [0, IMAGE_H * 0.4], [IMAGE_W, IMAGE_H * 0.4]])
    dst = np.float32([[240, IMAGE_H], [640 - 240, IMAGE_H], [-160, 0], [IMAGE_W+160, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H))
    return warped_img

def left_right_lane_points(img, draw=None):
    img_h, img_w = img.shape[:2]
    lines_y = int(img_h * 0.8)
    if draw is not None:
        cv2.line(draw, (0, lines_y), (img_w, lines_y), (0, 0, 0), 2)
    lines_points = img[lines_y, :]
    left_point = -1
    right_point = -1
    center = img_w // 2
    lane_width = 100
    for x in range(center, 0, -1):
        if lines_points[x] > 0:
            left_point = x
            break
    for x in range(center + 1, img_w, 1):
        if lines_points[x] > 0:
            right_point = x
            break

    if left_point != -1 and right_point == -1:
        right_point = left_point + lane_width

    if right_point != -1 and left_point == -1:
        left_point = right_point - lane_width
    if draw is not None:
        if left_point != -1:
            cv2.circle(draw, (left_point, lines_y), 7, (255, 255, 0), -1)
        if right_point != -1:
            cv2.circle(draw, (right_point, lines_y), 7, (255, 255, 0), -1)
    return draw, left_point, right_point

# PID Controller Class with Smoothed Control
class PID:
    def __init__(self, kp, ki, kd, max_output=1.0, min_output=-1.0):
        self.kp = kp  # hệ số tỉ lệ
        self.ki = ki  # hệ số tích phân
        self.kd = kd  # hệ số đạo hàm
        self.max_output = max_output  # giới hạn đầu ra tối đa
        self.min_output = min_output  # giới hạn đầu ra tối thiểu
        self.prev_error = 0
        self.integral = 0

    def compute(self, error):
        self.integral += error
        derivative = error - self.prev_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        # Giới hạn đầu ra để tránh quá điều khiển
        output = max(self.min_output, min(self.max_output, output))
        return output

def control_vehicle(img, draw, pid_steering=PID(kp=0.4, ki=0.05, kd=0.2), throttle=0.5):
    img_h, img_w = img.shape[:2]
    im_lane = find_lane_lines(img)
    im_bird = birdview_transform(im_lane)
    draw = birdview_transform(draw)
    lane_detect, left_point, right_point = left_right_lane_points(im_bird, draw=draw)
    
    center = img_w // 2  # Vị trí trung tâm ảnh
    if left_point != -1 and right_point != -1:
        # Trung tâm của làn đường
        lane_center = (left_point + right_point) // 2
        error = center - lane_center  # Sai lệch giữa trung tâm làn đường và xe
        
        # Tính toán góc lái (steering angle)
        steering_angle = pid_steering.compute(error)
    else:
        steering_angle = 0  # Nếu không tìm thấy lane, giữ nguyên góc lái

    # Điều chỉnh throttle linh hoạt hơn
    if abs(steering_angle) > 5:
        throttle = 0.4  # Nếu sai lệch lớn, giảm throttle để ổn định
    elif abs(steering_angle) < 2:
        throttle = 0.7  # Nếu sai lệch nhỏ, tăng throttle để giữ tốc độ
    else:
        throttle = 0.5  # Throttle mặc định nếu không quá lệch
        
    return throttle, steering_angle, lane_detect