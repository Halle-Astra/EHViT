import numpy as np
import cv2
import pyrealsense2 as rs

class StereoMatcher:
    def __init__(self):
        self.window_size = 5
        self.min_disp = 0
        self.num_disp = 112 - self.min_disp
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=self.min_disp,
            numDisparities=self.num_disp,
            blockSize=self.window_size,
            P1=8 * 3 * self.window_size ** 2,
            P2=32 * 3 * self.window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    def compute(self, img_left, img_right):
        # 将左右图像转换为灰度图像
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # 进行立体匹配
        disparity = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

        return disparity


if __name__ == '__main__':
    # 初始化Realsense相机
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    # 创建立体匹配器对象
    matcher = StereoMatcher()

    # 循环读取图像
    while True:
        # 读取彩色和深度图像
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 对左右图像进行裁剪，以保证它们的大小一致
        width, height = color_image.shape[:2]
        width_half = int(width/2)

        img_left = color_image[0:480, 0:640]
        img_right = color_image[0:480, 200:840]

        # 进行立体匹配
        disparity = matcher.compute(img_left, img_right)

        # 将视差图像可视化
        disparity_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disparity_color = cv2.applyColorMap(disparity_visual, cv2.COLORMAP_JET)

        # 显示图像
        cv2.imshow('Disparity', disparity_color)
        cv2.imshow('Left Image', img_left)
        cv2.imshow('Right Image', img_right)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    pipeline.stop()

