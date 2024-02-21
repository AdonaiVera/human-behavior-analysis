import cv2
import numpy as np

class BirdEyeTransformer:
    def __init__(self, src_points, dst_points, output_size):
        self.src_points = np.float32(src_points)
        self.dst_points = np.float32(dst_points)
        self.output_size = output_size
        self.matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

    def transform(self, detections):
        transformed_points = []
        for detection in detections.xyxy:
            # Assuming detection is a bounding box with format [x1, y1, x2, y2]
            center = np.array([(detection[0] + detection[2]) / 2, detection[3]])  # Bottom center point of the box
            transformed_point = cv2.perspectiveTransform(np.array([[center]]), self.matrix)
            transformed_points.append(transformed_point[0][0])
        return transformed_points

    def draw_bird_eye_view(self, transformed_points, background_image_path=None):
        if background_image_path:
            bird_eye_image = cv2.imread(background_image_path)
        else:
            bird_eye_image = np.zeros((self.output_size[1], self.output_size[0], 3), np.uint8)

        for point in transformed_points:
            cv2.circle(bird_eye_image, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

        return bird_eye_image
