import datetime
import random
import cv2

class RandomFrameCapturer:
    def __init__(self, interval_minutes=1, output_dir='captured_frames'):
        self.interval_minutes = interval_minutes
        self.next_capture_time = datetime.datetime.now()
        self.output_dir = output_dir

    def should_capture(self):
        current_time = datetime.datetime.now()
        if current_time >= self.next_capture_time:
            self.next_capture_time = current_time + datetime.timedelta(seconds=random.randint(0, self.interval_minutes * 60))
            return True
        return False

    def capture_frame(self, frame, frame_number):
        filename = f'{self.output_dir}/frame_{frame_number}.jpg'
        cv2.imwrite(filename, frame)
        print(f"Captured frame {frame_number} at {datetime.datetime.now()}")