import unittest
import cv2
import os

from cv2filters.video_filters import VideoFilters
from cv2filters.image_filters import Filters

script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, 'src', 'production_id_4629779(2160p).mp4')

filters = VideoFilters()
imgFilters = Filters()

class TestVideoFilters(unittest.TestCase):
    def setUp(self):
        self.filters = VideoFilters()
    def test_preview_webcam(self):
        self.filters.apply_haarcascade_eye(video_path,save_video=True)
    # def test_anything(self):
    #     self.filters.apply_haarcascade_eye(video_path)

if __name__ == '__main__':
    unittest.main()
