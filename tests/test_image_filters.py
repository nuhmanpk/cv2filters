import unittest
import cv2
import os

from cv2filters.image_filters import ImageFilters

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'src', 'albert-dera-ILip77SbmOE-unsplash.jpg')
image = cv2.imread(image_path)

class TestImageFilters(unittest.TestCase):
    def setUp(self):
        self.filters = ImageFilters()
    def test_filters(self):
        
        #Apply filter Here
        filtered_image = self.filters.crop_face(image)
        
        scale_percent = 12  # 10% of the original size


        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)

        # Resize the images
        original_resized = cv2.resize(image, (width, height))
        filtered_image = cv2.resize(filtered_image, (width, height))

        cv2.imshow("Original Image", original_resized)
        cv2.imshow("Filtered Image", filtered_image)

        # Wait for a key press and close the windows
        cv2.waitKey(10000)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    unittest.main()
