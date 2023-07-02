# CV2Filters

CV2Filters a powerful Python package designed as a wrapper around OpenCV, the popular open-source computer vision library. cv2Filters simplifies image processing tasks by providing a higher-level abstraction of the underlying OpenCV functionality. This package aims to make image processing more accessible to both beginner and advanced users, enabling them to efficiently perform a wide range of image manipulation and analysis tasks

CV2Filters empowers users to harness the power of OpenCV in a simplified and intuitive manner. By abstracting away the complexities, the package enables a broader audience to explore image processing, drive innovation, and unlock new possibilities in the field of computer vision.

[![Downloads](https://static.pepy.tech/personalized-badge/cv2filters?period=total&units=international_system&left_color=grey&right_color=yellow&left_text=Total-Downloads)](https://pepy.tech/project/cv2filters)
![PyPI - Format](https://img.shields.io/pypi/format/cv2filters)
[![GitHub license](https://img.shields.io/github/license/nuhmanpk/cv2filters.svg)](https://github.com/nuhmanpk/cv2filters/blob/main/LICENSE)
[![Upload Python Package](https://github.com/nuhmanpk/cv2filters/actions/workflows/publish.yml/badge.svg)](https://github.com/nuhmanpk/cv2filters/actions/workflows/publish.yml)
[![Supported Versions](https://img.shields.io/pypi/pyversions/cv2filters.svg)](https://pypi.org/project/cv2filters)
![PyPI](https://img.shields.io/pypi/v/cv2filters)
![PyPI - Downloads](https://img.shields.io/pypi/dm/cv2filters)
[![Downloads](https://static.pepy.tech/personalized-badge/cv2filters?period=week&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads/Week)](https://pepy.tech/project/cv2filters)

## Installation

[Refer Example](https://github.com/nuhmanpk/cv2filters/wiki) 

To install the library, you can use pip:

```shell
pip install cv2filters
```

## Usage

First, import the `Filters` class from the library:

```python
from cv2filters import Filters
# Create an instance of the Filters class
filters = Filters()
```

Now you can use the various filtering functions available in the library. For example, let's show how to use the increase_brightness and apply_blur functions:


Create an instance of the Filters class and then call the desired methods on the instance. For example:

```python
from cv2filters import Filters
# Create an instance of the Filters class
filters = Filters()

# Read an image
image = filters.read_image('path_to_image.jpg') # Provide exact path here eg: /usr/home/desktop/image.jpg

# Increase brightness of the image
brightened_image = filters.increase_brightness(image,value=10)

# Blur the image
blurred_image = filters.blur(image,kernal_size=(5,5))

# Rotate the image
rotated_image = filters.rotate_image(image, angle=45)

# Flip the image horizontally
flipped_image = filters.flip_image(image, flip_code=1)

# Crop a region from the image
cropped_image = filters.crop_image(image, x=100, y=100, width=200, height=200)

# Resize the image
resized_image = filters.resize_image(image, width=500, height=500)

# Convert the image to grayscale
grayscale_image = filters.convert_to_grayscale(image)

# Detect edges in the image
edges_image = filters.detect_edges(image,threshold1=100, threshold2=200)

# Apply the Sobel filter to detect edges
sobel_image = filters.sobel_filter(image,dx=1, dy= 1, ksize=3)

# Apply bilateral filtering to reduce noise
filtered_image = filters.bilateral_filter(image,d=9, sigma_color=75, sigma_space=75)

# Perform erosion on the image
eroded_image = filters.erosion(image,kernel=None, iterations=1)

# Perform dilation on the image
dilated_image = filters.dilation(image,kernel=None, iterations=1)

# Apply perspective transformation to the image
transformed_image = filters.perspective_transform(image, src_points, dst_points)

# Perform morphological opening on the image
opened_image = filters.morphological_opening(image,kernel=None, iterations=2)

# Perform morphological closing on the image
closed_image = filters.closing(image,kernel=None, iterations=2)

# Highlight a box region in the image
highlighted_image = filters.highlight_box(image, (x, y, width, height))
```

* Make sure to replace 'path_to_image.jpg' with the actual path to the image file you want to process, and adjust the method arguments as needed.

## Methods

1. **read_image**(file_path: str) -> np.ndarray:
This method reads an image from the specified file path and returns it as a NumPy array.

2. **increase_brightness**(image: np.ndarray, value: int = 10) -> np.ndarray:
This method increases the brightness of an image by the specified value.

3. **blur**(image: np.ndarray, kernel_size: tuple = (5, 5)) -> np.ndarray:
This method applies a Gaussian blur to the image using the specified kernel size.

4. **rotate_image**(image: np.ndarray, angle: float) -> np.ndarray:
This method rotates an image by the specified angle.

5. **flip_image**(image: np.ndarray, flip_code: int) -> np.ndarray:
This method flips an image horizontally or vertically based on the flip code.

6. **crop_image**(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
This method crops a rectangular region from an image based on the specified coordinates and dimensions.

7. **resize_image**(image: np.ndarray, width: int = 250, height: int = 250) -> np.ndarray:
This method resizes an image to the specified width and height.

8. **convert_to_grayscale**(image: np.ndarray) -> np.ndarray:
This method converts an image to grayscale.

9. **detect_edges**(image: np.ndarray, threshold1: float = 100, threshold2: float = 200) -> np.ndarray:
This method detects edges in an image using the Canny edge detection algorithm.

10. **sobel_filter**(image: np.ndarray, dx: int = 1, dy: int = 1, ksize: int = 3) -> np.ndarray:
This method applies the Sobel filter to an image for edge detection.

11. **bilateral_filter**(image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
This method applies a bilateral filter to an image for noise reduction.

12. **erosion**(image: np.ndarray, kernel: np.ndarray = None, iterations: int = 1) -> np.ndarray:
This method applies erosion to an image using the specified kernel and number of iterations.

13. **dilation**(image: np.ndarray, kernel: np.ndarray = None, iterations: int = 1) -> np.ndarray:
This method applies dilation to an image using the specified kernel and number of iterations.

14. **perspective_transform**(image: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
This method applies a perspective transformation to an image using the specified source and destination points.

15. **morphological_opening**(image: np.ndarray, kernel: np.ndarray = None, iterations: int = 2) -> np.ndarray:
This method applies morphological opening to an image to remove noise and small objects.

16. **closing**(image: np.ndarray, kernel: np.ndarray = None, iterations: int = 2) -> np.ndarray:
This method applies morphological closing to an image to close small holes.

17. **highlight_box**(image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
This method highlights a box region in an image by turning the rest of the image to black.

## Contributing

If you'd like to contribute to this library, please follow these steps:

* Fork the repository.
* Create a new branch.
* Make your changes and test them.
* Submit a pull request.
