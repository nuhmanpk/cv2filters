from typing import Tuple
import cv2
import numpy as np
import os


class Filters:
    @staticmethod
    def read_image(file_path: str) -> np.ndarray:
        """
        Read an image from the specified file path.

        Args:
            file_path (str): The full file path to the image file.

        Returns:
            np.ndarray: The loaded image as a NumPy array.

        Raises:
            ValueError: If the image cannot be read or the file does not exist.
        """
        try:
            absolute_path = os.path.abspath(file_path)
            image = cv2.imread(absolute_path)
            if image is None:
                raise ValueError(f"""Failed to read image from {file_path}. 
                                     Please check if the file exists and is a valid image file.""")
            return image
        except cv2.error as e:
            raise ValueError(f"OpenCV error: {str(e)}")
        except FileNotFoundError as e:
            raise ValueError(f"File not found: {str(e)}")

    @staticmethod
    def increase_brightness(image: np.ndarray, value: int = 10) -> np.ndarray:
        """
        Increase the brightness of an image.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            value (int, optional): The value to increase the brightness by. Default is 10.

        Returns:
            np.ndarray: The brightened image as a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input 'image' must be a NumPy array.")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Input 'image' must have three dimensions (height, width, channels).")

        try:
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_image[..., 2] = np.clip(hsv_image[..., 2] + value, 0, 255)
            return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        except cv2.error as e:
            raise ValueError(f"OpenCV error: {str(e)}")

    @staticmethod
    def blur(image: np.ndarray, kernal_size: tuple = (5, 5)) -> np.ndarray:
        """
        """
        if not isinstance(kernal_size, tuple) or len(kernal_size) != 2:
            raise ValueError('Kernal size must me a tuple of two intergers.')
        if kernal_size[0] % 2 == 0 or kernal_size[1] % 2 == 0:
            raise ValueError('Kernal Dimensions must be odd numbers.')
        try:
            return cv2.GaussianBlur(image, kernal_size, 0)
        except cv2.error as e:
            raise ValueError(f'OpenCV Error {str(e)}')

    @staticmethod
    def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate an image by the specified angle.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            angle (float): The rotation angle in degrees.

        Returns:
            np.ndarray: The rotated image as a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input 'image' must be a NumPy array.")

        try:
            rows, cols = image.shape[:2]
            center = (cols // 2, rows // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(
                image, rotation_matrix, (cols, rows))
            return rotated_image
        except cv2.error as e:
            raise ValueError(f"OpenCV error: {str(e)}")

    @staticmethod
    def flip_image(image: np.ndarray, flip_code: int) -> np.ndarray:
        """
        Flip an image horizontally or vertically.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            flip_code (int): Flip code for the flip operation (0 for vertical flip, 1 for horizontal flip).

        Returns:
            np.ndarray: The flipped image as a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input 'image' must be a NumPy array.")

        try:
            flipped_image = cv2.flip(image, flip_code)
            return flipped_image
        except cv2.error as e:
            raise ValueError(f"OpenCV error: {str(e)}")

    @staticmethod
    def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Crop a rectangular region from an image.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            x (int): The x-coordinate of the top-left corner of the region.
            y (int): The y-coordinate of the top-left corner of the region.
            width (int): The width of the region.
            height (int): The height of the region.

        Returns:
            np.ndarray: The cropped image as a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input 'image' must be a NumPy array.")

        try:
            cropped_image = image[y:y+height, x:x+width]
            return cropped_image
        except Exception as e:
            raise ValueError(f"Error cropping image: {str(e)}")

    @staticmethod
    def resize_image(image: np.ndarray, width: int = 250, height: int = 250) -> np.ndarray:
        """
        Resize an image to the specified dimensions.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            width (int): The target width of the image.
            height (int): The target height of the image.

        Returns:
            np.ndarray: The resized image as a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input 'image' must be a NumPy array.")

        try:
            resized_image = cv2.resize(image, (width, height))
            return resized_image
        except cv2.error as e:
            raise ValueError(f"OpenCV error: {str(e)}")

    @staticmethod
    def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Convert an image to grayscale.

        Args:
            image (np.ndarray): The input image as a NumPy array.

        Returns:
            np.ndarray: The grayscale image as a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input 'image' must be a NumPy array.")

        try:
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return grayscale_image
        except cv2.error as e:
            raise ValueError(f"OpenCV error: {str(e)}")

    @staticmethod
    def detect_edges(image: np.ndarray, threshold1: float = 100, threshold2: float = 200) -> np.ndarray:
        """
        Detect edges in an image using the Canny edge detection algorithm.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            threshold1 (float): The lower threshold for edge detection.
            threshold2 (float): The higher threshold for edge detection.

        Returns:
            np.ndarray: The image with detected edges as a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input 'image' must be a NumPy array.")

        try:
            edges_image = cv2.Canny(image, threshold1, threshold2)
            return edges_image
        except cv2.error as e:
            raise ValueError(f"OpenCV error: {str(e)}")

    @staticmethod
    def sobel_filter(image: np.ndarray, dx: int = 1, dy: int = 1, ksize: int = 3) -> np.ndarray:
        """
        Apply the Sobel filter to an image for edge detection.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            dx (int): The order of the derivative in the x-direction.
            dy (int): The order of the derivative in the y-direction.
            ksize (int): The size of the Sobel kernel.

        Returns:
            np.ndarray: The image with edges detected using the Sobel filter as a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input 'image' must be a NumPy array.")

        try:
            sobel_image = cv2.Sobel(image, cv2.CV_64F, dx, dy, ksize)
            return sobel_image
        except cv2.error as e:
            raise ValueError(f"OpenCV error: {str(e)}")

    @staticmethod
    def bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """
        Apply a bilateral filter to an image for noise reduction.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            d (int): Diameter of each pixel neighborhood.
            sigma_color (float): Standard deviation in the color space.
            sigma_space (float): Standard deviation in the coordinate space.

        Returns:
            np.ndarray: The filtered image as a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input 'image' must be a NumPy array.")

        try:
            filtered_image = cv2.bilateralFilter(
                image, d, sigma_color, sigma_space)
            return filtered_image
        except cv2.error as e:
            raise ValueError(f"OpenCV error: {str(e)}")

    @staticmethod
    def erosion(image: np.ndarray, kernel: np.ndarray = None, iterations: int = 1) -> np.ndarray:
        """
        Apply erosion to an image.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            kernel (np.ndarray): The erosion kernel.
            iterations (int): Number of times erosion is applied.

        Returns:
            np.ndarray: The eroded image as a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input 'image' must be a NumPy array.")

        if kernel is None:
            kernel = np.ones((3, 3), np.uint8)

        try:
            eroded_image = cv2.erode(image, kernel, iterations=iterations)
            return eroded_image
        except cv2.error as e:
            raise ValueError(f"OpenCV error: {str(e)}")

    @staticmethod
    def dilation(image: np.ndarray, kernel: np.ndarray = None, iterations: int = 1) -> np.ndarray:
        """
        Apply dilation to an image.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            kernel (np.ndarray): The dilation kernel.
            iterations (int): Number of times dilation is applied.

        Returns:
            np.ndarray: The dilated image as a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input 'image' must be a NumPy array.")

        if kernel is None:
            kernel = np.ones((3, 3), np.uint8)

        try:
            dilated_image = cv2.dilate(image, kernel, iterations=iterations)
            return dilated_image
        except cv2.error as e:
            raise ValueError(f"OpenCV error: {str(e)}")

    @staticmethod
    def perspective_transform(image: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
        """
        Apply a perspective transformation to an image.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            src_points (np.ndarray): Source points for the perspective transformation.
            dst_points (np.ndarray): Destination points for the perspective transformation.

        Returns:
            np.ndarray: The transformed image as a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input 'image' must be a NumPy array.")

        try:
            perspective_matrix = cv2.getPerspectiveTransform(
                src_points, dst_points)
            transformed_image = cv2.warpPerspective(
                image, perspective_matrix, (image.shape[1], image.shape[0]))
            return transformed_image
        except cv2.error as e:
            raise ValueError(f"OpenCV error: {str(e)}")

    @staticmethod
    def morphological_opening(image: np.ndarray, kernel: np.ndarray = None, iterations: int = 2) -> np.ndarray:
        """
        Apply morphological opening to an image to remove noise and small objects.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            kernel (np.ndarray): The morphological kernel.
            iterations (int): Number of times morphological opening is applied.

        Returns:
            np.ndarray: The processed image as a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input 'image' must be a NumPy array.")

        if kernel is None:
            kernel = np.ones((3, 3), np.uint8)

        try:
            opened_image = cv2.morphologyEx(
                image, cv2.MORPH_OPEN, kernel, iterations=iterations)
            return opened_image
        except cv2.error as e:
            raise ValueError(f"OpenCV error: {str(e)}")

    @staticmethod
    def closing(image: np.ndarray, kernel: np.ndarray = None, iterations: int = 2) -> np.ndarray:
        """
        Apply morphological closing to an image to close small holes.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            kernel (np.ndarray): The closing kernel.
            iterations (int): Number of times closing is applied.

        Returns:
            np.ndarray: The processed image as a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input 'image' must be a NumPy array.")

        if kernel is None:
            kernel = np.ones((3, 3), np.uint8)

        try:
            closed_image = cv2.morphologyEx(
                image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            return closed_image
        except cv2.error as e:
            raise ValueError(f"OpenCV error: {str(e)}")

    @staticmethod
    def highlight_box(image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Highlight a box region in an image by turning the rest of the image to black.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            box (Tuple[int, int, int, int]): The coordinates (x, y, width, height) of the box region.

        Returns:
            np.ndarray: The processed image as a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input 'image' must be a NumPy array.")

        x, y, w, h = box

        try:
            # Create a mask with the same shape as the input image
            mask = np.zeros_like(image)

            # Set the box region to white in the mask
            cv2.rectangle(mask, (x, y), (x+w, y+h),
                          (255, 255, 255), thickness=cv2.FILLED)

            # Apply the mask to the input image
            result = cv2.bitwise_and(image, mask)

            return result
        except cv2.error as e:
            raise ValueError(f"OpenCV error: {str(e)}")

    @staticmethod
    def image_segmentation(image: np.ndarray, method: str = 'simple_thresholding', threshold: int = 127) -> np.ndarray:
        """
        Perform image segmentation on the input image.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            method (str): The segmentation method to use ('simple_thresholding', 'adaptive_thresholding').
            threshold (int): The threshold value for simple thresholding.

        Returns:
            np.ndarray: The segmented image as a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input 'image' must be a NumPy array.")

        if method == 'simple_thresholding':
            _, segmented_image = cv2.threshold(
                image, threshold, 255, cv2.THRESH_BINARY)
        elif method == 'adaptive_thresholding':
            segmented_image = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        else:
            raise ValueError(
                "Invalid segmentation method. Supported methods are 'simple_thresholding' and 'adaptive_thresholding'.")

        return segmented_image