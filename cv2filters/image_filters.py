from typing import Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
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
            #grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3:
                grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                grayscale_image = image
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

    @staticmethod
    def image_negative(image: np.ndarray) -> np.ndarray:
        """
        Compute the negative of an image.

        Args:
            image (np.ndarray): The input image as a NumPy array.

        Returns:
            np.ndarray: The negative image as a NumPy array.
        """
        try:
            negative_image = 255 - image
            return negative_image
        except cv2.error as e:
            raise ValueError(f"OpenCV error: {str(e)}")

    @staticmethod
    def histogram_correction(image: np.ndarray) -> np.ndarray:
        """
        Apply histogram correction to the image.

        Args:
            image (np.ndarray): The input image as a NumPy array.

        Returns:
            np.ndarray: The histogram-corrected image as a NumPy array.
        """
        try:
            # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image
            equalized_image = cv2.equalizeHist(gray_image)
            return equalized_image
        except cv2.error as e:
            raise ValueError(f"OpenCV error: {str(e)}")

    @staticmethod
    def pencil(image: np.ndarray) -> np.ndarray:
        """
        Apply a pencil sketch effect to the image.

        Args:
            image (np.ndarray): The input image as a NumPy array.

        Returns:
            np.ndarray: The image with a pencil sketch effect applied, as a NumPy array.
        """
        try:
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

            kernel_size = 5  
            kernel = np.ones((kernel_size, kernel_size),
                                np.float32) / (kernel_size * kernel_size)
            blurred_image = cv2.filter2D(gray_image, -1, kernel)  


            pencil_image = cv2.divide(gray_image, blurred_image, scale=255)


            if len(image.shape) == 3:
                        pencil_image = cv2.cvtColor(pencil_image, cv2.COLOR_GRAY2BGR)
            
            return pencil_image
        except cv2.error as e:
            raise ValueError(f"OpenCV error: {str(e)}")

    @staticmethod
    def cartoon(image: np.ndarray, resize_width: int = 960, resize_height: int = 540) -> np.ndarray:
        """
            Apply a cartoon effect to an image.
            Fork from https://github.com/sachinkumar95/cartoon-image-filter
            Args:
                image (np.ndarray): The input image as a NumPy array.
                resize_width (int): The width to resize the image to before applying the effect. Default is 960.
                resize_height (int): The height to resize the image to before applying the effect. Default is 540.

            Returns:
                np.ndarray: The image with a cartoon effect applied, as a NumPy array.
        """
        try:
            # https://github.com/sachinkumar95/cartoon-image-filter

            resized1 = cv2.resize(image, (resize_width, resize_height))

            gray_image = cv2.cvtColor(resized1, cv2.COLOR_BGR2GRAY)
            resized2 = cv2.resize(gray_image, (resize_width, resize_height))

            smooth_gray_image = cv2.medianBlur(resized2, 5)
            resized3 = cv2.resize(smooth_gray_image, (resize_width, resize_height))


            edges = cv2.adaptiveThreshold(
                smooth_gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
            )
            resized4 = cv2.resize(edges, (resize_width, resize_height))


            color_image = cv2.bilateralFilter(image, 9, 300, 300)
            resized5 = cv2.resize(color_image, (resize_width, resize_height))

            cartoon_image = cv2.bitwise_and(color_image, color_image, mask=edges)
            resized6 = cv2.resize(cartoon_image, (resize_width, resize_height))

            return resized6
        except cv2.error as e:
            raise ValueError(f"OpenCV error: {str(e)}")

    @staticmethod      
    def painting(image: np.ndarray, resize_width: int = 960, resize_height: int = 540,
                bilateral_d: int = 10, bilateral_sigma_color: int = 20, bilateral_sigma_space: int = 30,
                threshold_value: int = 100, threshold_max_value: int = 255) -> np.ndarray:
        """
        Apply a painting effect to an image.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            resize_width (int): The width to resize the image to before applying the effect. Default is 960.
            resize_height (int): The height to resize the image to before applying the effect. Default is 540.
            bilateral_d (int): Diameter of each pixel neighborhood for the bilateral filter. Default is 10.
            bilateral_sigma_color (int): Standard deviation in the color space for the bilateral filter. Default is 20.
            bilateral_sigma_space (int): Standard deviation in the coordinate space for the bilateral filter. Default is 30.
            threshold_value (int): Threshold value for the adaptive thresholding. Default is 100.
            threshold_max_value (int): Maximum value for the adaptive thresholding. Default is 255.

        Returns:
            np.ndarray: The image with a painting effect applied, as a NumPy array.
        """
        try:
            # Resize image
            resized_image = cv2.resize(image, (resize_width, resize_height))

            # Convert to grayscale
            if len(resized_image.shape) == 3:
                gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = resized_image

            # Apply bilateral filter for smoothing and edge preservation
            filtered_image = cv2.bilateralFilter(gray_image, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)

            # Apply adaptive thresholding to create a binary image
            _, thresholded_image = cv2.threshold(filtered_image, threshold_value, threshold_max_value, cv2.THRESH_BINARY)

            # Convert binary image to RGB
            thresholded_image = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)

            return thresholded_image
        except cv2.error as e:
            raise ValueError(f"OpenCV error: {str(e)}")

    @staticmethod
    def color_distribution(image: np.ndarray) -> np.ndarray:
        """
        Display the color distribution of an image using a histogram.

        Args:
            image (np.ndarray): The input image as a NumPy array.

        Returns:
            np.ndarray: The histogram image as a NumPy array.
        """
        try:
            # Convert image to RGB if it has a single channel
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Split image into separate channels
            b, g, r = cv2.split(image)

            # Calculate color histograms for each channel
            hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
            hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

            # Create figure and axes
            fig, ax = plt.subplots()

            # Plot color histograms for each channel
            ax.plot(hist_b, color='blue', label='Blue')
            ax.plot(hist_g, color='green', label='Green')
            ax.plot(hist_r, color='red', label='Red')

            # Set labels and title
            ax.set_xlabel('Pixel Intensity')
            ax.set_ylabel('Frequency')
            ax.set_title('Color Distribution')

            # Add legend
            ax.legend()

            # Convert the plot to a NumPy array
            fig.canvas.draw()
            histogram_image = np.array(fig.canvas.renderer.buffer_rgba())

            # Close the plot to free up resources
            plt.close(fig)

            return histogram_image

        except Exception as e:
            raise ValueError(f"Error displaying color distribution: {str(e)}")

    @staticmethod
    def histogram(image: np.ndarray) -> np.ndarray:
        """
        Display the histogram of an image using matplotlib.pyplot.

        Args:
            image (np.ndarray): The input image as a NumPy array.

        Returns:
            np.ndarray: The histogram image as a NumPy array.
        """
        try:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

            plt.plot(hist)
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.title('Histogram')

            fig = plt.gcf()
            fig.canvas.draw()
            histogram_image = np.array(fig.canvas.renderer.buffer_rgba())

            plt.close(fig)

            return histogram_image

        except Exception as e:
            raise ValueError(f"Error displaying histogram: {str(e)}")

    @staticmethod
    def crop_face(image: np.ndarray) -> [np.ndarray]:
        """
        Detect frontal faces in the input image.

        Args:
            image (np.ndarray): The input image as a NumPy array.

        Returns:
            Optional[np.ndarray]: The region of interest (ROI) containing the detected face, or None if no face is detected or an error occurs.
        """
        try:
            # Load the pre-trained Haar Cascade for frontal face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # Read the image if it's a file path
            if isinstance(image, str):
                image = cv2.imread(image)

            # Convert the image to grayscale for face detection
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                # Assuming only one face is detected, return the region of interest (ROI) containing the face
                (x, y, w, h) = faces[0]
                face_roi = image[y:y + h, x:x + w]
                return face_roi
            else:
                # No face detected, return None
                return None

        except cv2.error as e:
            print(f"OpenCV error in detect_frontal_face: {str(e)}")
            return None

        except Exception as e:
            print(f"Error in detect_frontal_face: {str(e)}")
            return None

    @staticmethod
    def detect_eyes(image: np.ndarray) -> np.ndarray:
        """
        Detect eyes in the input image and draw rectangles around them.

        Args:
            image (np.ndarray): The input image as a NumPy array.

        Returns:
            np.ndarray: The input image with rectangles drawn around the detected eyes.
        """
        try:
            # Load the pre-trained Haar Cascade for eye detection
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            eyes = eye_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around the eyes
            for (x, y, w, h) in eyes:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            return image
        except cv2.error as e:
            print(f"OpenCV error in detect_frontal_face: {str(e)}")
            return None

        except Exception as e:
            print(f"Error in detect_frontal_face: {str(e)}")
            return None
        
    @staticmethod
    def detect_eyeglasses(image: np.ndarray) -> np.ndarray:
        try:
            # Load the pre-trained Haar Cascade for eye detection
            eye_glass = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            eyes = eye_glass.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around the eyes
            for (x, y, w, h) in eyes:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # handle returning x, y, w, h in future
            return image
        except cv2.error as e:
            print(f"OpenCV error in detect_frontal_face: {str(e)}")
            return None

        except Exception as e:
            print(f"Error in detect_frontal_face: {str(e)}")
            return None
    
    @staticmethod
    def detect_fullbody(image: np.ndarray) -> np.ndarray:
        try:
            fullbody = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


            body = fullbody.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around the body
            for (x, y, w, h) in body:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # handle returning x, y, w, h in future
            return image
        except cv2.error as e:
            print(f"OpenCV error in detect_frontal_face: {str(e)}")
            return None

        except Exception as e:
            print(f"Error in detect_frontal_face: {str(e)}")
            return None
    
    @staticmethod
    def detect_lefteye(image: np.ndarray) -> np.ndarray:
        try:
            lefteye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            eyes = lefteye.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around the eyes
            for (x, y, w, h) in eyes:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # handle returning x, y, w, h in future
            return image
        except cv2.error as e:
            print(f"OpenCV error in detect_frontal_face: {str(e)}")
            return None

        except Exception as e:
            print(f"Error in detect_frontal_face: {str(e)}")
            return None

    @staticmethod
    def detect_righteye(image: np.ndarray) -> np.ndarray:
        try:
            righteye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            eyes = righteye.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around the eyes
            for (x, y, w, h) in eyes:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # handle returning x, y, w, h in future
            return image
        except cv2.error as e:
            print(f"OpenCV error in detect_frontal_face: {str(e)}")
            return None

        except Exception as e:
            print(f"Error in detect_frontal_face: {str(e)}")
            return None

    @staticmethod
    def detect_lowerbody(image: np.ndarray) -> np.ndarray:
        try:
            lowerbody = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lowerbody.xml')

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            body = lowerbody.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around the body
            for (x, y, w, h) in body:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # handle returning x, y, w, h in future
            return image
        except cv2.error as e:
            print(f"OpenCV error in detect_frontal_face: {str(e)}")
            return None

        except Exception as e:
            print(f"Error in detect_frontal_face: {str(e)}")
            return None

    @staticmethod
    def detect_profile_face(image: np.ndarray) -> np.ndarray:
        try:

            profileface = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            profileface = profileface.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around the body
            for (x, y, w, h) in body:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # handle returning x, y, w, h in future
            return image
        except cv2.error as e:
            print(f"OpenCV error in detect_frontal_face: {str(e)}")
            return None

        except Exception as e:
            print(f"Error in detect_frontal_face: {str(e)}")
            return None

    @staticmethod
    def detect_smile(image: np.ndarray) -> np.ndarray:
        try:
            smile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            smile = smile.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around the body
            for (x, y, w, h) in body:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # handle returning x, y, w, h in future
            return image
        except cv2.error as e:
            print(f"OpenCV error in detect_frontal_face: {str(e)}")
            return None

        except Exception as e:
            print(f"Error in detect_frontal_face: {str(e)}")
            return None

    @staticmethod
    def detect_pedestrians(image: np.ndarray) -> np.ndarray:
        try:
            # Load the pre-trained Haar-like cascade classifier for pedestrian detection
            pedestrian_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'hogcascade_pedestrians.xml')

            # Convert the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect pedestrians in the image
            pedestrians = pedestrian_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around the detected pedestrians
            for (x, y, w, h) in pedestrians:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            return image

        except cv2.error as e:
            print(f"OpenCV error in detect_pedestrians: {str(e)}")
            return None

        except Exception as e:
            print(f"Error in detect_pedestrians: {str(e)}")
            return None
