import cv2
from typing import Any, Callable, Tuple
import time
from .image_filters import ImageFilters

class VideoFilters:
    @staticmethod
    def preview_webcam(filter_function: Callable[[Any], Any] = None, display_stats: bool = False, save:bool = False, output_filename : str = "output_video.mp4") -> None:
        """
        Preview the live video feed from the webcam with an optional specified filter applied.

        Args:
            filter_function (Callable, optional): A function that takes a frame as input and returns the filtered frame.
                If not provided, the raw camera footage will be displayed.
            display_stats (bool, optional): If True, display video statistics such as FPS and time on the preview window.
                Default is False.
        """
        try:
            # Start video capture
            cap = cv2.VideoCapture(0)


            # Get the frame width and height
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))

            if save:
                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))

            else:
                out = None

            prev_time = time.time()
            fps_counter = 0

            while True:
                # Read a frame from the video capture
                ret, frame = cap.read()

                if not ret:
                    break

                if filter_function:
                    # Apply the provided filter function to the frame
                    filtered_frame = filter_function(frame)
                else:
                    # If no filter function is provided, use the raw frame
                    filtered_frame = frame

                # Display video statistics (FPS and time) on the preview window
                if display_stats:
                    cur_time = time.time()
                    fps_counter += 1
                    if cur_time - prev_time >= 1:
                        fps = fps_counter / (cur_time - prev_time)
                        fps_counter = 0
                        prev_time = cur_time

                        cv2.putText(filtered_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(filtered_frame, time.strftime("%Y-%m-%d %H:%M:%S"), (10, frame_height - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Show the filtered frame in a live preview
                cv2.imshow("Video Preview", filtered_frame)

                if save:
                    # Write the frame to the output video file
                    out.write(filtered_frame)

                # Stop the live preview when 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release video capture and writer objects, and destroy all OpenCV windows
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        except Exception as e:
            raise ValueError(f"Error during video preview: {str(e)}")

    @staticmethod
    def save_video(filter_function: Callable[[Any], Any] = None, output_path: str = "output_video.avi", output_fps: float = 25.0) -> None:
        """
        Save the live video feed from the webcam with an optional specified filter applied.

        Args:
            filter_function (Callable, optional): A function that takes a frame as input and returns the filtered frame.
                If not provided, the raw camera footage will be saved.
            output_path (str, optional): The path to save the output video file. Default is "output_video.avi".
            output_fps (float, optional): The frame rate of the output video. Default is 25.0.
        """
        try:
            # Start video capture
            cap = cv2.VideoCapture(0)

            # Get the video frame size and frame rate
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, output_fps, (frame_width, frame_height))

            while True:
                # Read a frame from the video capture
                ret, frame = cap.read()

                if not ret:
                    break

                if filter_function:
                    # Apply the provided filter function to the frame
                    filtered_frame = filter_function(frame)
                else:
                    # If no filter function is provided, use the raw frame
                    filtered_frame = frame

                # Save the filtered frame to the output video file
                out.write(filtered_frame)

                # Show the filtered frame in a live preview
                cv2.imshow("Video Preview", filtered_frame)

                # Stop the video recording when 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release video capture and writer objects, and destroy all OpenCV windows
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        except Exception as e:
            raise ValueError(f"Error during video save: {str(e)}")

    @staticmethod
    def apply_haarcascade_eye(video: str, filter_function: Any = None, save_video: bool = True, output_path: str = "output_video.mp4", output_fps: float = 25.0,  display_stats: bool = False) -> None:
        """
        Apply Haar Cascade for detecting eyes in the live video feed from the webcam or from a video file.

        Args:
            video (str): The path to the video file. If not provided or invalid, the camera feed will be used.
            filter_function (Any, optional): A function that takes a frame as input and returns the filtered frame.
                If not provided, the raw camera footage will be displayed.
            save_video (bool, optional): If True, save the processed video to the output file. Default is False.
            output_path (str, optional): The path to save the output video file. Default is "output_video.avi".
            output_fps (float, optional): The frame rate of the output video. Default is 25.0.
        """
        # Use the provided filter function or default filter function
        filter_function = filter_function or ImageFilters.detect_eyes

        try:
            if video:
                # Start video capture from the provided video file
                cap = cv2.VideoCapture(video)
            else:
                # Start video capture from the camera feed
                cap = cv2.VideoCapture(0)

            # Get the video frame size and frame rate
            frame_width = int(cap.get(3)/100)
            frame_height = int(cap.get(4)/100)

            # Define the codec and create VideoWriter object if save_video is True
            out = None
            if save_video:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_path, fourcc, output_fps, (frame_width, frame_height))

            prev_time = time.time()
            fps_counter = 0

            while True:
                # Read a frame from the video capture
                ret, frame = cap.read()

                if not ret:
                    break

                # Apply the provided filter function to the frame
                filtered_frame = filter_function(frame)

                # Display video statistics (FPS and time) on the preview window
                if display_stats:
                    cur_time = time.time()
                    fps_counter += 1
                    if cur_time - prev_time >= 1:
                        fps = fps_counter / (cur_time - prev_time)
                        fps_counter = 0
                        prev_time = cur_time

                        cv2.putText(filtered_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(filtered_frame, time.strftime("%Y-%m-%d %H:%M:%S"), (10, frame_height - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Show the filtered frame in a live preview
                cv2.imshow("Video Preview", filtered_frame)

                # Write the frame to the output video file if save_video is True
                if save_video:
                    out.write(filtered_frame)

                # Stop the video processing when 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release video capture and writer objects, and destroy all OpenCV windows
            cap.release()
            if save_video:
                out.release()
            cv2.destroyAllWindows()

        except Exception as e:
            raise ValueError(f"Error during video processing: {str(e)}")


