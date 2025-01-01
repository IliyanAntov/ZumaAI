import numpy as np
import cv2 as cv
from PIL.Image import Image
from numpy import ndarray


class ImageProcessing:
    @staticmethod
    def get_masked_image(image):
        # Convert the image to HSV color space
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        lower_boundry1 = np.array([0, 120, 120])
        upper_boundry1 = np.array([24, 255, 255])

        lower_boundry2 = np.array([26, 120, 120])
        upper_boundry2 = np.array([200, 255, 255])

        mask1 = cv.inRange(hsv, lower_boundry1, upper_boundry1)
        mask2 = cv.inRange(hsv, lower_boundry2, upper_boundry2)

        image_filtered = cv.bitwise_and(image, image, mask=mask1 | mask2)
        return image_filtered

    @staticmethod
    def detect_circles(image, new_width=None, new_height=None):
        # Convert the original image to grayscale
        image_grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image_grayscale = cv.medianBlur(image_grayscale, 5)

        # Detect circles using Hough Circle Transform
        circles = cv.HoughCircles(image_grayscale, cv.HOUGH_GRADIENT, 1, 25,
                                  param1=230, param2=10, minRadius=10, maxRadius=20)

        original_height, original_width = image.shape[:2]
        width_scale = 1.0
        height_scale = 1.0

        # Resize the image if needed for drawing circles
        if new_width and new_height:
            image = cv.resize(image, (new_width, new_height))
            # Calculate the scaling factors
            width_scale = new_width / original_width
            height_scale = new_height / original_height

        if circles is not None:
            circles = np.uint16(np.around(circles))  # Round values and convert to integers

            # Create a black image of the resized image for drawing circles
            image_black = np.zeros_like(image)

            # Draw the detected circles on the resized black image
            for circle in circles[0, :]:
                # Original circle center and radius
                original_center = (circle[0], circle[1])
                original_radius = circle[2]

                # Scale the center and radius to the new resolution
                center = (int(original_center[0] * width_scale), int(original_center[1] * height_scale))
                radius = int(original_radius * min(width_scale, height_scale))  # Scale radius using the smaller factor

                # Create a mask for the circle
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv.circle(mask, center, radius, (255), thickness=-1)  # Fill the circle in the mask

                # Mask the image to isolate the region inside the circle
                masked_image = cv.bitwise_and(image, image, mask=mask)

                # Calculate the average color within the circle
                avg_color = cv.mean(masked_image, mask=mask)[:3] # Exclude alpha (mean of BGR channels)

                # Get the nearest predefined color
                nearest_color, color_name = ImageProcessing.get_nearest_color(np.array(avg_color))

                # Draw the circle and fill it with the nearest color on the resized black image
                cv.circle(image_black, center, 2, nearest_color.tolist(), thickness=-1)  # Filled circle

            # Return the resized image with circles drawn on it
            return image_black
        else:
            image_black = np.zeros_like(image)
            return image_black

    @staticmethod
    def get_nearest_color(avg_color):
        # Predefined BGR colors for yellow, red, green, blue
        colors = {
            "Yellow": np.array([0, 255, 255]),  # Yellow (BGR)
            "Red": np.array([0, 0, 255]),  # Red (BGR)
            "Green": np.array([0, 255, 0]),  # Green (BGR)
            "Blue": np.array([255, 0, 0])  # Blue (BGR)
        }

        # Calculate the Euclidean distance between the average color and each predefined color
        distances = {name: np.linalg.norm(avg_color - color) for name, color in colors.items()}

        # Get the nearest color by selecting the one with the minimum distance
        nearest_color_name = min(distances, key=distances.get)
        return colors[nearest_color_name], nearest_color_name

    @staticmethod
    def prepare_image(image) -> ndarray:
        np_arr = np.array(image)
        rgb = cv.cvtColor(np_arr, cv.COLOR_BGRA2RGB)
        image_masked = ImageProcessing.get_masked_image(rgb)
        image_filtered = ImageProcessing.detect_circles(image_masked, 112, 80)
        # cv.imshow("image_filtered", image_filtered)
        # img = cv.cvtColor(image_filtered, cv.COLOR_RGB2BGR)
        # cv.imwrite("test.jpg", image_filtered)
        # cv.waitKey(0)
        img_rgb = cv.cvtColor(image_filtered, cv.COLOR_RGB2BGR)
        return img_rgb


