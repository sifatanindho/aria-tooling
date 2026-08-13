import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Set before importing cv2
import cv2
import numpy as np
import random
import math
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


# Aria ET camera calibration defaults (can be overridden from VRS file)
ARIA_ET_DEFAULTS = {
    'image_size': (320, 240),  # width, height
    'focal_length': (279.0, 279.0),  # fx, fy in pixels
    'principal_point': (159.5, 119.5),  # cx, cy
    'model': 'KannalaBrandtK3'
}

# Standard human iris diameter in mm (used for calibration)
IRIS_DIAMETER_MM = 11.7

# Typical eye-to-camera distance for Aria glasses (mm)
ARIA_EYE_DISTANCE_MM = 20.0


def load_aria_calibration(vrs_path, eye='left'):
    """
    Load ET camera calibration from a VRS file.
    Returns calibration dict for the specified eye camera.
    
    Args:
        vrs_path: Path to VRS file
        eye: 'left' or 'right' eye camera to use
    
    Returns:
        dict with image_size, focal_length, principal_point for the specified eye
    """
    try:
        from projectaria_tools.core import data_provider
        
        provider = data_provider.create_vrs_data_provider(vrs_path)
        device_calib = provider.get_device_calibration()
        et_calib_list = device_calib.get_aria_et_camera_calib()
        
        calibration_all = {}
        for et_calib in et_calib_list:
            label = et_calib.get_label()
            img_size = et_calib.get_image_size()
            focal = et_calib.get_focal_lengths()
            principal = et_calib.get_principal_point()
            
            calibration_all[label] = {
                'label': label,
                'image_size': (int(img_size[0]), int(img_size[1])),
                'focal_length': (float(focal[0]), float(focal[1])),
                'principal_point': (float(principal[0]), float(principal[1])),
            }
        
        # Find the requested eye camera
        eye_lower = eye.lower()
        for label, calib in calibration_all.items():
            if eye_lower in label.lower():
                return calib
        
        # If not found by name, return first one
        if calibration_all:
            first_key = list(calibration_all.keys())[0]
            print(f"Warning: '{eye}' eye not found, using {first_key}")
            return calibration_all[first_key]
        
        return None
    except Exception as e:
        print(f"Warning: Could not load calibration from VRS: {e}")
        return None


def calculate_mm_per_pixel_from_calibration(focal_length_px, eye_distance_mm=ARIA_EYE_DISTANCE_MM):
    """
    Calculate mm_per_pixel using camera calibration.
    
    The relationship is: real_size = (pixel_size * distance) / focal_length
    So: mm_per_pixel = distance_mm / focal_length_px
    
    This gives us the mm per pixel at the eye distance.
    """
    avg_focal = (focal_length_px[0] + focal_length_px[1]) / 2.0
    mm_per_pixel = eye_distance_mm / avg_focal
    return mm_per_pixel


# Crop the image to maintain a specific aspect ratio (width:height) before resizing. 
def crop_to_aspect_ratio(image, width=640, height=480):
    
    # Calculate current aspect ratio
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height

    if current_ratio > desired_ratio:
        # Current image is too wide
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset+new_width]
    else:
        # Current image is too tall
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset+new_height, :]

    return cv2.resize(cropped_img, (width, height))

#apply thresholding to an image
def apply_binary_threshold(image, darkestPixelValue, addedThreshold):
    # Calculate the threshold as the sum of the two input values
    threshold = darkestPixelValue + addedThreshold
    # Apply the binary threshold
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    
    return thresholded_image

#Finds a square area of dark pixels in the image
#@param I input image (converted to grayscale during search process)
#@return a point within the pupil region
def get_darkest_area(image, min_pixel_threshold=5):
    """Find the darkest area in the image that's likely to be a pupil.
    
    Args:
        image: BGR image
        min_pixel_threshold: Ignore pixels darker than this (to skip black borders)
    """
    ignoreBounds = 20 #don't search the boundaries of the image for ignoreBounds pixels
    imageSkipSize = 10 #only check the darkness of a block for every Nth x and y pixel (sparse sampling)
    searchArea = 20 #the size of the block to search
    internalSkipSize = 5 #skip every Nth x and y pixel in the local search area (sparse sampling)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a mask to ignore black border regions (common in Aria ET images)
    # Pixels below min_pixel_threshold are likely border/padding, not actual eye data
    valid_mask = gray > min_pixel_threshold

    min_sum = float('inf')
    darkest_point = None
    
    # Also track the image center as fallback
    center_y, center_x = gray.shape[0] // 2, gray.shape[1] // 2

    # Loop over the image with spacing defined by imageSkipSize, ignoring the boundaries
    for y in range(ignoreBounds, gray.shape[0] - ignoreBounds, imageSkipSize):
        for x in range(ignoreBounds, gray.shape[1] - ignoreBounds, imageSkipSize):
            # Calculate sum of pixel values in the search area, skipping pixels based on internalSkipSize
            current_sum = np.int64(0)
            num_pixels = 0
            num_valid = 0  # Count pixels that are not black border
            
            for dy in range(0, searchArea, internalSkipSize):
                if y + dy >= gray.shape[0]:
                    break
                for dx in range(0, searchArea, internalSkipSize):
                    if x + dx >= gray.shape[1]:
                        break
                    pixel_val = gray[y + dy][x + dx]
                    # Only count pixels that are above the threshold (not black border)
                    if pixel_val > min_pixel_threshold:
                        current_sum += pixel_val
                        num_valid += 1
                    num_pixels += 1

            # Update the darkest point if:
            # 1. Current block is darker
            # 2. At least half the pixels in the block are valid (not black border)
            if num_valid > num_pixels // 2 and num_valid > 0:
                avg_brightness = current_sum / num_valid
                if avg_brightness < min_sum:
                    min_sum = avg_brightness
                    darkest_point = (x + searchArea // 2, y + searchArea // 2)  # Center of the block
    
    # Fallback to image center if no valid dark point found
    if darkest_point is None:
        darkest_point = (center_x, center_y)

    return darkest_point

#mask all pixels outside a square defined by center and size
def mask_outside_square(image, center, size):
    x, y = center
    half_size = size // 2

    # Create a mask initialized to black
    mask = np.zeros_like(image)

    # Calculate the top-left corner of the square
    top_left_x = max(0, x - half_size)
    top_left_y = max(0, y - half_size)

    # Calculate the bottom-right corner of the square
    bottom_right_x = min(image.shape[1], x + half_size)
    bottom_right_y = min(image.shape[0], y + half_size)

    # Set the square area in the mask to white
    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image
   
def optimize_contours_by_angle(contours, image):
    if len(contours) < 1:
        return contours

    # Holds the candidate points
    all_contours = np.concatenate(contours[0], axis=0)

    # Set spacing based on size of contours
    spacing = int(len(all_contours)/25)  # Spacing between sampled points

    # Temporary array for result
    filtered_points = []
    
    # Calculate centroid of the original contours
    centroid = np.mean(all_contours, axis=0)
    
    # Create an image of the same size as the original image
    point_image = image.copy()
    
    skip = 0
    
    # Loop through each point in the all_contours array
    for i in range(0, len(all_contours), 1):
    
        # Get three points: current point, previous point, and next point
        current_point = all_contours[i]
        prev_point = all_contours[i - spacing] if i - spacing >= 0 else all_contours[-spacing]
        next_point = all_contours[i + spacing] if i + spacing < len(all_contours) else all_contours[spacing]
        
        # Calculate vectors between points
        vec1 = prev_point - current_point
        vec2 = next_point - current_point
        
        with np.errstate(invalid='ignore'):
            # Calculate angles between vectors
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

        
        # Calculate vector from current point to centroid
        vec_to_centroid = centroid - current_point
        
        # Check if angle is oriented towards centroid
        # Calculate the cosine of the desired angle threshold (e.g., 80 degrees)
        cos_threshold = np.cos(np.radians(60))  # Convert angle to radians
        
        if np.dot(vec_to_centroid, (vec1+vec2)/2) >= cos_threshold:
            filtered_points.append(current_point)
    
    return np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))

#returns the largest contour that is not extremely long or tall
#contours is the list of contours, pixel_thresh is the max pixels to filter, and ratio_thresh is the max ratio
def filter_contours_by_area_and_return_largest(contours, pixel_thresh, ratio_thresh):
    max_area = 0
    largest_contour = None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= pixel_thresh:
            x, y, w, h = cv2.boundingRect(contour)
            length = max(w, h)
            width = min(w, h)

            # Calculate the length-to-width ratio and width-to-length ratio
            length_to_width_ratio = length / width
            width_to_length_ratio = width / length

            # Pick the higher of the two ratios
            current_ratio = max(length_to_width_ratio, width_to_length_ratio)

            # Check if highest ratio is within the acceptable threshold
            if current_ratio <= ratio_thresh:
                # Update the largest contour if the current one is bigger
                if area > max_area:
                    max_area = area
                    largest_contour = contour

    # Return a list with only the largest contour, or an empty list if no contour was found
    if largest_contour is not None:
        return [largest_contour]
    else:
        return []

#Fits an ellipse to the optimized contours and draws it on the image.
def fit_and_draw_ellipses(image, optimized_contours, color):
    if len(optimized_contours) >= 5:
        # Ensure the data is in the correct shape (n, 1, 2) for cv2.fitEllipse
        contour = np.array(optimized_contours, dtype=np.int32).reshape((-1, 1, 2))

        # Fit ellipse
        ellipse = cv2.fitEllipse(contour)

        # Draw the ellipse
        cv2.ellipse(image, ellipse, color, 2)  # Draw with green color and thickness of 2

        return image
    else:
        print("Not enough points to fit an ellipse.")
        return image

#checks how many pixels in the contour fall under a slightly thickened ellipse
#also returns that number of pixels divided by the total pixels on the contour border
#assists with checking ellipse goodness    
def check_contour_pixels(contour, image_shape, debug_mode_on):
    # Check if the contour can be used to fit an ellipse (requires at least 5 points)
    if len(contour) < 5:
        return [0, 0]  # Not enough points to fit an ellipse
    
    # Create an empty mask for the contour
    contour_mask = np.zeros(image_shape, dtype=np.uint8)
    # Draw the contour on the mask, filling it
    cv2.drawContours(contour_mask, [contour], -1, (255), 1)
   
    # Fit an ellipse to the contour and create a mask for the ellipse
    ellipse_mask_thick = np.zeros(image_shape, dtype=np.uint8)
    ellipse_mask_thin = np.zeros(image_shape, dtype=np.uint8)
    ellipse = cv2.fitEllipse(contour)
    
    # Draw the ellipse with a specific thickness
    cv2.ellipse(ellipse_mask_thick, ellipse, (255), 10) #capture more for absolute
    cv2.ellipse(ellipse_mask_thin, ellipse, (255), 4) #capture fewer for ratio

    # Calculate the overlap of the contour mask and the thickened ellipse mask
    overlap_thick = cv2.bitwise_and(contour_mask, ellipse_mask_thick)
    overlap_thin = cv2.bitwise_and(contour_mask, ellipse_mask_thin)
    
    # Count the number of non-zero (white) pixels in the overlap
    absolute_pixel_total_thick = np.sum(overlap_thick > 0)#compute with thicker border
    absolute_pixel_total_thin = np.sum(overlap_thin > 0)#compute with thicker border
    
    # Compute the ratio of pixels under the ellipse to the total pixels on the contour border
    total_border_pixels = np.sum(contour_mask > 0)
    
    ratio_under_ellipse = absolute_pixel_total_thin / total_border_pixels if total_border_pixels > 0 else 0
    
    return [absolute_pixel_total_thick, ratio_under_ellipse, overlap_thin]

#outside of this method, select the ellipse with the highest percentage of pixels under the ellipse 
#TODO for efficiency, work with downscaled or cropped images
def check_ellipse_goodness(binary_image, contour, debug_mode_on):
    ellipse_goodness = [0,0,0] #covered pixels, edge straightness stdev, skewedness   
    # Check if the contour can be used to fit an ellipse (requires at least 5 points)
    if len(contour) < 5:
        print("length of contour was 0")
        return 0  # Not enough points to fit an ellipse
    
    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(contour)
    
    # Create a mask with the same dimensions as the binary image, initialized to zero (black)
    mask = np.zeros_like(binary_image)
    
    # Draw the ellipse on the mask with white color (255)
    cv2.ellipse(mask, ellipse, (255), -1)
    
    # Calculate the number of pixels within the ellipse
    ellipse_area = np.sum(mask == 255)
    
    # Calculate the number of white pixels within the ellipse
    covered_pixels = np.sum((binary_image == 255) & (mask == 255))
    
    # Calculate the percentage of covered white pixels within the ellipse
    if ellipse_area == 0:
        print("area was 0")
        return ellipse_goodness  # Avoid division by zero if the ellipse area is somehow zero
    
    #percentage of covered pixels to number of pixels under area
    ellipse_goodness[0] = covered_pixels / ellipse_area
    
    #skew of the ellipse (less skewed is better?) - may not need this
    axes_lengths = ellipse[1]  # This is a tuple (minor_axis_length, major_axis_length)
    major_axis_length = axes_lengths[1]
    minor_axis_length = axes_lengths[0]
    ellipse_goodness[2] = min(ellipse[1][1]/ellipse[1][0], ellipse[1][0]/ellipse[1][1])
    
    return ellipse_goodness


def detect_iris(gray_frame, pupil_center, pupil_diameter):
    """
    Detect the iris boundary using edge detection and Hough circles.
    Uses the pupil center as a reference point.
    Returns the iris diameter in pixels, or None if not detected.
    
    Human iris is typically 11-12mm in diameter (average ~11.7mm).
    """
    if pupil_center is None or pupil_center[0] == 0:
        return None
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    # Apply edge detection
    edges = cv2.Canny(blurred, 30, 100)
    
    # Expected iris diameter is typically 2-4x the pupil diameter
    min_radius = int(pupil_diameter * 0.8) if pupil_diameter > 0 else 40
    max_radius = int(pupil_diameter * 3.0) if pupil_diameter > 0 else 200
    
    # Use Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        # Find the circle closest to the pupil center
        best_circle = None
        min_dist = float('inf')
        
        for circle in circles[0, :]:
            cx, cy, r = circle
            dist = np.sqrt((cx - pupil_center[0])**2 + (cy - pupil_center[1])**2)
            
            # The iris should be roughly centered on the pupil
            if dist < min_dist and dist < pupil_diameter * 0.5:
                min_dist = dist
                best_circle = circle
        
        if best_circle is not None:
            return float(best_circle[2] * 2)  # Return diameter
    
    return None


def detect_iris_by_gradient(gray_frame, pupil_center, pupil_diameter):
    """
    Alternative iris detection using radial gradient analysis.
    Looks for the limbus (iris-sclera boundary) by analyzing intensity changes.
    """
    if pupil_center is None or pupil_center[0] == 0 or pupil_diameter == 0:
        return None
    
    cx, cy = int(pupil_center[0]), int(pupil_center[1])
    
    # Search radially outward from the pupil center
    min_radius = int(pupil_diameter * 0.6)
    max_radius = int(pupil_diameter * 2.5)
    
    if max_radius >= min(gray_frame.shape) // 2:
        max_radius = min(gray_frame.shape) // 2 - 1
    
    # Sample along multiple radial directions
    num_angles = 36
    radial_profiles = []
    
    for angle_idx in range(num_angles):
        angle = 2 * np.pi * angle_idx / num_angles
        profile = []
        
        for r in range(min_radius, max_radius):
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            
            if 0 <= x < gray_frame.shape[1] and 0 <= y < gray_frame.shape[0]:
                profile.append((r, gray_frame[y, x]))
        
        if len(profile) > 10:
            radial_profiles.append(profile)
    
    if len(radial_profiles) < 10:
        return None
    
    # Find the radius where there's a significant intensity increase (iris->sclera transition)
    edge_radii = []
    
    for profile in radial_profiles:
        radii = [p[0] for p in profile]
        intensities = [p[1] for p in profile]
        
        # Compute gradient
        if len(intensities) > 5:
            gradient = np.diff(intensities)
            # Find the largest positive gradient (dark iris to bright sclera)
            if len(gradient) > 0:
                max_grad_idx = np.argmax(gradient)
                if gradient[max_grad_idx] > 15:  # Threshold for significant edge
                    edge_radii.append(radii[max_grad_idx])
    
    if len(edge_radii) > 5:
        # Use median to be robust to outliers
        iris_radius = np.median(edge_radii)
        return float(iris_radius * 2)  # Return diameter
    
    return None


def process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, debug_mode_on, render_cv_window, draw_on_frame=True):
    """Process thresholded images to find pupil ellipse.
    
    Args:
        draw_on_frame: If True, draw the ellipse annotation directly on `frame`
    """
    final_rotated_rect = ((0,0),(0,0),0)

    image_array = [thresholded_image_relaxed, thresholded_image_medium, thresholded_image_strict] #holds images
    name_array = ["relaxed", "medium", "strict"] #for naming windows
    final_image = image_array[0] #holds return array
    final_contours = [] #holds final contours
    ellipse_reduced_contours = [] #holds an array of the best contour points from the fitting process
    goodness = 0 #goodness value for best ellipse
    best_array = 0 
    kernel_size = 5  # Size of the kernel (5x5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gray_copy1 = gray_frame.copy()
    gray_copy2 = gray_frame.copy()
    gray_copy3 = gray_frame.copy()
    gray_copies = [gray_copy1, gray_copy2, gray_copy3]
    final_goodness = 0
    
    #iterate through binary images and see which fits the ellipse best
    for i in range(1,4):
        # Dilate the binary image
        dilated_image = cv2.dilate(image_array[i-1], kernel, iterations=2)#medium
        
        # Find contours
        contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty image to draw contours
        contour_img2 = np.zeros_like(dilated_image)
        reduced_contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)

        if len(reduced_contours) > 0 and len(reduced_contours[0]) > 5:
            current_goodness = check_ellipse_goodness(dilated_image, reduced_contours[0], debug_mode_on)
            #gray_copy = gray_frame.copy()
            #cv2.drawContours(gray_copies[i-1], reduced_contours, -1, (255), 1)
            ellipse = cv2.fitEllipse(reduced_contours[0])
            if debug_mode_on: #show contours 
                cv2.imshow(name_array[i-1] + " threshold", gray_copies[i-1])
                
            #in total pixels, first element is pixel total, next is ratio
            total_pixels = check_contour_pixels(reduced_contours[0], dilated_image.shape, debug_mode_on)                 
            
            cv2.ellipse(gray_copies[i-1], ellipse, (255, 0, 0), 2)  # Draw with specified color and thickness of 2
            font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
            
            final_goodness = current_goodness[0]*total_pixels[0]*total_pixels[0]*total_pixels[1]
            
            #show intermediary images with text output
            if debug_mode_on:
                cv2.putText(gray_copies[i-1], "%filled:     " + str(current_goodness[0])[:5] + " (percentage of filled contour pixels inside ellipse)", (10,30), font, .55, (255,255,255), 1) #%filled
                cv2.putText(gray_copies[i-1], "abs. pix:   " + str(total_pixels[0]) + " (total pixels under fit ellipse)", (10,50), font, .55, (255,255,255), 1    ) #abs pix
                cv2.putText(gray_copies[i-1], "pix ratio:  " + str(total_pixels[1]) + " (total pix under fit ellipse / contour border pix)", (10,70), font, .55, (255,255,255), 1    ) #abs pix
                cv2.putText(gray_copies[i-1], "final:     " + str(final_goodness) + " (filled*ratio)", (10,90), font, .55, (255,255,255), 1) #skewedness
                cv2.imshow(name_array[i-1] + " threshold", image_array[i-1])
                cv2.imshow(name_array[i-1], gray_copies[i-1])

        if final_goodness > 0 and final_goodness > goodness: 
            goodness = final_goodness
            ellipse_reduced_contours = total_pixels[2]
            best_image = image_array[i-1]
            final_contours = reduced_contours
            final_image = dilated_image
    
    if debug_mode_on:
        cv2.imshow("Reduced contours of best thresholded image", ellipse_reduced_contours)

    test_frame = frame.copy()
    
    final_contours = [optimize_contours_by_angle(final_contours, gray_frame)]
    
    iris_diameter_px = None  # Will store iris diameter for calibration
    
    if final_contours and not isinstance(final_contours[0], list) and len(final_contours[0] > 5):
        #cv2.drawContours(test_frame, final_contours, -1, (255, 255, 255), 1)
        ellipse = cv2.fitEllipse(final_contours[0])
        final_rotated_rect = ellipse
        cv2.ellipse(test_frame, ellipse, (55, 255, 0), 2)
        #cv2.circle(test_frame, darkest_point, 3, (255, 125, 125), -1)
        center_x, center_y = map(int, ellipse[0])
        cv2.circle(test_frame, (center_x, center_y), 3, (255, 255, 0), -1)
        
        # Draw ellipse on the original frame for output video
        if draw_on_frame:
            cv2.ellipse(frame, ellipse, (0, 255, 0), 2)  # Green ellipse
            cv2.circle(frame, (center_x, center_y), 3, (0, 255, 255), -1)  # Yellow center dot
        
        cv2.putText(test_frame, "SPACE = play/pause", (10,410), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,90,30), 2) #space
        cv2.putText(test_frame, "Q      = quit", (10,430), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,90,30), 2) #quit
        cv2.putText(test_frame, "D      = show debug", (10,450), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,90,30), 2) #debug
        
        # Detect iris for calibration
        pupil_diameter = (ellipse[1][0] + ellipse[1][1]) / 2.0
        iris_diameter_px = detect_iris(gray_frame, ellipse[0], pupil_diameter)
        
        # If Hough method fails, try gradient method
        if iris_diameter_px is None:
            iris_diameter_px = detect_iris_by_gradient(gray_frame, ellipse[0], pupil_diameter)

    if render_cv_window:
        cv2.imshow('best_thresholded_image_contours_on_frame', test_frame)
    
    # Create an empty image to draw contours
    contour_img3 = np.zeros_like(image_array[i-1])
    
    if len(final_contours[0]) >= 5:
        contour = np.array(final_contours[0], dtype=np.int32).reshape((-1, 1, 2)) #format for cv2.fitEllipse
        ellipse = cv2.fitEllipse(contour) # Fit ellipse
        cv2.ellipse(gray_frame, ellipse, (255,255,255), 2)  # Draw with white color and thickness of 2

    #process_frames now returns a tuple: (rotated_rect, iris_diameter_px)
    return final_rotated_rect, iris_diameter_px


# Finds the pupil in an individual frame and returns the center point
def process_frame(frame):

    # Crop and resize frame
    frame = crop_to_aspect_ratio(frame)

    #find the darkest point
    darkest_point = get_darkest_area(frame)

    # Convert to grayscale to handle pixel value operations
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
    
    # apply thresholding operations at different levels
    # at least one should give us a good ellipse segment
    thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, 5)#lite
    thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, 250)

    thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, 15)#medium
    thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, 250)
    
    thresholded_image_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, 25)#heavy
    thresholded_image_relaxed = mask_outside_square(thresholded_image_relaxed, darkest_point, 250)
    
    #take the three images thresholded at different levels and process them
    final_rotated_rect, iris_diameter_px = process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, False, False)
    
    return final_rotated_rect, iris_diameter_px

# Loads a video and finds the pupil in each frame
def process_video(video_path, input_method, headless=True, calibration=None, output_dir_override=None, eye_side='left'):
    """Process video to extract pupil metrics.
    
    Args:
        video_path: Path to video file
        input_method: 1 for video file, 2 for webcam
        headless: Run without display
        calibration: Aria camera calibration dict
        output_dir_override: Custom output directory
        eye_side: 'left', 'right', or 'both' - for Aria dual-eye ET images
    """
    import csv
    import json
    from scipy import integrate
    from scipy.stats import linregress
    
    # Determine output path based on input video path
    if video_path:
        output_dir = output_dir_override if output_dir_override else os.path.dirname(video_path)
        if not output_dir:
            output_dir = '.'
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_pupil_output.mp4")
        csv_path = os.path.join(output_dir, f"{base_name}_pupil_metrics.csv")
        summary_path = os.path.join(output_dir, f"{base_name}_pupil_summary.json")
    else:
        output_dir = output_dir_override if output_dir_override else '.'
        output_path = os.path.join(output_dir, 'output_video.mp4')
        csv_path = os.path.join(output_dir, 'pupil_metrics.csv')
        summary_path = os.path.join(output_dir, 'pupil_summary.json')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Use calibration info if provided
    if calibration:
        # Calibration is now a single dict for the selected eye camera
        image_size = calibration['image_size']
        focal_length = calibration['focal_length']
        
        # Calculate calibration-based mm_per_pixel
        calib_mm_per_pixel = calculate_mm_per_pixel_from_calibration(focal_length)
        print(f"\nUsing Aria ET camera calibration ({calibration.get('label', 'unknown')}):")
        print(f"  Image size: {image_size[0]}x{image_size[1]}")
        print(f"  Focal length: {focal_length}")
        print(f"  Calibration-based mm_per_pixel: {calib_mm_per_pixel:.4f}")
        print(f"  (at assumed eye distance of {ARIA_EYE_DISTANCE_MM}mm)")
    else:
        image_size = ARIA_ET_DEFAULTS['image_size']
        focal_length = ARIA_ET_DEFAULTS['focal_length']
        calib_mm_per_pixel = calculate_mm_per_pixel_from_calibration(focal_length)
        print(f"\nUsing default Aria ET camera calibration:")
        print(f"  Image size: {image_size[0]}x{image_size[1]}")
        print(f"  Focal length: {focal_length}")
        print(f"  Calibration-based mm_per_pixel: {calib_mm_per_pixel:.4f}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    # Use native Aria ET resolution for output
    out = cv2.VideoWriter(output_path, fourcc, 30.0, image_size)
    print(f"\nVideo output will be saved to: {output_path}")
    print(f"Pupil metrics will be saved to: {csv_path}")
    print(f"Summary metrics will be saved to: {summary_path}")
    
    # Open CSV file for writing pupil metrics
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'timestamp_ms', 'center_x', 'center_y', 'width_px', 'height_px', 'angle', 
                         'pupil_diameter_px', 'iris_diameter_px', 'mm_per_pixel', 'pupil_diameter_mm'])
    
    # Lists to store data for summary calculations
    timestamps = []
    pupil_diameters_px = []
    pupil_diameters_mm = []
    iris_diameters_px = []
    
    # Running estimate of mm_per_pixel based on iris detections
    mm_per_pixel_estimates = []

    if input_method == 1:
        cap = cv2.VideoCapture(video_path)
    elif input_method == 2:
        cap = cv2.VideoCapture(0)  # Camera input (removed CAP_DSHOW for Linux)
        cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    else:
        print("Invalid video source.")
        return

    if not cap.isOpened():
        print("Error: Could not open video.")
        csv_file.close()
        return
    
    debug_mode_on = False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    temp_center = (0,0)
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Adaptive mask size based on image dimensions
    mask_size = min(image_size[0], image_size[1]) * 3 // 4  # 75% of smaller dimension
    
    # Check if this is a dual-eye image (Aria ET produces 640x480 side-by-side)
    # We'll detect this based on the first frame
    is_dual_eye = False
    original_frame_size = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processing frame {frame_count}/{total_frames}")
        
        # Detect dual-eye format on first frame
        if original_frame_size is None:
            original_frame_size = (frame.shape[1], frame.shape[0])  # (width, height)
            # Aria ET produces 640x480 dual-eye images (320x240 per eye)
            if original_frame_size[0] == 640 and original_frame_size[1] == 480:
                is_dual_eye = True
                print(f"Detected dual-eye Aria ET format (640x480). Processing {eye_side} eye.")
            elif original_frame_size[0] == 2 * original_frame_size[1] * 4 // 3:
                # Aspect ratio suggests side-by-side dual eye
                is_dual_eye = True
                print(f"Detected dual-eye format ({original_frame_size[0]}x{original_frame_size[1]}). Processing {eye_side} eye.")
        
        # Split dual-eye image if needed
        if is_dual_eye and eye_side != 'both':
            h, w = frame.shape[:2]
            half_w = w // 2
            if eye_side == 'left':
                frame = frame[:, :half_w]  # Left half
            else:  # right
                frame = frame[:, half_w:]  # Right half

        # Crop out black border regions - Aria ET has black corners
        # Crop 15% from each edge to focus on the center where the eye actually is
        crop_percent = 0.15
        h, w = frame.shape[:2]
        crop_x = int(w * crop_percent)
        crop_y = int(h * crop_percent)
        frame = frame[crop_y:h-crop_y, crop_x:w-crop_x]
        
        # Resize frame to target resolution (Aria ET is 320x240 per eye)
        if frame.shape[1] != image_size[0] or frame.shape[0] != image_size[1]:
            frame = cv2.resize(frame, image_size)

        #find the darkest point
        darkest_point = get_darkest_area(frame)

        if debug_mode_on and not headless:
            darkest_image = frame.copy()
            cv2.circle(darkest_image, darkest_point, 10, (0, 0, 255), -1)
            cv2.imshow('Darkest image patch', darkest_image)

        # Convert to grayscale to handle pixel value operations
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
        
        # apply thresholding operations at different levels
        # at least one should give us a good ellipse segment
        thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, 5)#lite
        thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, mask_size)

        thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, 15)#medium
        thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, mask_size)
        
        thresholded_image_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, 25)#heavy
        thresholded_image_relaxed = mask_outside_square(thresholded_image_relaxed, darkest_point, mask_size)
        
        #take the three images thresholded at different levels and process them
        pupil_rotated_rect, iris_diameter_px = process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, debug_mode_on and not headless, not headless, draw_on_frame=True)
        
        # Save pupil metrics to CSV
        # pupil_rotated_rect format: ((center_x, center_y), (width, height), angle)
        center_x, center_y = pupil_rotated_rect[0]
        width, height = pupil_rotated_rect[1]
        angle = pupil_rotated_rect[2]
        timestamp_ms = (frame_count / fps) * 1000
        
        # Calculate pupil diameter as the average of width and height (ellipse axes)
        pupil_diameter_px = (width + height) / 2.0
        
        # OUTLIER FILTERING: Reject measurements that are unreasonably large
        # Typical pupil is 2-8mm, with iris ~12mm. In 320x240 image at ~0.07 mm/px,
        # max reasonable pupil is ~120px (8mm / 0.07). Use 200px as generous threshold.
        # Also reject if ellipse is very elongated (aspect ratio > 3)
        MAX_PUPIL_DIAMETER_PX = 200  # ~14mm at typical Aria calibration
        MIN_PUPIL_DIAMETER_PX = 10   # ~0.7mm minimum
        MAX_ASPECT_RATIO = 3.0
        
        aspect_ratio = max(width, height) / max(min(width, height), 1)
        is_valid_measurement = (
            MIN_PUPIL_DIAMETER_PX < pupil_diameter_px < MAX_PUPIL_DIAMETER_PX and
            aspect_ratio < MAX_ASPECT_RATIO and
            center_x > 0 and center_y > 0  # Has a valid center
        )
        
        if not is_valid_measurement:
            # Mark as invalid - set diameter to 0
            pupil_diameter_px = 0
        
        # Calculate mm_per_pixel - prefer iris-based, fall back to calibration-based
        mm_per_pixel = None
        pupil_diameter_mm = None
        
        if iris_diameter_px is not None and iris_diameter_px > 0 and iris_diameter_px < 300:
            # Iris-based mm_per_pixel (more accurate as it accounts for actual eye distance)
            # Also filter out unreasonable iris measurements
            mm_per_pixel = IRIS_DIAMETER_MM / iris_diameter_px
            mm_per_pixel_estimates.append(mm_per_pixel)
            iris_diameters_px.append(iris_diameter_px)
        
        # Use running median of mm_per_pixel estimates for stability
        # If no iris detected, fall back to calibration-based estimate
        if is_valid_measurement and len(mm_per_pixel_estimates) > 0:
            current_mm_per_pixel = np.median(mm_per_pixel_estimates[-50:])  # Use last 50 estimates
            pupil_diameter_mm = pupil_diameter_px * current_mm_per_pixel
        elif is_valid_measurement and pupil_diameter_px > 0:
            # Fall back to calibration-based mm_per_pixel
            pupil_diameter_mm = pupil_diameter_px * calib_mm_per_pixel
            current_mm_per_pixel = calib_mm_per_pixel
        else:
            current_mm_per_pixel = None
        
        # Store for summary calculations
        timestamps.append(timestamp_ms)
        pupil_diameters_px.append(pupil_diameter_px)
        pupil_diameters_mm.append(pupil_diameter_mm if pupil_diameter_mm is not None else 0)
        
        # Write to CSV
        iris_px_str = f"{iris_diameter_px:.2f}" if iris_diameter_px is not None else ""
        mm_per_px_str = f"{current_mm_per_pixel:.4f}" if current_mm_per_pixel is not None else ""
        pupil_mm_str = f"{pupil_diameter_mm:.2f}" if pupil_diameter_mm is not None else ""
        
        csv_writer.writerow([frame_count, f"{timestamp_ms:.2f}", f"{center_x:.2f}", f"{center_y:.2f}", 
                            f"{width:.2f}", f"{height:.2f}", f"{angle:.2f}", f"{pupil_diameter_px:.2f}",
                            iris_px_str, mm_per_px_str, pupil_mm_str])
        
        # Write processed frame to output
        out.write(frame)
        
        if not headless:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('d') and debug_mode_on == False:  # Press 'd' to start debug mode
                debug_mode_on = True
            elif key == ord('d') and debug_mode_on == True:
                debug_mode_on = False
                cv2.destroyAllWindows()
            if key == ord('q'):  # Press 'q' to quit
                out.release()
                break   
            elif key == ord(' '):  # Press spacebar to start/stop
                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):  # Press spacebar again to resume
                        break
                    elif key == ord('q'):  # Press 'q' to quit
                        break

    cap.release()
    out.release()
    csv_file.close()
    if not headless:
        cv2.destroyAllWindows()
    
    # Calculate summary metrics
    print("\nCalculating summary metrics...")
    
    timestamps_arr = np.array(timestamps)
    pupil_arr_px = np.array(pupil_diameters_px)
    pupil_arr_mm = np.array(pupil_diameters_mm)
    
    # Filter out invalid measurements (diameter = 0)
    valid_mask = pupil_arr_px > 0
    valid_timestamps = timestamps_arr[valid_mask]
    valid_pupils_px = pupil_arr_px[valid_mask]
    
    # Calculate mm_per_pixel calibration
    if len(mm_per_pixel_estimates) > 0:
        final_mm_per_pixel = float(np.median(mm_per_pixel_estimates))
        iris_detection_rate = len(mm_per_pixel_estimates) / frame_count * 100
        avg_iris_diameter_px = float(np.median(iris_diameters_px)) if iris_diameters_px else 0
    else:
        final_mm_per_pixel = None
        iris_detection_rate = 0
        avg_iris_diameter_px = 0
    
    # Calculate mm values using the final calibration
    if final_mm_per_pixel is not None:
        valid_pupils_mm = valid_pupils_px * final_mm_per_pixel
    else:
        valid_pupils_mm = None
    
    if len(valid_pupils_px) > 0:
        # PIXEL-BASED METRICS
        # peak_pupil: Maximum pupil diameter
        peak_pupil_px = float(np.max(valid_pupils_px))
        
        # time_to_peak: Latency to maximum pupil diameter (in ms)
        peak_idx = np.argmax(valid_pupils_px)
        time_to_peak = float(valid_timestamps[peak_idx])
        
        # pupil_mean: Mean pupil diameter
        pupil_mean_px = float(np.mean(valid_pupils_px))
        
        # avg_pupil_size: Mean pupil size (same as pupil_mean, alternative computation using median for robustness)
        avg_pupil_size_px = float(np.median(valid_pupils_px))
        
        # pupil_AUC: Area under the pupil diameter curve (using trapezoidal rule)
        # Convert timestamps to seconds for AUC
        timestamps_sec = valid_timestamps / 1000.0
        pupil_AUC_px = float(integrate.trapezoid(valid_pupils_px, timestamps_sec))
        
        # pupil_slope: Linear slope of pupil diameter over time
        if len(valid_timestamps) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(valid_timestamps, valid_pupils_px)
            pupil_slope_px = float(slope)  # units: pixels per millisecond
        else:
            pupil_slope_px = 0.0
        
        # avg_pupil_size_downsample: Mean pupil size on downsampled data (every 10th frame)
        downsampled_pupils_px = valid_pupils_px[::10]
        avg_pupil_size_downsample_px = float(np.mean(downsampled_pupils_px)) if len(downsampled_pupils_px) > 0 else 0.0
        
        # MM-BASED METRICS (if calibration available)
        if valid_pupils_mm is not None:
            peak_pupil_mm = float(np.max(valid_pupils_mm))
            pupil_mean_mm = float(np.mean(valid_pupils_mm))
            avg_pupil_size_mm = float(np.median(valid_pupils_mm))
            pupil_AUC_mm = float(integrate.trapezoid(valid_pupils_mm, timestamps_sec))
            if len(valid_timestamps) > 1:
                slope_mm, _, _, _, _ = linregress(valid_timestamps, valid_pupils_mm)
                pupil_slope_mm = float(slope_mm)
            else:
                pupil_slope_mm = 0.0
            downsampled_pupils_mm = valid_pupils_mm[::10]
            avg_pupil_size_downsample_mm = float(np.mean(downsampled_pupils_mm)) if len(downsampled_pupils_mm) > 0 else 0.0
        else:
            peak_pupil_mm = None
            pupil_mean_mm = None
            avg_pupil_size_mm = None
            pupil_AUC_mm = None
            pupil_slope_mm = None
            avg_pupil_size_downsample_mm = None
        
        # Determine calibration source
        if len(mm_per_pixel_estimates) > 0:
            calibration_source = "iris_detection"
        elif calibration is not None:
            calibration_source = "aria_camera_calibration"
        else:
            calibration_source = "default_estimate"
        
        # Create summary dictionary
        summary = {
            # Pixel-based metrics
            "peak_pupil_px": peak_pupil_px,
            "pupil_AUC_px": pupil_AUC_px,
            "pupil_slope_px": pupil_slope_px,
            "pupil_mean_px": pupil_mean_px,
            "avg_pupil_size_px": avg_pupil_size_px,
            "avg_pupil_size_downsample_px": avg_pupil_size_downsample_px,
            # MM-based metrics
            "peak_pupil_mm": peak_pupil_mm,
            "pupil_AUC_mm": pupil_AUC_mm,
            "pupil_slope_mm": pupil_slope_mm,
            "pupil_mean_mm": pupil_mean_mm,
            "avg_pupil_size_mm": avg_pupil_size_mm,
            "avg_pupil_size_downsample_mm": avg_pupil_size_downsample_mm,
            # Calibration info
            "calibration_source": calibration_source,
            "mm_per_pixel": final_mm_per_pixel,
            "calib_mm_per_pixel": calib_mm_per_pixel,  # From Aria camera or default
            "avg_iris_diameter_px": avg_iris_diameter_px,
            "iris_detection_rate_pct": iris_detection_rate,
            # Camera calibration info (if available)
            "aria_calibration": calibration,
            # Timing
            "time_to_peak_ms": time_to_peak,
            "total_frames": frame_count,
            "valid_frames": int(np.sum(valid_mask)),
            "fps": fps,
            "duration_ms": float(timestamps_arr[-1]) if len(timestamps_arr) > 0 else 0.0
        }
        
        # Save summary to JSON
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("PUPIL METRICS SUMMARY")
        print("="*60)
        print("\n  CALIBRATION:")
        if final_mm_per_pixel is not None:
            print(f"    mm_per_pixel:                {final_mm_per_pixel:.4f}")
            print(f"    avg_iris_diameter:           {avg_iris_diameter_px:.2f} px")
            print(f"    iris_detection_rate:         {iris_detection_rate:.1f}%")
        else:
            print("    WARNING: No iris detected, mm values unavailable")
        
        print("\n  PIXEL MEASUREMENTS:")
        print(f"    peak_pupil:                  {peak_pupil_px:.2f} px")
        print(f"    pupil_mean:                  {pupil_mean_px:.2f} px")
        print(f"    avg_pupil_size (median):     {avg_pupil_size_px:.2f} px")
        print(f"    avg_pupil_size_downsample:   {avg_pupil_size_downsample_px:.2f} px")
        print(f"    pupil_AUC:                   {pupil_AUC_px:.2f} px·s")
        print(f"    pupil_slope:                 {pupil_slope_px:.6f} px/ms")
        
        if valid_pupils_mm is not None:
            print("\n  MILLIMETER MEASUREMENTS:")
            print(f"    peak_pupil:                  {peak_pupil_mm:.2f} mm")
            print(f"    pupil_mean:                  {pupil_mean_mm:.2f} mm")
            print(f"    avg_pupil_size (median):     {avg_pupil_size_mm:.2f} mm")
            print(f"    avg_pupil_size_downsample:   {avg_pupil_size_downsample_mm:.2f} mm")
            print(f"    pupil_AUC:                   {pupil_AUC_mm:.2f} mm·s")
            print(f"    pupil_slope:                 {pupil_slope_mm:.6f} mm/ms")
        
        print(f"\n  TIMING:")
        print(f"    time_to_peak:                {time_to_peak:.2f} ms")
        print(f"    valid_frames:                {int(np.sum(valid_mask))}/{frame_count}")
        print("="*60)
    else:
        print("Warning: No valid pupil measurements found!")
        summary = {"error": "No valid pupil measurements"}
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"  Video saved to: {output_path}")
    print(f"  Metrics saved to: {csv_path}")
    print(f"  Summary saved to: {summary_path}")

def main():
    """Main entry point with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract pupil metrics from eye tracking video.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video file with auto-detected iris calibration
  python 2_extract_pupil_metrics.py --video eye_video.mp4
  
  # Process video with Aria VRS calibration
  python 2_extract_pupil_metrics.py --video eye_video.mp4 --vrs recording.vrs
  
  # Process webcam feed
  python 2_extract_pupil_metrics.py --webcam
  
  # Specify output directory
  python 2_extract_pupil_metrics.py --video eye_video.mp4 --output ./results/
        """
    )
    
    parser.add_argument('--video', '-v', type=str, help='Path to video file')
    parser.add_argument('--vrs', type=str, help='Path to VRS file for Aria camera calibration')
    parser.add_argument('--webcam', '-w', action='store_true', help='Use webcam instead of video file')
    parser.add_argument('--output', '-o', type=str, help='Output directory for results')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no display)')
    parser.add_argument('--eye', type=str, choices=['left', 'right'], default='left',
                        help='Which eye to use for calibration (default: left)')
    parser.add_argument('--eye-side', type=str, choices=['left', 'right', 'both'], default='right',
                        help='Which side of dual-eye Aria ET image to process (default: right)')
    
    args = parser.parse_args()
    
    # Load calibration from VRS if provided
    calibration = None
    if args.vrs:
        if os.path.exists(args.vrs):
            calibration = load_aria_calibration(args.vrs, eye=args.eye)
            if calibration:
                print(f"Loaded Aria {args.eye} ET camera calibration from VRS")
                print(f"  Image size: {calibration['image_size']}")
                print(f"  Focal length: {calibration['focal_length']}")
        else:
            print(f"Warning: VRS file not found: {args.vrs}")
    
    # Determine input source
    if args.webcam:
        process_video(None, 2, calibration=calibration, output_dir_override=args.output, eye_side=args.eye_side)
    elif args.video:
        if os.path.exists(args.video):
            process_video(args.video, 1, calibration=calibration, output_dir_override=args.output, eye_side=args.eye_side)
        else:
            print(f"Error: Video file not found: {args.video}")
            sys.exit(1)
    else:
        # Try default path or GUI selection (legacy behavior)
        video_path = 'C:/Google Drive/Eye Tracking/fulleyetest.mp4'
        if os.path.exists(video_path):
            process_video(video_path, 1, calibration=calibration, output_dir_override=args.output, eye_side=args.eye_side)
        else:
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                print("No video specified. Please select a video file.")
                video_path = filedialog.askopenfilename(
                    title="Select Video File", 
                    filetypes=[("Video Files", "*.mp4;*.avi")]
                )
                if video_path:
                    process_video(video_path, 1, calibration=calibration, output_dir_override=args.output)
                else:
                    print("No file selected. Exiting.")
                    sys.exit(0)
            except (tk.TclError, ImportError):
                parser.print_help()
                sys.exit(1)

if __name__ == "__main__":
    import sys
    main()


