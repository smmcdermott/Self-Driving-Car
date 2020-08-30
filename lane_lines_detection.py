import cv2
import numpy as np
#import matplotlib.pyplot as plt

# Get coordinates of left and right average lines
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    # Get y coordinates which are bottom of image and 3/5 of the way up
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    # Get x coordinates which can be derived from y = mx + b
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    return np.array([x1, y1, x2, y2])

# Get the average line values
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    # For each line, get the points of the lines
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # Fit line (polynomial of degree 1) to points and return the slope and y intercept
        parameters = np.polyfit(x = (x1, x2), y=(y1,y2), deg=1)
        # Get slope and y-intercept
        slope = parameters[0]
        intercept = parameters[1]
        # If slope is negative, add it to left list
        if slope < 0:
            left_fit.append((slope, intercept))
        # Else add it to right list
        else:
            right_fit.append((slope, intercept))
        if len(left_fit) and len(right_fit):
            # Get average slope and y intecept of left and right lines
            left_fit_average = np.average(left_fit, axis=0)
            right_fit_average = np.average(right_fit, axis=0)
            # Get coordinates of left and right lines
            left_line = make_coordinates(image, left_fit_average)
            right_line = make_coordinates(image, right_fit_average)
            # Return lines
            return np.array([left_line, right_line])

# Get the gradient of the image
def canny(image):
    # Convert colored image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Reduce Noise in image by applying Gaussian Blur with 5x5 filter (weighted average of filter and image)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # Apply Canny function to calculate gradient of adjacent cells to detect edges.
    # Only identifies components with high changes in brightness
    canny = cv2.Canny(image=blur, threshold1=50, threshold2=150)
    return canny

# Display the lines on black image
def display_lines(image, lines):
    # Get black image of same dimensions as input image
    line_image = np.zeros_like(image)
    # If line is not empty
    if lines is not None:
        # For each line, get the coordinate points
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # Put the lines on the black image that correspond to the points with a blue color and a thickness of 10
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), thickness=10)
    return line_image

# Specify the region of interest by applying polygonal contour
def region_of_interest(image):
    # Get image height (vertical)
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    # Get full black image (all pixels have zero intensity) with same dimensions as image
    mask = np.zeros_like(a=image)
    # Fill image with triangle (region of interest)
    cv2.fillPoly(img=mask, pts=polygons, color=255)
    
    # Only show region of interest by expressing elements as binary values where 0s are 00000000 and 255s are 11111111
    # Using bitwise &, the values compare the same element from both the mask and the image and only accept the element
    # as 1 if both elements are equal to 1. 
    # This way, all images not in the region of interest (value of 0) will be 00000000s, and all regions in the region of
    # interest will remain the same
    masked_image = cv2.bitwise_and(src1=image, src2=mask)
    return masked_image

# =============================================================================
# # Read image data as multidimensional numpy array
# image = cv2.imread('test_image.jpg')
# # Work with copy of array
# lane_image = np.copy(image)
# # Only return areas of high gradients
# canny_image = canny(lane_image)
# # Only return region of interest
# cropped_image = region_of_interest(canny_image)
# # Use the cropped image with a perpendicular distance of 2, theta of 1 radians and a threshold
# # (the minimum number of votes needed) of 100
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 
#                        100, np.array([]), minLineLength=40, 
#                        maxLineGap=5)
# # Get average value of lines to have single line detection
# average_lines = average_slope_intercept(lane_image, lines)
# # Detect average lines on black image
# line_image = display_lines(lane_image, average_lines)
# # Display detected lines on actual image. Multiply image by 0.8 to darken and keep 
# # line image the same to increase contrast
# combo_image = cv2.addWeighted(src1=lane_image, alpha=0.8, src2=line_image, beta=1, gamma=1)
# # Render image
# cv2.imshow('result', combo_image)
# # Display image 
# cv2.waitKey(0)
# =============================================================================

cap = cv2.VideoCapture("test2.mp4")
while (cap.isOpened()):
    _, frame = cap.read()
    # Only return areas of high gradients
    canny_image = canny(frame)
    # Only return region of interest
    cropped_image = region_of_interest(canny_image)
    # Use the cropped image with a perpendicular distance of 2, theta of 1 radians and a threshold
    # (the minimum number of votes needed) of 100
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 
                           100, np.array([]), minLineLength=40, 
                           maxLineGap=5)
    # Get average value of lines to have single line detection
    average_lines = average_slope_intercept(frame, lines)
    # Detect average lines on black image
    line_image = display_lines(frame, average_lines)
    # Display detected lines on actual image. Multiply image by 0.8 to darken and keep 
    # line image the same to increase contrast
    combo_image = cv2.addWeighted(src1=frame, alpha=0.8, src2=line_image, beta=1, gamma=1)
    # Render frame
    cv2.imshow('result', combo_image)
    # Display frame until q is entered
    if cv2.waitKey(1) == ord('q'):
        break
    
# Stop showing frames and close window
cap.release()
cv2.destroyAllWindows()

# Edge Detection - finding sharp changes within an image among adjacent cells
    # Gradient - Change of brightness within an image (High Gradient - image goes from white to black)
    # Areas with high gradients are the edges due to the rapid change in brightness in adajacent cells

# Step 1 - Greyscale our image
# Step 2 - Reduce Noise (Smooth image)
# Step 3 - Apply the Canny function which calculates the gradient (derviative) of all adjacent pixels in both x and y directions
    # If Gradient is above high threshold, it is accepted
    # If below low threshold, it is denied
    # If between high and low thresholds, then it is only accepted if adjacent to strong edge (high gradient value)
# Step 4 - Only return the region of interest
# Step 5 - Detect lines through Hough Transform
    # Hough Transform - the slope (m) and y-intercept (b) pairing that corresponds to a given line (y = mx + b)
    # Suppose we have a series lines as represented by the x, y pair, we then represent these lines in hough space.
    # In order to find a line of best fit, we vote on the bin (area) in hough space where the lines most closely intersect
    # We then use that m and b pair from the bin to set the line of best fit
    # To make this method more robust (since vertical lines have an infinite slope), we map the x, y pair as polar coordinates
        # p = xsin(theta) + ycos(theta)
            # p is the perpendicular distance from the origin
            # theta is the angle between p and the x axis in radians (angle of inclination)
