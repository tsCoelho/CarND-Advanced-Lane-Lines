#!/usr/bin/env python
# coding: utf-8

# ## Advanced Lane Finding Project
# 
# The goals / steps of this project are the following:
# 
# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# * Apply a distortion correction to raw images.
# * Use color transforms, gradients, etc., to create a thresholded binary image.
# * Apply a perspective transform to rectify binary image ("birds-eye view").
# * Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
# 

# In[1]:

import traceback 

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'qt')


# ## First, I'll compute the camera calibration using chessboard images

# In[2]:


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('../camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:


    img = mpimg.imread(fname)
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        
        #plt.imshow(img)

        
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


# ## Helper functions

# In[3]:

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def weighted_img(img, initial_img, α=1, β=.3, γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)
    

def add_annotation(img, text, position = (80, 80), typed_position='TOP_LEFT'):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 0.9
    color = (255, 255, 255) 
    thickness = 2

    img = cv2.putText(img, text, position, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
    
    return img
        
def undistort(img, mtx, dist):
    return  cv2.undistort(img, mtx, dist, None, mtx)


def color_gradient_threshold_pipeline(img):

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    r_channel = img[:,:,0]
    g_channel = img[:,:,1]
    
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    
    # Color threshold
    rgb_color_tresh = (185, 255)
    rgb_color_binary = np.zeros_like(r_channel)
    rgb_color_binary[(r_channel > rgb_color_tresh[0]) & (r_channel <= rgb_color_tresh[1]) & (g_channel > rgb_color_tresh[0]) & (g_channel <= rgb_color_tresh[1]) ] = 1
    
    # Gradient threshold
    sx_thresh = (10, 120)
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    color_gradient_binary = cv2.bitwise_or(sxbinary, rgb_color_binary, mask = None) 
    return color_gradient_binary
    

def color_gradient_threshold_pipeline_v2(img):
    
    r_channel = img[:,:,0]
    g_channel = img[:,:,1]
    
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
            

    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(img,img, mask= mask)
    #return res


    r_channel = gaussian_blur(r_channel,3)
    g_channel = gaussian_blur(g_channel,3)
    s_channel = gaussian_blur(s_channel,3)
    
    sx_thresh = (5, 120)
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    
    grad_img = cv2.bitwise_and(img,img, mask=sxbinary)
    
    #return grad_img
    hsv = cv2.cvtColor(grad_img, cv2.COLOR_RGB2HSV)
    
    # define range of blue color in HSV
    #lower_yellow = np.array([20,20,50])
    # define range of blue color in HSV
    lower_yellow = np.array([20,25,50])
    upper_yellow = np.array([40,255,255])
    
    lower_white = np.array([0,0,100])
    upper_white = np.array([150,65,255])
    
    
    
    # Threshold the HSV image 
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)//255
    mask_white = cv2.inRange(hsv, lower_white, upper_white)//255

    #mask = mask_yellow + mask_white    
    
    hsv_mask = cv2.bitwise_or(mask_yellow, mask_white)
    
    return hsv_mask
    
    r_channel = grad_img[:,:,0]
    g_channel = grad_img[:,:,1]
    
    rgb_color_tresh = (180, 255)
    rgb_color_binary = np.zeros_like(r_channel)
    
    rgb_color_binary[(r_channel > rgb_color_tresh[0]) & (r_channel <= rgb_color_tresh[1]) & (g_channel > rgb_color_tresh[0]) & (g_channel <= rgb_color_tresh[1]) ] = 1
    
    return rgb_color_binary


def _color_gradient_threshold(img, gradient_channel, s_thresh, color_channel, sx_thresh):
    # Sobel x
    sobelx = cv2.Sobel(gradient_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(color_channel)
    s_binary[(color_channel >= s_thresh[0]) & (color_channel <= s_thresh[1])] = 1
    

    # Stack each channel
    #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    color_binary = cv2.bitwise_or(sxbinary, s_binary, mask = None) 

    return color_binary

def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image    


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    onethirdpoint = np.int(histogram.shape[0]//3)
    
    
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    
    #leftx_base = np.argmax(histogram[:onethirdpoint])
    #rightx_base = np.argmax(histogram[(midpoint + onethirdpoint):]) + midpoint + onethirdpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 20
    # Set the width of the windows +/- margin
    margin = 70
    # Set minimum number of pixels found to recenter window
    minpix = 90
    
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    
        
    lane_width_pixels = np.absolute(rightx_current - leftx_current)
    lane_width_m = lane_width_pixels * xm_per_pix
    

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        win_y_mid = (win_y_high + win_y_low) // 2
        
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin       

        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        ##############
        # Left line                
        ##############        
        if ( win_y_low < binary_warped.shape[0] // 2 ) or ( win_xleft_low > -margin * 0.25 ):
            #or ( (leftx_base - leftx_current) > 1.5* margin) :                              
    
            good_left_bool = (nonzerox >= win_xleft_low) & (nonzerox <= win_xleft_high) & (nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)
            #& (win_xleft_low > 0)
            #& (nonzerox < win_xright_low)
            good_left_inds = good_left_bool.nonzero()[0]
                            
            left_pixels_found = np.shape(good_left_inds)[0]
            
            ## DEBUG
            #add_annotation(out_img, '{}'.format(left_pixels_found), ( win_xleft_high, win_y_mid ) )

            # If you found > minpix pixels, recenter next window ###
            if left_pixels_found > minpix:
                leftx_mean = np.mean(nonzerox[good_left_inds], dtype=np.int32)
                leftx_current = leftx_mean
                
                left_lane_inds.append(good_left_inds)

                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),
                (win_xleft_high,win_y_high),(255,0,0), 2)            
            

        ##############
        # Right line
        ##############        
        if ( win_y_low < binary_warped.shape[0] // 2 )  or ( win_xright_high < binary_warped.shape[1] + margin * 0.25 ):
            #or ( ( rightx_current - rightx_base ) > 1.5* margin):
                    
            good_right_bool = (nonzerox >= win_xright_low) & (nonzerox <= win_xright_high) & (nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)        
            good_right_inds = good_right_bool.nonzero()[0]              
            
            right_pixels_found = np.shape(good_right_inds)[0]
            
            ## DEBUG
            #add_annotation(out_img, '{}'.format(right_pixels_found), ( win_xright_low, win_y_mid ) )

            if right_pixels_found > minpix:
                rightx_mean = np.mean(nonzerox[good_right_inds], dtype=np.int32)        
                rightx_current = rightx_mean
                
                # Append these indices to the lists
                right_lane_inds.append(good_right_inds)  

                cv2.rectangle(out_img,(win_xright_low,win_y_low),
                (win_xright_high,win_y_high),(0,0,255), 2)         

    
    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        if (np.shape(left_lane_inds)[0] > 0):
            left_lane_inds = np.concatenate(left_lane_inds)
        
    except ValueError as err:
        print('oops {}'.format(repr(err)))
        
    try:
        if (np.shape(right_lane_inds)[0] > 0):
            right_lane_inds = np.concatenate(right_lane_inds)
        
    except ValueError as err:
        print('oops {}'.format(repr(err)))
        


    try:
        
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
    except IndexError as err:
        print('oops {}'.format(err))
        
        print('nonzerox: {}, left_lane_inds: {}, right_lane_inds: {}'.format(repr(nonzerox), repr(left_lane_inds), repr(right_lane_inds)))
    
    add_annotation(out_img, 'Lane width: {} px'.format(lane_width_pixels), (500, 80 ))
    add_annotation(out_img, 'Lane width: {:.2f} m'.format(lane_width_m), (500, 120 ))

    return leftx, lefty, rightx, righty, out_img, histogram

    
    
def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img, histogram = find_lane_pixels(binary_warped)

    left_fit = None
    right_fit = None
    left_curverad = 0
    right_curverad = 0
    
    left_fitx = []
    right_fitx = []
    
    poly_order = 2
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    try:
        if (np.shape(leftx)[0] > 0):
            left_fit = np.polyfit(lefty, leftx, poly_order)            
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            
            left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, poly_order)        
            if left_fit_cr is not None:
                left_curverad = measure_curvature_real(ploty, left_fit_cr)            
    
        
        if (np.shape(rightx)[0] > 0):
            right_fit = np.polyfit(righty, rightx, poly_order)                                         
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, poly_order)                
            if right_fit_cr is not None:
                right_curverad = measure_curvature_real(ploty, right_fit_cr)
    
            

    except TypeError as err:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('ERROR | fit_polynomial | The function failed to fit a line! {}'.format(err))

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    
    
    return left_fit, right_fit, left_fitx, right_fitx, left_curverad, right_curverad, ploty, out_img

def _fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### Fit a second order polynomial to each with np.polyfit() ###
    left_fit = None
    right_fit = None
    
    left_fitx = []
    right_fitx = []
    
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    
    if np.shape(leftx)[0] > 0 and np.shape(lefty)[0] > 0:
        left_fit = np.polyfit(lefty, leftx , 2)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]

    if np.shape(rightx)[0] > 0 and np.shape(righty)[0] > 0:
        right_fit = np.polyfit(righty, rightx , 2)
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return left_fit, right_fit, left_fitx, right_fitx, ploty

def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    
    left_fit_cr = None
    right_fit_cr = None
    
    left_curverad = 0
    right_curverad = 0
    
    poly_order = 2
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    
    ploty = nonzeroy
    
    prev_fit_leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    upper_margin_leftx = prev_fit_leftx + margin
    bottom_margin_leftx = prev_fit_leftx - margin
    
    prev_fit_rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    upper_margin_rightx = prev_fit_rightx + margin
    bottom_margin_rightx = prev_fit_rightx - margin
    
    good_left_bool = (nonzerox >= bottom_margin_leftx) & (nonzerox <= upper_margin_leftx)
    good_right_bool = (nonzerox >= bottom_margin_rightx) & (nonzerox <= upper_margin_rightx) 
        
    good_left_inds = good_left_bool.nonzero()[0]
    good_right_inds = good_right_bool.nonzero()[0]
        

    left_lane_inds = good_left_inds
    right_lane_inds = good_right_inds
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
        
    # Fit new polynomials
    left_fit, right_fit, left_fitx, right_fitx, ploty = _fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    if np.shape(left_fitx)[0] > 0:
        left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, poly_order)        
        if left_fit_cr is not None:
            left_curverad = measure_curvature_real(ploty, left_fit_cr)      
    
    if np.shape(right_fitx)[0] > 0:    
        right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, poly_order)                
        if right_fit_cr is not None:
            right_curverad = measure_curvature_real(ploty, right_fit_cr)
    

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
            
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    result = out_img
    try:

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    except Exception as err: 
        print('ERROR | search_around_poly | Couldnt make image: {}'.format(err))
        
    finally:
        
        # Plot the polynomial lines onto the image
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        ## End visualization steps ##
    
        return left_fit, right_fit, left_fitx, right_fitx, left_curverad, right_curverad, ploty, result, window_img


def measure_curvature_real(ploty, fit_cr):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
        
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    
    return curverad

def draw_lane_boundaries(warped_img, Minv, left_fitx, right_fitx, ploty, img_size):
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    try:    
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
    
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    except ValueError as err:
        print('Error | draw_lane_boundaries | {}'.format(err))
        print('left_fitx: {}, right_fitx: {}'.format(left_fitx, right_fitx))
        print('')
        
    finally:
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        return cv2.warpPerspective(color_warp, Minv, img_size) 
    


# ### Read a test image

# In[4]:


#reading in an image
#image = mpimg.imread('test_images/straight_lines1.jpg')
image = mpimg.imread('../camera_cal/calibration1.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
#plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

undist = undistort(image,mtx,dist)
plt.imshow(undist)


# In[5]:


#reading in an image
#image = mpimg.imread('test_images/straight_lines1.jpg')
image = mpimg.imread('../test_images/straight_lines1.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


# In[6]:


_MAX_QUADRADIC_DIFF = 1
_MAX_QUADRADIC_DEVIATION = 1

_MAX_LINEAR_DIFF = 1
_MAX_LINEAR_DEVIATION = 1
    

# Define a class to receive the characteristics of each line detection
class Line():

    def __init__(self):

        self.n_iterations = 50
        self.detection_window = self.n_iterations * 10

        # was the line detected in the last iteration?
        self.detected = False  

        #average x values of the fitted line over the last n iterations
        self.bestx = None     

        #polynomial coefficients averaged over the last n iterations
        self.best_fit = np.array([0,0,0], dtype='float') 

        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        
        self.curvatures= np.array([])

        #polynomial coefficients for the most recent fit
        self.current_fit = np.array([0,0,0], dtype='float') 
        self.good_fits = np.array([ [0,0,0] ], dtype='float')
        
        self.good_or_bad_fits = np.array([])


        #radius of curvature of the line in some units
        self.radius_of_curvature = 0 

        #distance in meters of vehicle center from the line
        self.line_base_pos = 0 

        #difference in fit coefficients between last and new fits
        self.current_diff = np.array([0,0,0], dtype='float') 
        self.diffs = np.array([[0,0,0]], dtype='float') 

        self.current_deviation = np.array([0,0,0], dtype='float') 
        self.deviations = np.array([[0,0,0]], dtype='float') 

        #x values for detected line pixels
        self.allx = None  

        #y values for detected line pixels
        self.ally = None  


    def is_good_fit(self):        
        #return np.absolute(self.current_deviation[0]) < _MAX_QUADRADIC_DEVIATION and np.absolute(self.current_deviation[1]) < _MAX_LINEAR_DEVIATION
        return True
    
    def update_good_fit(self, fit, fitted_x, y, curvature):

        if fit is None or len(fitted_x) == 0:
            self.detected = False
            return self.detected            
        
        try:            
            temp_fits = self.good_fits                    
            temp_fits = np.append(temp_fits, [fit], axis=0)         
            if ( np.shape(temp_fits)[0] > self.n_iterations ):
                temp_fits = np.delete(temp_fits, 0, axis=0)

            self.curvatures = np.append(self.curvatures, curvature)                                     
            if ( np.shape( self.curvatures)[0] > self.n_iterations ):
                self.curvatures = np.delete(self.curvatures, 0)                                      
    
    
            self.current_diff = self.current_fit - fit        
            self.diffs = np.append(self.diffs, [self.current_diff], axis=0) 
            
            if ( np.count_nonzero(self.current_fit) > 0 ):
                self.current_deviation = self.current_diff / self.current_fit    
                self.deviations = np.append(self.deviations, [self.current_deviation], axis=0)         
               
            if self.is_good_fit():            
                # Update line
                self.detected = True
                self.current_fit = fit
                self.good_fits = temp_fits
                
                self.allx = fitted_x
                self.ally = y
                
                # Update best fit                            
                my_weights = 0.5**np.arange(len(self.good_fits), 0, -1)
                my_weights[-1] = my_weights[-1]+ (1 - sum(my_weights))
                
                #my_weights = [0.75, 0.15, 0.05, 0.025, 0.025]
            
                self.best_fit = np.average(self.good_fits, axis=0, weights=my_weights[:len(self.good_fits)])
                self.bestx = self.best_fit[0]*self.ally**2 + self.best_fit[1]*self.ally + self.best_fit[2]
    
                self.radius_of_curvature = np.average(self.curvatures, axis=0, weights=my_weights[:len(self.curvatures)])

                
            else:
                self.detected = False
    
        
        except Exception as err:
            self.detected = False

            print('Error when updating line props: {}'.format(err))
            print('fit: {}, fitted_x: {}'.format(repr(fit), fitted_x))
            traceback.print_exc() 


        
        finally:
            return self.detected        

    def has_previous_good_fit(self):
        return self.detected == True
    
    def update_detection_ratio(self, is_line_detected):
        self.good_or_bad_fits = np.append(self.good_or_bad_fits, is_line_detected)         
        if ( np.shape( self.good_or_bad_fits)[0] > self.detection_window ):
                self.good_or_bad_fits = np.delete(self.good_or_bad_fits, 0)   
        
    def get_latest_detection_ratio(self):
        return np.mean(self.good_or_bad_fits)
      
        
    def get_mean_curvature(self):
        return np.mean(self.curvatures)
    
    
class LaneDetector():
    def __init__(self, bird_eye_lane_img, original_image, left_line_obj, right_line_obj ):
        
        self.current_frame = bird_eye_lane_img
        self.original_image = original_image
        
        self.lines = [left_line_obj, right_line_obj]
        
        self.left_line = left_line_obj
        self.right_line = right_line_obj
        
        self.lane_width_m = 3.6
        self.paralellity = 0
        self.slopes_coeherent = False

        
    def search_around_previous_fit(self):
        return search_around_poly(self.current_frame, left_line.current_fit, right_line.current_fit)                
        
    def fit_sliding_window(self):
        return fit_polynomial(self.current_frame)
    
    def draw_lane_area(self, Minv, image):
        img_size = (image.shape[1], image.shape[0])

        lane_boundaries_img = draw_lane_boundaries(self.current_frame, Minv, left_line.bestx, right_line.bestx, left_line.ally, img_size)
        
        lane_boundaries_merged = weighted_img(lane_boundaries_img, image)
        
        return lane_boundaries_merged
    
    def add_annotation(self, text, position = (80, 80)):
        return add_annotation(self.original_image, text, position)
        
    def check_sanity(self):
        return self.paralellity > 40 and self.paralellity < 300 and self.lane_width_m > 2
        #return True    
    
        
    def check_lane_width(self):    
        pass
        
    def check_curvature(self):
        pass



# In[ ]:



def process_image_v2(image, left_line, right_line):    

    img = np.copy(image)

    # Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    # Apply a distortion correction to raw images.
    undist = undistort(img, mtx, dist)
   # plt.show(undist)
   # plt.savefig('../output_images/undistorted_road.png')
    # Use color transforms, gradients, etc., to create a thresholded binary image.

    # Good lighting conditions
    #grad = color_gradient_threshold_pipeline(undist)

    
    # Varying lighting conditions
    grad = color_gradient_threshold_pipeline_v2(undist)
    
    img_size = (undist.shape[1], undist.shape[0])
    
    
    SHORT_RADIUS_CURVATURE = 300
    
    badly_fitted_lines = (left_line.get_latest_detection_ratio() < 0.5) or (right_line.get_latest_detection_ratio() < 0.5)
    short_radius_curvature = (left_line.get_mean_curvature() < SHORT_RADIUS_CURVATURE) or (right_line.get_mean_curvature() < SHORT_RADIUS_CURVATURE)

    '''
    #if short_radius_curvature and badly_fitted_lines:
    if False: 
        # look closer
        src = np.array([(320, 675),
                    (548, 476),
                    (720, 476),
                    (1070, 675)], dtype=np.float32)
    

    
    else:    
        # straight lines
        # look further
        src = np.array([(190, img_size[1]),
                (565, 464.0),
                (690, 464.0),
                (1125, img_size[1])], dtype=np.float32)
    '''

    src = np.array([(258, 675),
        (569, 466.0),
        (713, 466.0),
        (1049, 675)], dtype=np.float32)
    
    #src_roi = np.int32([src])
    #roi = region_of_interest(undist, src_roi)
    #cv2.polylines(image, src_roi, True, (255,0,0), 2)

    offset_h = 250
    offset_v_bottom = 0
    offset_v_top = 0
    dst = np.float32([
                    [offset_h, img_size[1]-offset_v_bottom],
                    [offset_h, offset_v_top], 
                    [img_size[0]-offset_h, offset_v_top],                 
                    [img_size[0]-offset_h, img_size[1]-offset_v_bottom]])


    # Apply a perspective transform to rectify binary image ("birds-eye view").
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(grad, M, img_size)

    # Start lane detection pipeline
    
    lane_detector = LaneDetector(warped, image, left_line, right_line )
    # UNCOMENT FOR BOTH SEARCH METHODS
    
    '''    

    if left_line.has_previous_good_fit() and right_line.has_previous_good_fit():
        # Search for pixels around previous fit 
        left_fit, right_fit, left_fitted_x, right_fitted_x, left_curverad, right_curverad, ploty, search_poly_lines, output_binary_lanes = lane_detector.search_around_previous_fit()
        debug_search_img = search_poly_lines

        lane_detector.add_annotation('Search around poly', (80, 120) );    

    else:        
        # Detect lane pixels and fit to find the lane boundary
        left_fit, right_fit, left_fitted_x, right_fitted_x, left_curverad, right_curverad, ploty, sliding_window_img = lane_detector.fit_sliding_window()
        debug_search_img = sliding_window_img
        
        lane_detector.add_annotation('Sliding window', (80, 120) );    
    '''

    left_fit, right_fit, left_fitted_x, right_fitted_x, left_curverad, right_curverad, ploty, sliding_window_img = lane_detector.fit_sliding_window()
    #lane_detector.add_annotation('Sliding window', (80, 120) );  
    debug_search_img = sliding_window_img
    
    
    quadratic_slope = False
    linear_slope = False

    try:
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        lane_detector.lane_width_m = abs(right_fit[2] - left_fit[2]) * xm_per_pix
        
        quadratic_slope = ( left_fit[0] * right_fit[0] ) > 0
        linear_slope = ( left_fit[1] * right_fit[1] ) > 0
        lane_detector.slopes_coeherent = linear_slope and quadratic_slope
    
        lane_detector.paralellity = abs( left_fit[1] / right_fit[1] ) * 100        

        add_annotation(debug_search_img, 'Slope ok: {}'.format(lane_detector.slopes_coeherent), (500, 210) );    
        add_annotation(debug_search_img, 'Parallelity: {:.2f}%'.format(lane_detector.paralellity), (500, 240) );    
    
        
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitted_x, ploty]))])
        left_line_pts = left_line_window1
        
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitted_x, ploty]))])    
        right_line_pts = right_line_window1
        
        # Draw the lane onto the warped blank image
        cv2.polylines( debug_search_img, np.int_([left_line_pts]), isClosed=False, color=(255,255, 0), thickness=2)
        cv2.polylines( debug_search_img, np.int_([right_line_pts]), isClosed=False, color=(255,255, 0), thickness=2)
    
    
        add_annotation(debug_search_img, 'Left fit: {:.5f} {:.2f} {:.2f}'.format(left_fit[0], left_fit[1], left_fit[2]), (500, 150) );    
        add_annotation(debug_search_img, 'Right fit: {:.5f} {:.2f} {:.2f}'.format(right_fit[0], right_fit[1], right_fit[2]), (500, 180) );    

        
    except Exception as err:
        print('ERROR | process_image | Couldnt compute sanity metrics: {}'.format(err))
        traceback.print_exc() 

    
   

    if lane_detector.check_sanity():
        is_good_left_fit_detected = left_line.update_good_fit(left_fit, left_fitted_x, ploty, left_curverad)
        is_good_right_fit_detected = right_line.update_good_fit(right_fit, right_fitted_x, ploty, right_curverad)

    else:
        is_good_left_fit_detected = False
        is_good_right_fit_detected = False
        
    left_line.update_detection_ratio(is_good_left_fit_detected)
    right_line.update_detection_ratio(is_good_left_fit_detected)

    
    left_line.current_fit = left_line.best_fit
    right_line.current_fit = right_line.best_fit

    if ( is_good_left_fit_detected == False):    
        #lane_detector.add_annotation('Left line not detected! :(', (80, 80) );    
        left_line.current_fit = left_line.best_fit
        left_line.allx = left_line.bestx


    if ( is_good_right_fit_detected == False):
        #lane_detector.add_annotation('Right line not detected! :(', (500, 80));    
        right_line.current_fit = right_line.best_fit
        right_line.allx = right_line.bestx


    
    # Determine the curvature of the lane and vehicle position with respect to center.
    # TODO
    #left_curvature, right_curvature, mean_curvature, center_lane = measure_curvature_pixels(img_size[1], left_line.current_fit, right_line.current_fit)
    
    if left_line.current_fit is not None and len(left_line.current_fit) > 0 and right_line.current_fit is not None and len(right_line.current_fit) > 0:
                
        mean_curvature = np.mean([ left_line.radius_of_curvature, right_line.radius_of_curvature ])
        
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    
        center_lane_m = np.mean([left_line.current_fit[2], right_line.current_fit[2]]) * xm_per_pix        
        center_vehicle_m = img_size[0]*0.5*xm_per_pix 
        
        distance_from_center_m = center_vehicle_m - center_lane_m   
        abs_distance_from_center_m = np.absolute(distance_from_center_m)    
                
        side = ''
        vehicle_position_annotation = 'Vehicle is centered!' 
        if distance_from_center_m > 0:
            side = 'right'
        elif distance_from_center_m < 0:
            side = 'left'
        else:
            side = 'center'             
        
        vehicle_position_annotation = 'Vehicle is {:.2f} m to the {} from the center'.format(abs_distance_from_center_m, side)

        lane_detector.add_annotation('Radius of curvature: {:.2f} m'.format(mean_curvature), (80, 160) );  
        lane_detector.add_annotation(vehicle_position_annotation, (80, 200) );  
       
        

    # Warp the detected lane boundaries back onto the original image.
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    lane_area_img = lane_detector.draw_lane_area(Minv, image)
    
    
    debug_imgs = {'color_gradient': grad, 'warped_gradient': warped, 'search_method': debug_search_img }
    return lane_area_img, debug_imgs



image_frame = cv2.imread('../debug/debug_curv1.png')
#image_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)

left_line = Line()
right_line = Line()

img, debug_imgs = process_image_v2(image_frame, left_line, right_line)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 9))
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 9))
f.tight_layout()

ax1.imshow(image_frame)
ax1.set_title('Original image', fontsize=40)

ax2.imshow(debug_imgs['warped_gradient'])
ax2.set_title('Warped gradient', fontsize=40)

ax3.imshow(debug_imgs['search_method'])
ax3.set_title('Search method', fontsize=40)

ax4.imshow(img)
ax4.set_title('Result', fontsize=40)

cv2.destroyAllWindows()  



# In[ ]:

# Import everything needed to edit/save/watch video clips\n",
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip
from IPython.display import HTML


#input_video = '../project_video.mp4'
input_video = '../challenge_video.mp4'
#input_video = '../harder_challenge_video.mp4'


# HOMEMADE VIDEO
# Falta calibrar src e dst e talvez camera
#input_video = '../my_videos/my_new_video.mp4'


myclip = VideoFileClip(input_video)
#.subclip(20,26)
#myclip = VideoFileClip("../project_video.mp4").subclip((40),(50))


WAIT_FOR_FRAME_MS = 1

KEYS = { '1': 49, '2': 50, '3': 51, '4': 52, 'ESC': 27, 'SPACE': 32 }
 
left_line = Line()
right_line = Line()
images=[]
debug_images=[]

frame_counter = 0
for image_frame in myclip.iter_frames():
    
    img, debug_imgs = process_image_v2(image_frame, left_line, right_line)
    
    
    add_annotation(img, 'Frame: {}'.format(frame_counter), (80,35))
    add_annotation(img, 'Time: {:.2f} s'.format(frame_counter / myclip.fps), (80,80))

    frame_counter= frame_counter + 1 
    
    
    result = np.copy(img)
    color_gradient = debug_imgs['color_gradient']
    binary_warped = debug_imgs['warped_gradient']
    search_poly_lines = debug_imgs['search_method']
 
    scale_percent = 70 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    search_poly_lines = cv2.resize(search_poly_lines, dim, interpolation = cv2.INTER_AREA)

    
    color_gradient = np.dstack((color_gradient, color_gradient, color_gradient))*255
    color_gradient = cv2.resize(color_gradient, dim, interpolation = cv2.INTER_AREA)

    
    binary_stack = np.dstack((binary_warped, binary_warped, binary_warped))*255
    binary_stack = cv2.resize(binary_stack, dim, interpolation = cv2.INTER_AREA)

    numpy_image_concat_search = np.concatenate((img, search_poly_lines), axis=1)
    numpy_image_concat_binary = np.concatenate((color_gradient, binary_stack), axis=1)

    numpy_image_concat = np.concatenate((numpy_image_concat_search , numpy_image_concat_binary ), axis=0)

    cv2.imshow('debug', numpy_image_concat)
    
    debug_images.append(numpy_image_concat)
    
    pressed_key = cv2.waitKey(WAIT_FOR_FRAME_MS)
    
    
    if pressed_key == KEYS['1']:
        cv2.imwrite('..\debug\debug_1.png', cv2.cvtColor(result, cv2.COLOR_BGR2RGB)) 
    
    elif pressed_key == KEYS['2']:
        cv2.imwrite('..\debug\debug_search_poly_lines.png', search_poly_lines) 
        
    elif pressed_key == KEYS['3']:
        cv2.imwrite('..\debug\debug_color_grad.png', color_gradient) 
 
    elif pressed_key == KEYS['4']:
        cv2.imwrite('..\debug\debug_binary_warped.png', binary_stack) 

    elif pressed_key == KEYS['ESC']:
        cv2.destroyAllWindows()  
        break   

    elif pressed_key == KEYS['SPACE']:
        
        if WAIT_FOR_FRAME_MS == 0:        
            WAIT_FOR_FRAME_MS = 1
        else:
            WAIT_FOR_FRAME_MS = 0
    
    images.append(img)

cv2.destroyAllWindows()  


clip = ImageSequenceClip(images, fps=20)
clip.write_videofile('../challenge_video_output.mp4', audio=False)
'''
clip = ImageSequenceClip(debug_images, fps=20)
clip.write_videofile('../challenge_video_debug_output.mp4', audio=False)

'''
