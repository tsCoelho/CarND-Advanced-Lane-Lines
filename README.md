# Project Writeup 
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undistorted_calibration]: ./output_images/undistorted.png "Undistorted"
[undistorted_road]: ./output_images/undistorted_road.png "Road Transformed"   
[color_grad]: ./output_images/color_gradient_thresholded.png "Binary Example"        
[perspective_transform]: ./output_images/perspective_transform.png "Warp Example"
[sliding_window_search_poly_lines]: ./output_images/sliding_window_search_poly_lines.png "Fit Visual"
[output]: ./output_images/output.png "Output"
[debug_output]: ./output_images/debug_visualization.png "Debug visualization"

[project_video]: ./project_video_output_writeup.mp4 "Project video"
[challenge_video]: ./challenge_video_output.mp4 "Challenge video"
[harder_challenge_video]: ./harder_challenge_video_output.mp4 "Harder challenge video"



## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

#### The project is contained in the python file `src/AdvancedLaneFinding_vFinal.py`.

---
### Camera Calibration

#### 1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images

The code for this step is in lines 35 through 67 of the given py file.

I start by considering the chessboard corners 9x6 and then preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][undistorted_calibration]

### Pipeline (single images)

#### 1. Correct road distortion

Having the camera calibration matrix (`mtx`) and distortion coefficients (`dist`) for our car camera we can simply apply these parameters directly to undistort captured frames

```python
    cv2.undistort(img, mtx, dist, None, mtx)
```

The correction (despite slight) can be seen here:

![alt text][undistorted_road]                                                                                                             
                                                                                                                          
                                                                                                                          
#### 2. Create a tresholded binary image using color transforms and gradients

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at function `color_gradient_threshold_pipeline` in py file provided).  

I filtered for "whiteish" and "yellowish" colors from colorspace RGB and used the S channel from HLS colorspace for the Sobel derivative in x direction.

Combining (summing) both thresholds we get the following binary image:

![alt text][color_grad]

#### 3. Perspective transform

The code for my perspective transform is included in the provided py file in function process_image_v2, from lines 876 to 896.  I chose the hardcode the source and destination points in the following manner:

```python

src = np.float32([
                (258, 675),
                (569, 466),
                (713, 466),
                (1049, 675)])        
offset_h = 300
offset_v_bottom = 0
offset_v_top = 0

dst = np.float32([
                [offset_h, img_size[1]-offset_v_bottom],
                [offset_h, offset_v_top], 
                [img_size[0]-offset_h, offset_v_top],                 
                [img_size[0]-offset_h, img_size[1]-offset_v_bottom]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 258, 675      | 300, 720      | 
| 569, 466      | 320, 0        |
| 713, 466      | 980, 0        |
| 1049, 675     | 980, 720      |


We can then apply a perspective transform to get a "birds-eye view" binary image.

```python    
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(grad, M, img_size)
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][perspective_transform]

#### 4. Detect lane pixels and fit to find the lane boundary

I used the sliding window method to find the lane pixels positions and then fitted the positions with a 2nd order polynomial.
The algorithm details and hyper-parameters can be found in function `find_lane_pixels()`.
The fitted lines and quality metrics can be seen here:

![alt text][sliding_window_search_poly_lines]


#### 5. Determine the curvature of the lane and vehicle position with respect to center

I calculated the curvature using the fitted polynomial coefficients for real space coordinates (meters) using function:
```python
                                                                                                                          
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
                                                                                                                          
```
                                                                                                                                                                                                                                               
The lane center is derived from the mean of the lines third coefficient since it represents the intercept with x-axis.
Since the camera is firmly tighed to the center of the vehicle, the vehicle position can be assumed as the middle of the given image.                                                   

The distance of the vehicle from the center is then simply the diffence between these two. According to the signal we know if the vehicle is on the left or right of the center.

```python

xm_per_pix = 3.7/700 # meters per pixel in x dimension

center_lane_m = np.mean([left_line.current_fit[2], right_line.current_fit[2]]) * xm_per_pix        
center_vehicle_m = img_size[0]*0.5*xm_per_pix 

distance_from_center_m = center_vehicle_m - center_lane_m   

```                                                                                                                          
                                                                                                                        
#### 6. Visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position

I implemented this step in `draw_lane_area()` method of class `LaneDetector`:


![alt text][output]

---

### Pipeline (video)

#### 1. Project video [project_video] (fully working)
#### 2. Challenge video [challenge_video] (missing some lighting improvements)
#### 3. Harder challenge video [harder_challenge_video] (missing some curvature and lighting improvements)
---

### Discussion

This project took longer than antecipated, unfortunately I had to leave some optimizations/adjustments out.
The pipeline offers the sufficient processing to be robust for highway or speed way driving. 

Merits:
* Addresses all mandatory project rubric points;
* Detects and classifies lane 'sanity' (lane width and parallelity);
* Lane output is averaged to provide a smoother visualization;
* Rich debug visualization, displaying four images simultaneously 
                                                                                                                          
![alt text][debug_output]

                                                                                                                          
Improvements/Shortcomings:
* Processing speed is compromised since we are using the sliding window for every frame. I did use the search around polyline approach but was getting some mixed results and prefered to leave the "safer" (and costlier) option; 
**Suggestion:** Tweak the pipeline to use search around polylines and if it fails below certain quality metrics then use the sliding window.
  
* More robust methods for gradient and color transform under different ambient conditions (this is especially evident in the harder challenge video). 
**Suggestion:** Use an histogram of image colors to choose different modes of color and gradient thresholds

* Better tracking of hard turns, such as the infamous sharp turn in the mountain video. 
**Suggestion:** Stop propagating sliding windows if they touch the side-limits of the image and have a certain distance from the pixel position base

In other words, I would spend more time to make this pipeline pass all videos provided, however as it stands the pipeline will fail for mountain trips or some particular varying ambient conditions in speedways.


### Thank you for your attention.