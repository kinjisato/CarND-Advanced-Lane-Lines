## Writeup CarND-Advanced-Lane-Lines

### Kinji Sato  30th/June/2018
---

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

[image1]: ./output_images/undst_camera/camera_calibration.png "Undistorted"
[image2]: ./output_images/undst_test_images/test5_undist.jpg "Road Transformed"
[image3]: ./output_images/binary_imagds/test4_binary.jpg "Binary Example"
[image4]: ./output_images/binary_imagds/color_binary.png "binary, colored"
[image5]: ./output_images/code_description/transform.png "code transform"
[image6]: ./output_images/code_description/perspective_transform.png "code perspective transform"
[image7]: ./output_images/perspective_transform_images/sl1.png "perspective transform result"
[image8]: ./output_images/code_description/histgram.png "histgram"
[image9]: ./output_images/lanes/sliding_window.png "sliding window"
[image10]: ./output_images/lanes/selection_window.png "selection window"
[image11]: ./output_images/projected/1_projected.jpg "projected image"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

Here is a link to my [project code](https://github.com/kinjisato/CarND-Advanced-Lane-Lines/blob/master/P01_camera_calibration.ipynb).

#### 1. Briefly state how I computed the camera matrix and distortion coefficients. Examples of a distortion corrected calibration images are provided in the directory, ./output_images/undst_camera/

start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

Here is a link to my [project code](https://github.com/kinjisato/CarND-Advanced-Lane-Lines/blob/master/P02_test_image_pipeline.ipynb).

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

I corrected the images with using the `cv2.undistort()`, and its parameters `mtx` and `dist` were carried from the camera calibration I descrived above.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image.  Here's an example of my output for this step. I applied this steps to the undistorted images those generated above distortion-correction.

My code is,

![alt text][image5]

The points colored green are came from sobel x gradient (l channel), and points colored blue are came from color threshold of s channel.

![alt text][image4]

And then finally, when I combined two methods of gradient and color, I got following binary image.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`.
My code is,

![alt text][image6]


The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 25), img_size[1]],
    [(img_size[0] * 5 / 6) + 75, img_size[1]],
    [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 188, 720      | 320, 720      |
| 1141, 720     | 960, 720      |
| 700, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

First, I took the histgram of the half of warped binary image to find the peaks those might be lane lines. 

![alt text][image8]

And then, I applied sliding window methods to find the pixels those might be lane lines step by step in the image.

```python
for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = binary_warped.shape[0] - (window+1)*window_height
    win_y_high = binary_warped.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    # Draw the windows on the visualization image
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
    (0,255,0), 2) 
    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
    (0,255,0), 2) 
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
    (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
    (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:        
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
```

And, the polynomial lines are computed with using following function `np.polyfit()`, the order is 2nd.

```python
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```

`left_fit` and `right_fit` return x position from y position in the image.

```python
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
```

The result of sliding window method is,

![alt text][image9]

Once I got lane lines images, after that, I applied the other method (same as the lecture video).
The code is,

```python
# Assume you now have a new warped binary image 
# from the next frame of video (also called "binary_warped")
# It's now much easier to find line pixels!
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
margin = 100
left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
left_fit[1]*nonzeroy + left_fit[2] + margin))) 

right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
right_fit[1]*nonzeroy + right_fit[2] + margin)))  

# Again, extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]
# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
# Generate x and y values for plotting
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
```

With using this, I also could got `left_fit` and `right_fit` to compute polynomial curve. Following is the image after applying this method. (But, I applied this on the same image I used for sliding window, because I don't have "next flame" of this image.)

![alt text][image10]



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I had curvature of left and right lane lines. So, I computed the "center" of both two lines. I computed `center_fitx` by following code.

```python
# add center
center_fitx = (left_fitx + right_fitx) / 2.0
```

After I get `center_fitx`, the method is the same as computing the curvature of left and right lane.

`center_fitx[-1]` is the value of mid point of 2 lanes at bottom of image. 
So comparing this value with the midpoint of image (center of the car), I could get of offset value.


The method to compute the radius of curvature is the same as lecture video.

```python
y_eval = np.max(ploty)

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Fit new polynomials to x,y in world space
left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
center_fit_cr = np.polyfit(ploty*ym_per_pix, center_fitx*xm_per_pix, 2)
# Calculate the new radii of curvature
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
center_curverad = ((1 + (2*center_fit_cr[0]*y_eval*ym_per_pix + center_fit_cr[1])**2)**1.5) / np.absolute(2*center_fit_cr[0])
# Now our radius of curvature is in meters
#print(left_curverad, 'm', center_curverad, 'm', right_curverad, 'm')
print('Radius of carvature = {0} m'.format(int(center_curverad)))

# cal car offset
car_offset = (center_fitx[-1] - midpoint) * xm_per_pix
if car_offset < 0:
    print('Vehicle is {0:.2} m left of center'.format(np.absolute(car_offset)))
else:
    print('Vehicle is {0:.2} m right of center'.format(np.absolute(car_offset)))
```


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I applied all of the methods descriebed avobe, I got this image.

![alt text][image11]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
