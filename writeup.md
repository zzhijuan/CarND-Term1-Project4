
Advanced Lane Finding Project

The goals / steps of this project are the following:

Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
Apply a distortion correction to raw images.

Use color transforms, gradients, etc., to create a thresholded binary image.
Apply a perspective transform to rectify binary image ("birds-eye view").

Detect lane pixels and fit to find the lane boundary.

Determine the curvature of the lane and vehicle position with respect to center.

Warp the detected lane boundaries back onto the original image.

Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

0. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. Here is a template writeup for this project you can use as a guide and a starting point.

Answer: Done. 

1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Answer: 
Purpose of camera calibration is to undistort the image detected by camera. Camera's lenz always cauess either radial distortion or tangential distortion. Before distortion, we have to remember that camera maps 3D object into 2D image. Therefore we need to first figure out the camera matrix which converts 3D to 2D. Then derive the distortion coefficients to undistort the image. OpenCV provides a function to derive the camera matrix as well as distortion coefficients, which is called calirationCamera. Its input is a chessboard library. I loaded the chessboard libary which is provided by Udacity and created objects matrix as well as image corner points matrix. Then calibrationCamera automatically calculates the camera matrix and distortion coefficients. After that, I applied undistort function to undistort any future image. Here I assume all the images are measured by the same camera. 

Note the calivrationCamera is only used once and once the camera matrix as well as distortion coeffiients are determined, they are saved to cameraCalibraton.p file and directly called by future pipeline. 

Here is the lines showing the camera calibration and image undistort. 
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
dst = cv2.undistort(mpimg.imread(images[0]), mtx, dist, None, mtx)

Those are the lines showing that I saved these useful coefficients. 
camera = {}
camera["mtx"] = mtx
camera["dist"] = dist
pickle.dump(camera, open('cameraCalibration.p', 'wb'))

Check picture_1_a and picture_1_b from the output_images folder. Picture_1_a is an original chessboard image and picture_1_b is the undistort chessboard image.  

2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Answer:
After undistorting image, we need to work on the perspective correction. Perspective is a phenomenon where an object appears smaller father away. It is from a viewpoint like a camera. it gives the wrong distance estimation and should be correct for safety purpose. Then I applied getPerspectiveTransform function from openCV to help me implement perspective transform. The output is a new image without perspective. In order to complete the transform, we need to select a polygon from the original image and create a new polygan for the new image. THe function of getPerspectiveTransform is to find a transform which can convert the old polygon to the new one. Then the distortion caused by the perspective is corrected. Since then, we will work on the new unwraped image for line detection. 

Here is the src and dst maxtrix used in perpective transform:
src = np.float32([[230, 670],[540, 470],[730, 470], [1050, 670]])
dst = np.float32([[380, 720],[380, 0], [1100, 0],[1100, 720]])

Checking pictue_2 and picture_3 from the output_images folder. Picture_2 shows the stc polygon on the undistort image. Picture_3 shows the image with perspective transformation. 

3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

Answer:

For color transform, I used S channel from HLS and B channel from LAB. S channel is sensitive to color and not affected by light. It is a great way to enhance color lines from the background which may contains tree shadows or uneven light. B channel is sensitive to yellow line/dashed line. Based on all the test images, the B channel is able to detect yellow lines. So I combined these two channels together to create a thresholded binary image. 

For other methods, I use gradient/sobel methods to create theresholded binary images together with the above color transforms. I applied x sobel, y sobel, magnitude sobel, and direction angle operations.

Below is the threshold I used for each transform/method:

ksize = 7

gradx = sobel_operation(test_unwrap, orient='x', sobel_kernel=ksize, thresh = (60, 255))
grady = sobel_operation(test_unwrap, orient='y', sobel_kernel=ksize, thresh = (60, 255))
mag_binary = mag_thresh(test_unwrap, sobel_kernel=ksize, mag_thresh = (200, 255))
dir_binary = dir_threshold(test_unwrap, sobel_kernel=ksize, angle_thresh = (0, 0.4))
hlsinary = hls_selectL(test_unwrap, (200,255))
labinary = lab_select(test_unwrap, (190,255))

I notice that color trasnforms generally have stronger noise resistnace capability than sobel operations.

Images in the below are examples of binary images after applying:
picture_4: x sobel operation (y sobel transform is ignored here, but is is used in the later official processing);
picture_5: magnitude sobel operation;
picture_6: direction angle operation;
picture_7: x sobel operation, y sobel operation, magnitude sobel operation and direction angle operations;
picture_8: hls transform and thresholding S channel;
picture_9: lab transform and thresholding B channel;
picture_10: above all transforms/operations.

Check pictue_4 and picture_10 from the output_images folder.


4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Answer:

Two steps are involved in lane line detection. 

First, if the program is working on the first image or no previous lane fitting has been saved, I used the peaks in a histogram to detect lane lines. We calcualte the histogram of the bottom image with a height of 80 pixels and a width of 1280 pixels. Basd on the histogram and the middle point of the x axis, we detect local left and right peaks from the histogram. These two points are the bases. After that, I created two windows with a height of 80 pixels and a width of 80 pixels. The initial middle points of the subwindows are the bases. Any pixels within them are selected and appended to a corresponding list. Take the mean value of each list to update the bases. Then subwindows are moving along the direction of the new base from the bottom to the top. After each moving step, the bases have to be upodated. Once finish scanning the image, I obtain all qualified pixels. Note all pixels are grouped into left group and right group. Then I used curve fitting (order of 2) to identify the left and right lane lines based on selected pixels from the corresponding groups. 

Second, once the first image is fitted or any previous lane fitting has been saved, we don't need to repeat the same processing as above in any coming images. I selected all qualified pixels one time based on the previous fitting coefficients. Any pixels which are along the previous lane direction and within the predetermined window should be selected. After that, I refitted their positions and got the most recent fitting coefficients. 

Check pciture_11 and picture_12 from teh output_images folder. The first image picture_11 is after appling peaks histogram detection. The second image is after applying sliding window detection approach. 

5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Answer:
I used curve fitting to detect the radius of the curvature and the position of the vehicle with respect to center. Below is the lines:
left_curverad=((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad=((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5)/np.absolute(2*right_fit_cr[0])

The radius showing on the final video is the mean value of left and righ curvatures. One thing I have to emphasie is the scale I used to correct experiment to the real world, as below:
ym_per_pix = 30/720;#10.0*0.3048/100 # meters per pixel in y dimension
xm_per_pix = 12.0*0.3048/800 # meters per pixel in x dimension

I assumes 30 meter long and 3.7 meter wide lane line. 

In order to calcuate the shift with respect to the center, I calculated the intersection of x axis of both left lane and right lane. The difference between their mean value and the image middle point is the position of the vehicle with respect to center. 

6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Answer:
Picture_13 is the one that I plotted the fitting curves back to the original image. The curvertures and position are also included. 

7. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!)

Answer:

The complete pipeline is consisted of two parts: pipeline and process_image. 

Pipeline function implements camera calibration, undistort image, color tranform and sobel operations. Its output is color and gradient thresholded binary image, camera inverse matrix and undistorted 2D color image. 

Process_image implements line fitting and curverture calculations. It includes the peaks on the histrogram approach and refitting approach. 

A line class is created to store current and previous fitting coefficients of lanes. 
