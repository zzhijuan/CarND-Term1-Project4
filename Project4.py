
# coding: utf-8

# In[ ]:

#load chestboard for calibration
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg
import pickle
get_ipython().magic('matplotlib inline')

images = glob.glob('./camera_cal/calibration*.jpg')


# In[ ]:

objpoints = []
imgpoints = []

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) #?

fig, axes = plt.subplots(4, 5)
axes = axes.ravel()

for i, fname in enumerate(images):
    
    image = mpimg.imread(fname)
    objp = np.zeros((9*6, 3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
   
    if ret == True:
        
        imgpoints.append(corners)
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        img = cv2.drawChessboardCorners(image, (9, 6), corners2, ret)
        
        axes[i].axis('off')
        axes[i].imshow(img)


# In[ ]:

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
dst = cv2.undistort(mpimg.imread(images[0]), mtx, dist, None, mtx)

camera = {}
camera["mtx"] = mtx
camera["dist"] = dist
pickle.dump(camera, open('cameraCalibration.p', 'wb'))

plt.figure()
plt.imshow(mpimg.imread(images[0]))
plt.figure()
plt.imshow(dst)


# In[ ]:

#camera calibration
def cameraCalibration(img, mtx, dist):
    
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    
    return dst


# In[ ]:

#perspective transform
def unwrap(img):
        
        h, w = img.shape[0], img.shape[1]

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([[230, 670],[540, 470],[730, 470], [1050, 670]])
        dst = np.float32([[380, 720],[380, 0], [1100, 0],[1100, 720]])

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(img, M, (w, h))
        
        return warped, M, Minv


# In[ ]:

test_image = mpimg.imread('./test_images/test4.jpg')
plt.figure()
plt.imshow(test_image)

src = np.float32([[230, 670],[540, 470],[730, 470], [1050, 670]])
x = [src[0][0],src[1][0],src[2][0],src[3][0],src[0][0]]
y = [src[0][1],src[1][1],src[2][1],src[3][1],src[0][1]]
plt.plot(x, y, color = 'green')

test_cal = cameraCalibration(test_image, mtx, dist)
test_unwrap, transfer_mtx, Minv = unwrap(test_cal)

plt.figure()
plt.imshow(test_unwrap)


# In[ ]:

#sobel operation
def sobel_operation(img, orient, sobel_kernel, thresh):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, orient == 'x', orient == 'y', ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel_x)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sxbinary = np.zeros_like(gray)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return sxbinary


# In[ ]:

plt.figure()
sxbinary = sobel_operation(test_unwrap, 'x', 7, (30, 255))
plt.imshow(sxbinary, cmap='gray')


# In[ ]:

def mag_thresh(img, sobel_kernel, mag_thresh):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    return binary_output


# In[ ]:

plt.figure()
sxbinary = mag_thresh(test_unwrap, 7, (30, 255))
plt.imshow(sxbinary, cmap='gray')


# In[ ]:

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel, angle_thresh):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= angle_thresh[0]) & (absgraddir <= angle_thresh[1])] = 1
    # Return the binary image
    return binary_output


# In[ ]:

plt.figure()
sxbinary = dir_threshold(test_unwrap, 7, (0, 0.5))
plt.imshow(sxbinary, cmap='gray')


# In[ ]:

# Sobel Combined
ksize = 7
# Apply each of the thresholding functions
gradx = sobel_operation(test_unwrap, orient='x', sobel_kernel=ksize, thresh = (30, 255))
grady = sobel_operation(test_unwrap, orient='y', sobel_kernel=ksize, thresh = (30, 255))
mag_binary = mag_thresh(test_unwrap, sobel_kernel=ksize, mag_thresh = (15, 255))
dir_binary = dir_threshold(test_unwrap, sobel_kernel=ksize, angle_thresh = (0, 0.5))
combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

plt.figure()
plt.imshow(combined, cmap = 'gray')


# In[ ]:

# Define a function that thresholds the S-channel of HLS
def hls_select(img, thresh):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


# In[ ]:

plt.figure()
sxbinary = hls_select(test_unwrap, (200, 255))
plt.imshow(sxbinary, cmap='gray')


# In[ ]:

# Define a function that thresholds the S-channel of HLS
def hls_selectL(img, thresh):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,1]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


# In[ ]:

plt.figure()
sxbinary = hls_selectL(test_unwrap, (200, 255))
plt.imshow(sxbinary, cmap='gray')


# In[ ]:

# Define a function that thresholds the B-channel of LAB
def lab_select(img, thresh):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab_b = lab[:,:,2]
    if np.max(lab_b) > 175:
        lab_b = lab_b*(255/np.max(lab_b))
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    return binary_output


# In[ ]:

plt.figure()
sxbinary = lab_select(test_unwrap, (190, 255))
print(np.min(sxbinary))
plt.imshow(sxbinary, cmap = 'gray')


# In[ ]:

# Apply each of the thresholding functions
ksize = 7
# Apply each of the thresholding functions
gradx = sobel_operation(test_unwrap, orient='x', sobel_kernel=ksize, thresh = (60, 255))
grady = sobel_operation(test_unwrap, orient='y', sobel_kernel=ksize, thresh = (60, 255))
mag_binary = mag_thresh(test_unwrap, sobel_kernel=ksize, mag_thresh = (200, 255))
dir_binary = dir_threshold(test_unwrap, sobel_kernel=ksize, angle_thresh = (0, 0.4))
hlsinary = hls_selectL(test_unwrap, (200,255))
labinary = lab_select(test_unwrap, (190,255))

combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))|(hlsinary == 1)|(labinary == 1)] = 1
#((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | 

plt.figure()
plt.imshow(combined, cmap = 'gray')


# In[ ]:

# Implement sliding window and fit a polynomial
def his_polyfit(binaryimg, nwindows, margin, minpix):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binaryimg[np.int(binaryimg.shape[0]/2):,:], axis=0)#along the column; the bottom half image
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binaryimg, binaryimg, binaryimg))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Set height of windows
    window_height = np.int(binaryimg.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero  = binaryimg.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binaryimg.shape[0] - (window+1)*window_height
        win_y_high = binaryimg.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    return left_fit, right_fit, leftx, rightx, lefty, righty


# In[ ]:

ploty = np.linspace(0, combined.shape[0]-1, combined.shape[0])
left_fit, right_fit, leftx, rightx, lefty, righty = his_polyfit(combined, 10, 40, 40)
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

plt.figure()
plt.imshow(combined, cmap = 'gray')
plt.plot(left_fitx, ploty, color='red')
plt.plot(right_fitx, ploty, color='red')


# In[ ]:

def slide_window(img, margin, left_fit, right_fit):
    #sliding window
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit, leftx, rightx, lefty, righty


# In[ ]:

left_fit, right_fit, leftx, rightx, lefty, righty = slide_window(combined, 40, left_fit, right_fit)

# Generate x and y values for plotting
ploty      = np.linspace(0, img.shape[0]-1, img.shape[0])
left_fitx  = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

plt.figure()
plt.plot(leftx, lefty, 'o', color='green')
plt.plot(rightx, righty, 'o', color='blue')
plt.plot(left_fitx, ploty, color = 'red')
plt.plot(right_fitx, ploty, color = 'red')

plt.xlim(0, 1280)
plt.ylim(720, 0)
print(left_fit, right_fit)


# In[ ]:

def curver_fit(left_fit, right_fit, leftx, lefty, rightx, righty, img):
    
    # Generate some fake data to represent lane-line pixels
    lefty_eval = np.max(lefty)
    righty_eval = np.max(righty)
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720;# meters per pixel in y dimension
    xm_per_pix = 12.0*0.3048/800 # meters per pixel in x dimension
 
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    h = img.shape[0]
    ploty = np.linspace(0, h-1, h)
    y_eval = np.max(ploty)
    
    left_curverad = None
    right_curverad = None
    
    if len(leftx) != 0  and len(rightx) != 0:
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
        # Distance from center 
    if left_fit is not None and right_fit is not None:
        car_position = img.shape[1]/2
        h = img.shape[0]
        left_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
        right_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
        lane_center_position = (right_fit_x_int + left_fit_x_int) /2
        center_dist = (car_position - lane_center_position) * xm_per_pix

    return left_curverad, right_curverad, center_dist


# In[ ]:

left_curverad, right_curverad, center_dist = curver_fit(left_fit, right_fit, leftx, lefty, rightx, righty, combined)
    
# Now our radius of curvature is in meters
print(left_curverad, 'm', right_curverad, 'm')
print(center_dist, 'm')


# In[ ]:

def draw_lines(original_image, image_unwrap, left_fit, right_fit, Minv):
    
    # Create an image to draw the lines on
    binary_image = cv2.cvtColor(image_unwrap, cv2.COLOR_RGB2GRAY)
    warp_zero = np.zeros_like(binary_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
  
    # Generate x and y values for plotting
    ploty      = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx  = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,0), thickness=60)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(255, 0, 0), thickness=60)
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (binary_image.shape[1], binary_image.shape[0])) 
    
    # Combine the result with the original image
    result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)
    
    return result


# In[ ]:

def write_data(img, left, right, center_dist):
    new_img = np.copy(img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'Cuve radius: ' + '{:04.2f}'.format((left+right)/2) +'m'
    cv2.putText(new_img, text, (40, 100), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    direction = ''
    if center_dist > 0:
        direction = 'right'
    elif center_dist < 0:
        direction = 'left'
    abs_center_dist = abs(center_dist)
    text = '{:04.3f}'.format(abs_center_dist) + 'm ' + direction + ' away from image center'
    cv2.putText(new_img, text, (40, 150), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    return new_img


# In[ ]:

result = write_data(result, left_curverad, right_curverad, center_dist)
plt.imshow(result)
print('...')


# In[ ]:

#pipeline
def pipeline(test_image, mtx, dist):
   
    test_cal = cameraCalibration(test_image, mtx, dist)
    
    test_unwrap, transfer_mtx, Minv = unwrap(test_cal)
    
    gradx = sobel_operation(test_unwrap, orient='x', sobel_kernel=ksize, thresh = (60, 255))
    grady = sobel_operation(test_unwrap, orient='y', sobel_kernel=ksize, thresh = (60, 255))
    mag_binary = mag_thresh(test_unwrap, sobel_kernel=ksize, mag_thresh = (200, 255))
    dir_binary = dir_threshold(test_unwrap, sobel_kernel=ksize, angle_thresh = (0, 0.4))
    hlsinary = hls_selectL(test_unwrap, (200,255))
    labinary = lab_select(test_unwrap, (190,255))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))|(hlsinary == 1)|(labinary == 1)] = 1

        
    return combined, Minv, test_unwrap


# In[ ]:

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detect = False     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        
    def add_fit(self, fit):
        if fit is not None:
            if self.best_fit is not None:
                self.diffs = abs(fit - self.best_fit)
                if (self.diffs[0] > 0.001) or (self.diffs[1] > 0.001) or (self.diffs[2] > 20):
                    self.detect = False
                else:
                    self.detect = True
                    self.current_fit.append(fit)
                    self.best_fit = np.average(self.current_fit[len(self_current)-4:], axis = 0)                           
        else:
            self.detect = False
            if (len(self_current_fit) > 5):
                self.current_fit = self.current_fit[len(self_current)-4:]
                self.best_fit = np.average(self.current_fit, axis = 0)          


# In[ ]:

def process_image(image):
    with open('cameraCalibration.p', 'rb') as calibration:
        f = pickle.load(calibration);
    mtx, dist = f['mtx'], f['dist']
    result, Minv, image_unwrap = pipeline(image, mtx, dist)
    
    processed = image

    if left_line.detect is False or right_line.detect is False:
        left_fit, right_fit, leftx, rightx, lefty, righty = his_polyfit(result, 10, 40, 40) #fit by itself
    else:
        left_fit, right_fit, leftx, rightx, lefty, righty = slide_window(result, 40, left_fit, right_fit)
    
    if left_fit is not None and right_fit is not None:
        
        #Generate x and y values for plotting
        ploty      = np.linspace(0, image.shape[0]-1, image.shape[0])
        left_fitx  = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]     
        left_curverad, right_curverad, center_dist = curver_fit(left_fit, right_fit, left_fitx, ploty, right_fitx, ploty, result)
        
        if left_curverad is not None and right_curverad is not None:
            left_line.add_fit(left_fit)
            right_line.add_fit(right_fit)
            processed = draw_lines(image, image_unwrap, left_fit, right_fit, Minv)
            processed = write_data(processed, left_curverad, right_curverad, center_dist)
        else:
            left_fit = None
            right_fit = None
            left_fit.add_fit(left_fit)
            right_fit.add_fit(right_fit)
       
    return processed


# In[ ]:

from moviepy.editor import VideoFileClip

video_output1 = 'advancedline.mp4'
video_input1 = VideoFileClip('project_video.mp4')

left_line = Line()
right_line = Line()
print(left_line.detect)
processed_video = video_input1.fl_image(process_image)
get_ipython().magic('time processed_video.write_videofile(video_output1, audio=False)')

