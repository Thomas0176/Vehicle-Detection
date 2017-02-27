##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images. AND ####2. Explain how you settled on your final choice of HOG parameters. AND ####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I generate hog_features in my function:
	def get_hog_features

that method takes the img, orientation, pix_per_cell and cell_per_block as input. 

After playing around with the different parameters (mainly hist_feat, spatial_feat and hog_feat) I settled on the following: 
	color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
	orient = 9  # HOG orientations
	pix_per_cell = 8 # HOG pixels per cell
	cell_per_block = 2 # HOG cells per block
	hog_channel = 0 # Can be 0, 1, 2, or "ALL"
	spatial_size = (32, 32) # Spatial binning dimensions
	hist_bins = 32    # Number of histogram bins
	spatial_feat = False # Spatial features on or off
	hist_feat = False # Histogram features on or off
	hist_range=(0,256)
	hog_feat = True # HOG features on or off

Interestingly, I turned of the spatial_feat and hist_feat as they didn't seem to add better accuracy.
I also used RGB colorspace. 

I normalized the feature with  StandardScaler from sklearn.preprocessing. 



I started by reading in all the `vehicle` and `non-vehicle` images. I then shuffled them to ensure to not have time-series data implications. Since I had a lot of challenges with False Positives, I ended up using a 10:1 ration between non-vehicle and vehicle images for training. 


Note: I tried various things to overcome false positives, e.g. I tried class_weight parameter in my svc but it didn't really work... # svc = svm.SVC(kernel='linear', class_weight={0:.01, 1:.99})
So I ended up just using a normal svc 
'svc = svm.SVC(kernel='linear')''


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I played a lot around with the sliding window. I found that a big window size (128), a high overlap (~0.8) and higher threshold for the heatmap (2) worked well for me. Different (in particular smaller window sizes) just increased the false positive counts dramatially. Smaller overlaps made it hard to differentiate between false positives, since false positives usually are not as 'hot' as real cars, so it made it easier to filter out.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?
For result images please check: ./report_images folder

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


Besides the vehicle detection, I also added my previous implementation of lane detection into the video. 


See 'project_video_output.mp4'. 



####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are 4 frames and their corresponding heatmaps:

For result images please check: ./report_images folder

![alt text][image5]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
-- I still see a lot of false positives. So 
	 I could defintely improve the classifier. e.g. play with different kernels or play with class_weights which would reduce a 'car' prediction
	 I could also implement an frame-to-frame smoother (basically add a tracker over frames) to reduce false Positives

- currently my detector runs very slowly. So generating a hog_features per image rather than per window would increase the performance.

- I would be interested to use different detectors outside of SVM. E.g. using CNN would be very interesting to test.




