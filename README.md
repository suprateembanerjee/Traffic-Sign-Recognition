# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/graph.png "Visualization"
[image2]: ./examples/normalized.png "Normalized Image"
[image3]: ./examples/signs_test.png "German Traffic Signs"
[image4]: ./examples/signs_test_resized.png "Resized German Signs"
[image5]: ./examples/preprocessed.png "Preprocessed Signs"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/suprateem48/UdacitySDCND-P3/blob/main/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used native python libraries and numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data has been split across sets, so as to indicate proportion.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to shuffle the sets to avoid regions of similar data. This is intuitively and logically essential as it helps the model generalize better.

I have skipped Grayscaling for this project altogether. This is because I felt the model could use the red and blue colour signatures in traffic signs to better classify images. This intuition was supported by the model's performance (or lack thereof) when grayscaled images were used.

As a last step, I normalized the image data because it helped the image present itself in a clearer manner to the algorithm. The image became much 'flatter' and the pixels had a range from 0 to 1.

An example of my normalized images:
![alt text][image2]

I decided against augmenting the data because of two primary reasons. First, I had enough data to train a satisfactory model, and second, rotational augmentation could have harmed the model more than serving it: the dataset contains arrows, which are direction-specific to their labels. Having augmented data could have induced additional noise. The use of CNN's induced translation invariance by logic. Thus, I felt data augmentation would not be necessary.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Global Parameters (same across model):
* mu = 0
* sigma = 0.1
* bias: zeros
* Dropout keep_prob = 0.75

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 Stride, Valid Padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 Stride, Valid Padding, outputs 14x14x6 				|
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 Stride, Valid Padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 Stride, Valid Padding, outputs 5x5x16 				|
| Dropout		| outputs 5x5x16 	|
| Flatten     	| outputs 1x400 	|
| Fully Connected      	| outputs 1x120				|
| RELU					|												|
| Dropout		| outputs 1x120 	|
| Fully Connected      	| outputs 1x84				|
| RELU					|												|
| Fully Connected		| outputs 1x43        									|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer, with a learning rate of 0.001. I used a batch size of 512, which was reasonably fast to train, and trained for a total of 50 epochs. Having experimented with over 100 epochs, I found that after 50 epochs the model learns little and tends to overfit the data. A higher constant learning rate tends to become unstable once accuracy has reached satisfactory levels. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.96
* test set accuracy of 0.952

I have used the LeNet architecture to start with, as it was explained in the lectures. 

The accuracy of the vanilla LeNet was not satisfactory, barely touching the 89% accuracy mark. This was mostly because the model tended to overfit the data, and could not generalize well.

I adjusted the architecture by implementing Dropout. Some research on the Internet got me the idea that it is best applied after Fully Connected Layers, since they are the ones that tends to overfit. That is when I tried implementing a Dropout layer after the Fully Connected Layers towards the end of the model. However, I experimented by placing Dropout layers at multiple places, and eventually decided to place two of them, one after the second Convolutional layer, and one after the first FC layer, which seemed to perform the best.

Some of the important design choices were the use of Convolutional Layers and implementing Dropout. The Convolutional Layers induced multiple benefits in the task of classification, such as translational invariance. The Convolutional Layers are also less prone to overfit. Although a deeper model can be used, which uses many more Convolutional Layers, the present dataset is simple enough to not need them. Dropout was important because it helped the LeNet-based model generalize better and provide reasonably accurate results.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3]

These images were not resized to 32x32, and so could not be used for testing the model, which only works with 32x32 images. However, I processed these images in two steps:
* Crop the images so the signs are visible more clearly (Without this step, accuracy falls drastically due to too much lost information during compression)
* Resize them to 32x32 images.

These images looked like this:

![alt text][image4]

I normalized these images using the same technique as used for the dataset. These normalized images look like this:

![alt text][image5]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30km/h      		| 30km/h   									| 
| 50km/h     			| 30km/h 										|
| No entry					| No entry											|
| Stop	      		| Stop					 				|
| Keep right			| Keep right      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.952.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 20th cell of the Ipython notebook.

For the first image, the model is extremely sure that this is a 30km/h sign (probability of 1), and the image does contain a 30km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| 30km/h  									| 
| 0.0     				| 20km/h 										|
| 0.0					| 50km/h											|
| 0.0	      			| 80km/h					 				|
| 0.0				    | 70km/h      							|


For the second image, the model is extremely sure that this is a 30km/h sign (probability of 0.99), and the image does contain a 50km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| 30km/h  									| 
| 0.1     				| 50km/h 										|
| 0.0					| Keep right											|
| 0.0	      			| 80km/h					 				|
| 0.0				    | Turn right ahead      							|

For the third image, the model is extremely sure that this is a no entry sign (probability of 1), and the image does contain a no entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No entry  									| 
| 0.0     				| 20km/h 										|
| 0.0					| bicycle crossing											|
| 0.0	      			| Bumpy road					 				|
| 0.0				    | 30km/h      							|

For the fourth image, the model is extremely sure that this is a stop sign (probability of 1), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop  									| 
| 0.0     				| No entry 										|
| 0.0					| 60km/h											|
| 0.0	      			| Road work				 				|
| 0.0				    | 80km/h      							|

For the fifth image, the model is extremely sure that this is a keep right sign (probability of 0.99), and the image does contain a keep right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99        			| Keep right  									| 
| 0.01     				| 50km/h 										|
| 0.0					| Roundabout mandatory											|
| 0.0	      			| Turn left ahead					 				|
| 0.0				    | Yield      							|

### Observations

I found one aspect of the model particularly troubling. The model is overconfident. It's prediction confidence is extremely high, often assigning absolute 100% confidence to the chosen label and 0% to everything else, no matter the correctness of the prediction. This is problematic because the model does not seem to give enough consideration to other possibilities. This is visible not only in the wrongly predicted image (in the 5 test images), but also in the first two images, where they are speed limit signs which look similar in appearance. One would expect the model to not be so confident in at least those cases, but it is, and has assigned absolute probability of 1, even when it has wrongly classified. I hope to keep working on this issue, trying to make the model less confident of its prediction.
