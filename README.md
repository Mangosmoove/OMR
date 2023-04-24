# OMR System

# How to Run Program
##### After cd-ing into the correct directory
##### Terminal command for Windows: pythonFile imagePath(with correct extension)
##### omr.py test-images/music1.png
##### Terminal command for Mac/Linux: python3 omr.py test-images/music1.png
##### Running the command will result in the detect.txt and detect.png files to appear in the directory.

# Design Decisions and Assumptions
##### No note can be labeled as sharp or flat. All notes are assumed to be labeled as natural.
##### All the staves will have a slope of 0 and be perfectly straight lines. Therefore when assigning names to notes, we use the y-axis to serve as a measure of distance as the determinant. 
##### The templates will be the same exact size as the notes it encounters in the music sheet. Therefore we did not resize the template or the music sheet.


# Accuracy (Qualitatively and Quantitatively)
##### The algorithm can identify the clefs correctly and can correctly name the notes for each respective cleff. It can also place the box around the notes.


# Improvements
##### For number 6, dynamic programming could be implemented to decrease the time complexity. 
##### We could have also implemented a resizing method that would ensure that the template would be the appropriate size to convolve with.

# Coding Classes and Functions
## Introduction 
##### To solve the proposed problem of the assignment, we created two Python files: _test.py_ and _omr.py_. _test.py_ was more of a test file so the finalized code is in _omr.py_.


## Convolution
##### We created a convolution function called _general_conv_. This function takes in three parameters: the image, the filter that it should be convolved with, and whether or not that filter is a template. The function, after determining whether we are doing convolution or cross-correlation flips the filter horizontally and vertically to make the filter a convolutional kernel. Then, two for loops are used to traverse through the image space and calculate the dot product of the kernel and image. The results are then sent to a function called The results are then stored in an image variable that is returned. Initially, we did not include the third parameter but we implemented it for our template matching algorithm.

## Separable Convolution
##### For this, we created a function called _sep_conv_. It takes in three parameters: the image, an H_x filter, and an H_y filter. We make a call to _general_conv_ and send in the image with one filter at a time and store the results in a variable that is returned.


## Template Matching
##### We created a function _detect_template_ which takes in two parameters: the image adn the template. The template will contain the type of note that can be encountered in image (which is a music sheet). A cross-correlation is performed on the image with the template and the results being stored in _score_image_. Then we have a for loop that traverses over the image space and checks if, at a position in the image space, if there is a match with the template and image. We experimented with threshold values and used 0.7 as it helped detect more true positives. There was an issue of values exceeding the 255 limit, so normalization had to be implmeneted.


## Edge Map Scoring
##### For this problem, we created a function _sobel_edge_detector_ which takes in an image and two filters. We performed the separable convolution on the image with the two filters and then computed the gradient magnitude and normalized it after to ensure that the gradient magnitude did not exceed 255.
##### The function to calculate the edge match scores, we created a function called _score_func_ that was inside of another function called _sobel_matching_. This algorithm utilizes the _sobel_edge_detection_ function made earlier and calculates a score for the similarities. In our current implementation, it would take quite some time to compute the scoring matrix, but utilizing dynamic programming would definitely speed up the process.


## Hough Space
##### A brutal process to get to where it is now but it works! The function we used for it is _hough_space_. With this function, we are able to determine the staff heights which helps with pitch detection. It also determines the locations of the staves. This function required a lot of tuning on the extrema detector. Generally, the more staves/more complexity the music has, the higher the neighborhood. It would probably make sense to establish the neighborhood as a function of the size of the image, but we did not do that. This is an example of the Huber Space with local extrema indicating y-index on the y axis and gap length of the x axis
![Hough Space with Extrema](/hough_space_example.png)


## Pitch Detection
##### The function we created for this is called _get_pitch_. To determine what note is on the staff, we determine what clef the notes on the staves are in. Then, based on the distance the note is away from the nearest staff, the note can be labeled by using the index of where the note is relative to the staves. 
