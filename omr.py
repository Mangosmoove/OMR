import argparse
from PIL import Image, ImageDraw
from PIL import ImageFilter
import numpy as np
import math
# random number generator
import random
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

# for drawing text over image
from PIL import ImageDraw

import cv2

# ------------------------------------ Helper Funcs ------------------------------------ #
def conv_flipper(filter):
  newFilter = np.empty(filter.shape)
  for i in range(filter.shape[0]):
    for j in range(filter.shape[1]):
      newFilter[i][j] = filter[filter.shape[0]-1-i][filter.shape[1]-1-j]
  return newFilter

def border_corrector(resultIm, filter): #done, tested
    kern_h, kern_w = filter.shape[0] // 2 , filter.shape[1] // 2
    # left and right
    for i in range(kern_w): #for i in width of border
        for j in range(kern_h, resultIm.height-kern_h): # for j in height of area to fill
            resultIm.putpixel((i,j), resultIm.getpixel((kern_w+(kern_w-i)-1,j)))
            resultIm.putpixel((resultIm.width-i-1,j), resultIm.getpixel((resultIm.width-(kern_w + (kern_w-i)),j)))

    for i in range(kern_h):
        for j in range(resultIm.width):
            resultIm.putpixel((j, i), resultIm.getpixel((j, kern_h + (kern_h - i) - 1)))
            resultIm.putpixel((j, resultIm.height - i - 1), resultIm.getpixel((j, resultIm.height - (kern_h + (kern_h - i)))))
    return resultIm

# ------------------------------------ No 3 ------------------------------------ #
# Implementing convolution, adapted from sample_code1.ipynb in Module 8
def general_conv(image, kernel, scorer = False):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #kernel = cv2.cvtColor(kernel, cv2.COLOR_BGR2GRAY)
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
 
    output = np.zeros(image.shape)
 
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
 
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
 
    for row in range(image_row):
        for col in range(image_col):
            if not scorer:
              output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if scorer:
                output[row, col] = np.count_nonzero(kernel == padded_image[row:row+kernel_row, col:col+kernel_col]) /(kernel_col*kernel_row)*255
    plt.imshow(output)
    return output

# ------------------------------------ No 4 ------------------------------------ #
def sep_conv(image, filterx,filtery):
    filterx = filterx.T
    image = np.array(image)
    result = general_conv(image, filterx)
    result = general_conv(result, filtery)
    return result

# ------------------------------------ No 5 ------------------------------------ #
def detect_template(image, template):
  max_score = template.shape[0]*template.shape[1]
  score_image = general_conv(image, template, True)
  hits = []
  for i in range(score_image.shape[1]):
    for j in range(score_image.shape[0]):
      if score_image[j,i]/255 >= .75:
        hits.append((i, j, score_image[j,i]/255))
        for x in range(-1*template.shape[1]//2, template.shape[1]//2):
          for y in range(-1*template.shape[0]//2, template.shape[0]//2):
            if i+x > 0 and i+x < score_image.shape[1] and y+j > 0 and y+j < score_image.shape[0]:
              score_image[y+j, i+x] = 0
  plt.imshow(score_image)
  return hits

# ------------------------------------ No 6 ------------------------------------ #
# NOTE: The sobel matching function is TERRIBLY inefficient

# Implementing the Sobel operator, adapted from sample_code1.ipynb in Module 8
def sobel_edge_detection(image, filterx, filtery):
  '''
  Takes in an image and a vertical sobel filter, returns a gradient magnitude. Used for edge detection. Make sure the image is padded
  '''
  new_image_x = sep_conv(image, filterx, filtery)
  new_image_y = sep_conv(image, filtery, filterx)
  # first calculate the magnitude 
  gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
  # then you have to normalize the result to make sure the pixel values are valid
  gradient_magnitude *= 255.0 / gradient_magnitude.max()

  return gradient_magnitude

def sobel_matching(image, filter):
  '''
  Applies the matching algorithm to Image and Filter by first getting the edge maps of both and then applying the correct template. Returns scores for images
  '''
  sobel_v_x = np.array([[-1,0,1]])
  sobel_v_y = np.array([[1,2,1]])
  # Applying the sobel operator
  image = sobel_edge_detection(image, sobel_v_x, sobel_v_y)
  filter = sobel_edge_detection(filter, sobel_v_x, sobel_v_y)

  def score_func(i,j): # Used for calculating the scores later on
    r = 0
    

    for l in range(filter.shape[1]):
      for k in range(filter.shape[0]):                           
        r += filter[k][l]*d[i+k][j+l]
    return r

  # Precomputing the values for D(i,j)
  # TODO: DP implementation of D for faster testing
  print("Computing D(i,j)...")
  d = np.empty(image.shape)
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      print("\r", "{}/{} : {}/{}".format(i, image.shape[0], j, image.shape[1]), end="")
      min_val = float('inf')
      for a in range(image.shape[0]):
          for b in range(image.shape[1]):
            val = (0 if (image[a][b]) == 0 else float('inf')) + math.sqrt((i-a)**2 + (j-b)**2)
            min_val = min(val, min_val)
      d[i][j] = min_val
  scores = np.empty(image.shape)
  # Applying the scoring matrix, this will take a while...
  print("Scoring...")
  for i in range(image.shape[1]-filter.shape[1]):
    for j in range(image.shape[0]-filter.shape[0]):
      scores = score_func(i,j)
  print("Finished scoring")
  return scores
# ------------------------------------ No 7 ------------------------------------ #
# Calculate the Hough space for a range of r and theta
# change the parameterization to not use r and theta. We'll instead use p, for a specific pixel value, and n for the length of the staff
def hough_space(img, x_size, y_size, p_dim, p_max, n_dim, n_max, n_min): 
  '''
  Uses the hough space to identify the locations p and line distances n of all the possible staves in the image
  '''

  hough_space = np.zeros((p_dim,n_dim))

  sobel_v_x = np.array([[-1,0,1]])
  sobel_v_y = np.array([[1,2,1]])
  # Applying the sobel operator
  image = sobel_edge_detection(img, sobel_v_x, sobel_v_y)
  #image = np.asarray(img)

  for x in range(x_size):
    for y in range(y_size):
      if image[y,x] == 0: continue
      for n in range(n_min, n_dim):
        # Y is on the lines where p + n*i = y so p = y - n * i
        for j in range(1,5):
          p = y - n * j
          if p >= y_size or p < 0:
            continue
          hough_space[p,n] = hough_space[p,n] + 1 #(1 if image[p,x] > 0 else 0)
        # Now for the cases where Y is the first line
        for j in range(5):
          p = y + n * j
          if p >= y_size or p < 0:
            continue
          hough_space[y,n] = hough_space[y,n] + 1 #(1 if image[p,x] > 0 else 0)
  return hough_space

# This func shouldn't require any changes as it's just calculating local extrema in the hough space, invariant of the parameterization
def local_extrema(hough_space, neighborhood_size = 20, threshold = 140): 

  data_max = filters.maximum_filter(hough_space, neighborhood_size)
  maxima = (hough_space == data_max)

  data_min = filters.minimum_filter(hough_space, neighborhood_size)
  diff = (np.subtract(data_max, data_min) > threshold)
  maxima[diff == 0] = 0

  labeled, num_objects = ndimage.label(maxima)
  slices = ndimage.find_objects(labeled)

  x, y = [], []
  for dy,dx in slices:
    x_center = (dx.start + dx.stop - 1)/2
    x.append(x_center)
    y_center = (dy.start + dy.stop - 1)/2    
    y.append(y_center)

  return x, y

def staff_length(image, neighborhood_size = 80, threshold = 200):
  '''
  Given an image, identifies the staff locations and lengths for that image through the hough space.
  Returns the following:
  p: the y values of each staff
  n: the gap lengths for the corresponding staff in p
  h_s: the huber space
  n_hat: the estimate for n with the highest number of votes
  conf: the confidence in n_hat. the number of votes n_hat obtained
  '''
  #image = np.asarray(image)
  # First, get the hough space of the image
  img_shape = image.size
  x_max = image.width
  y_max = image.height

  # Assuming the image contains atleast a single staff, the maximum val for n would be 1/5 of the image
  n_min = 1
  n_max = y_max//5
  
  # these are just for graphing
  p_min = 0.0
  p_max = math.hypot(x_max, y_max)

  # p could be any value in y and n_dim was established earlier
  p_dim = y_max
  n_dim = n_max

  # Scoping magic happens now so I'm not passing in the n_max and p_max vars.
  h_s = hough_space(image, x_max, y_max, p_dim, p_max, n_dim, n_max, n_min)
  # Getting the local extrema, throwing away p as we are only concerned with n
  p, n = local_extrema(h_s, neighborhood_size = neighborhood_size, threshold = threshold)

  # Gets us the coordinate with the most votes, which is likely to contain the true value of n
  coords = (0,0)
  max_votes = 0
  for x in range(h_s.shape[0]):
    for y in range(h_s.shape[1]):
      if h_s[x,y] > max_votes:
        coords = (x,y)
        max_votes = h_s[x,y]
  conf = max_votes
  n_hat = coords[1]
  return (p, n, h_s, n_hat, conf)



# ------------------------------------ No 8 ------------------------------------ #
# The code for running the OMR

# Parsing the argument from the CLI
parser = argparse.ArgumentParser(description="Runs the OMR for the given input image")

parser.add_argument("image", nargs=1, metavar="img_path", type= str, help = "The path to the input image")

args = parser.parse_args()

image = Image.open(args.image[0]).convert('L')

# Get Filters
note_filter = Image.open("test-images/template1.png").convert("L")
qrest_filter = Image.open("test-images/template2.png").convert("L")
erest_filter = Image.open("test-images/template3.png").convert("L")

# Detect the Staves of the image
n, p, h_s, n_hat, conf = staff_length(image, neighborhood_size=175, threshold=150)
p = np.sort(p)

print(n_hat)
# Resizing Filters
ratio = n_hat / note_filter.height
# note_filter = note_filter.resize((int(note_filter.size[0]*ratio), int(note_filter.size[1]*ratio)))
# erest_filter = erest_filter.resize((int(erest_filter.size[0]*ratio), int(erest_filter.size[1]*ratio)))
# qrest_filter = qrest_filter.resize((int(qrest_filter.size[0]*ratio), int(qrest_filter.size[1]*ratio)))

print(np.asarray(image))
print(np.asarray(note_filter))

# Getting image positions
note_positions = detect_template(np.asarray(image), np.asarray(note_filter))
qrest_positions = detect_template(np.asarray(image), np.asarray(qrest_filter))
erest_positions = detect_template(np.asarray(image), np.asarray(erest_filter))

print(n)
print(p)
def get_pitch(y):
    # Find closest staff line
    min_dist = float('inf')
    treble_cleff = False # True if Treble Cleff, False if Base Cleff
    is_treble = False
    for location in p:
        is_treble = not is_treble
        if abs(y-location) < abs(min_dist):
            min_dist = y-location
            treble_cleff = is_treble
    
    #use distance from staff line to see how many lines away it is
    # if dist is positive, y is below the first line, if dist is negative, y is above the first line
    # 0 is just below the first line, aka a C note
    num_lines = int((min_dist * 2) // n_hat)

    # From the number of lines away we can calculate the index
    treble_note_chars = ["F","E","D","C","B","A","G"] #["Z"]*7
    base_note_chars = ["A","G","F","E","D","C","B"] #["B"]*7
    note_chars = treble_note_chars if treble_cleff else base_note_chars

    return note_chars[num_lines % 7]

print(len(note_positions))
print(len(qrest_positions))
print(len(erest_positions))

# Writing the results out to the txt and making the result picture
result_image = image.copy().convert("RGB")
result_draw = ImageDraw.Draw(result_image)
output = open('detected.txt', 'w')
for symb in note_positions:
    pitch = get_pitch(symb[1])
    output.write("<{row}><{col}><{height}><{width}><{type}><{pitch}><{conf}>\n".format(row=symb[0],col=symb[1],height=note_filter.height,width=note_filter.width,type="filled_note",pitch=pitch,conf=symb[2]))
    # drawing bounding box
    result_draw.rectangle([(symb[0] - note_filter.width // 2, symb[1] - note_filter.height // 2), (symb[0] + note_filter.width // 2, symb[1] + note_filter.height // 2)], outline="red")
    result_draw.text((symb[0] + note_filter.width, symb[1]), pitch, fill="red")
    #TODO: draw pitch
for symb in qrest_positions:
    pitch = "_"
    output.write("<{row}><{col}><{height}><{width}><{type}><{pitch}><{conf}>\n".format(row=symb[0],col=symb[1],height=qrest_filter.height,width=qrest_filter.width,type="eighth_rest",pitch=pitch,conf=symb[2]))
    # drawing bounding box
    result_draw.rectangle([(symb[0] - qrest_filter.width // 2, symb[1] - qrest_filter.height // 2), (symb[0] + qrest_filter.width // 2, symb[1] + qrest_filter.height // 2)], outline="green")
for symb in erest_positions:
    pitch = "_"
    output.write("<{row}><{col}><{height}><{width}><{type}><{pitch}><{conf}>\n".format(row=symb[0],col=symb[1],height=erest_filter.height,width=erest_filter.width,type="quarter_rest",pitch=pitch,conf=symb[2]))
    # drawing bounding box
    result_draw.rectangle([(symb[0] - erest_filter.width // 2, symb[1] - erest_filter.height // 2), (symb[0] + erest_filter.width // 2, symb[1] + erest_filter.height // 2)], outline="blue")
output.close()

result_image.save("detected.png")
