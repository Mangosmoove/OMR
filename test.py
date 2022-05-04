from PIL import Image
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

# for file uploading

#
# I copied this to see if the naive implementation of the sobel operator would run faster on my local machine than it did on the google colab. It did not.
# -Trevor
#

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

def general_conv(image, filter):
    filter = conv_flipper(filter)
    result = Image.new('L', (image.width, image.height), color=0)  # creates blank canvas of same dimensions
    h, w = filter.shape[0] // 2, filter.shape[1] // 2  # get an integer value of starting position of kernel
    dp = 0.0  # dot product
    for i in range(h, image.height - h):
        for j in range(w, image.width - w):
            dp = 0.0
            # kernel shtuff
            for k in range(filter.shape[0]):
                for l in range(filter.shape[1]):
                    dp += image.getpixel((j-w+l, i-h+k)) * filter[k][l]
            result.putpixel((j, i), int(dp))
    result = border_corrector(result, filter)
    return result

def sep_conv(image, filterx,filtery):
    filterx = filterx.T
    result = general_conv(image, filterx)
    result = border_corrector(result, filtery)
    result = general_conv(result, filtery)
    result = border_corrector(result, filterx)
    return result

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

if __name__ == '__main__':
  # # No 6
  # Open image and convert to grayscale
  image = Image.open('test-images/music1.png').convert('L')
  filter = Image.open('test-images/template1.png').convert('L')
  plt.imshow(image, cmap='gray')
  plt.title("Image")
  plt.show()
  plt.imshow(filter, cmap='gray')
  plt.title("Filter")
  plt.show()
  print(image.size, filter.size)
  print(image.width, image.height)
  s = sobel_matching(image, filter)
  print(s)