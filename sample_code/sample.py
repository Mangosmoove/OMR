from PIL import Image
from PIL import ImageFilter
import numpy as np
# random number generator
import random

# for drawing text over image
from PIL import ImageDraw


# spatial separable convolution (it's like a specific kind because there's this other thing called deep separable convolution but we can't use that one i think)
def sepConv(image, rF, cF):
    result = Image.new('RGB', (image.width, image.height), color=0)  # creates blank canvas of same dimensions
    temp = Image.new('RGB', (image.width, image.height), color=0)
    dp = 0.0  # dot product
    h, w = rF.shape[0], cF.shape[1]

    for i in range(image.width):
        for j in range(image.height):
            dp = 0.0
            # kernel shtuff with temp

        temp.putpixel((i, j), dp)

    for i in range(image.width):
        for j in range(image.height):
            dp = 0.0
            # another loop here for convolution stuff
        result.putpixel((i, j), dp)
    result = borderCorrector(image, result, filter)
    return result


# just a helper function
def makeKernel(size):
    corn = []  # kernel
    for i in range(size):
        row = []
        for j in range(size):
            row.append(1 / size)
        corn.append(row)
    return corn


# Hamming distance using convolution for template matching helper function
def general_conv(image, filter):
    result = Image.new('RGB', (image.width, image.height), color=0)  # creates blank canvas of same dimensions
    h, w = filter.shape[0] // 2, filter.shape[1] // 2  # get an integer value of starting position of kernel
    dp = 0.0  # dot product
    for i in range(h, image.height - h):
        for j in range(w, image.width - w):
            dp = 0.0
            # kernel shtuff
            for k in range(filter.shape[0]):
                for l in range(filter.shape[1]):
                    dp += image[i - h + k][j - w + l] * filter[k][l]
            result.putpixel((i, j), dp)
    result = borderCorrector(image, result, filter)
    return result


# also for template matching
def sobel():
    return 1


def borderCorrector(ogIm, resultIm, filter):
    h, w = filter.shape[0] // 2, filter.shape[1] // 2

    for i in range(h, -1, -1):  # make it inclusive of 0
        for j in range(ogIm.width):
            resultIm.putpixel((i, j), (ogIm.getpixel(i, j) + resultIm.getpixel(i+1, j)) / 2)

    for i in range(ogIm.width-h-1, ogIm.width):
        for j in range(ogIm.height):
            resultIm.putpixel((i, j), (ogIm.getpixel(i, j) + resultIm.getpixel(i-1, j)) / 2)

    for j in range(w, -1, -1):
        for i in range(ogIm.width):
            resultIm.putpixel((i, j), (ogIm.getpixel(i, j)+resultIm.getpixel(i, j+1))/2)

    for j in range(ogIm.height-w-1, ogIm.height):
        for i in range(ogIm.width):
            resultIm.putpixel((i, j), (ogIm.getpixel(i, j)+resultIm.getpixel(i, j-1))/2)
    return resultIm


# for rescaling images
def rescaling(image):
    return 1


if __name__ == '__main__':
    # load the grayscale image
    im = Image.open('first_photograph.png')

    # separable kernel for #4
    h_x = np.array([1, 0, 1])
    h_y = np.array([0, 3, 0])
    convolvedIm = sepConv(im, h_x, h_y)

    # 5
    template = Image.open('test-images/template1.png')
    kernel = makeKernel(3)
    general_conv(template, kernel)

'''
    #Check it's  width ,  height, and number of  color channels
    print('Image is %s pixels  wide. '%im.width)
    print('Image is %s  pixels high. '%im.height)
    print('Image mode  is %s.'% im.mode) #(8-bit pixels, black and white)
    
    #pixels are accessed via a (X,Y) tuple
    print('Pixel value is %s '% im.getpixel((10 ,10)))
    #pixels can be modified by specifying the coordinate and RGB value
    im.putpixel((10 ,10), 20)
    print('New pixel value is %s'%im.getpixel((10 ,10)))

# Create a new blank color image the same  size as the input
colorim = Image.new('RGB', (im.width , im.height), color =0)
#Image.draw(colorim)
# Loops over the new color image and 
# fills in brighter area that was white first grayscale image we loaded with red colors! 
# Basically we transformed the gray colored first ever photograph into a red colored one!
for x in range(im.width):
    for y in range(im.height):
        grayscale_val = im.getpixel((x,y))
        if ( grayscale_val >= 190):
            colorim.putpixel((x,y), (grayscale_val,0,0))              
        else:            
            colorim.putpixel((x,y), (grayscale_val, grayscale_val, grayscale_val))
    
colorim.show()
#colorim.save('output.png')

# adding text on top of the image at a random position

colorim_with_text = ImageDraw.Draw(colorim)
# generate a random X-coordinate and a random Y-coordinate
text_x_coord = random.randint(0 , colorim.width//2)
text_y_coord = random.randint(0 , colorim.height)
colorim_with_text.text((text_x_coord, text_y_coord),
                       'View from the Window at Le Gras',(255,255,255))
colorim.save('output_with_text.png')
'''
