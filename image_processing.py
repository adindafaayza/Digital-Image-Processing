import numpy as np
# import pandas as pd
# from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import math
from collections import Counter
from pylab import savefig
import cv2
import matplotlib
matplotlib.use('Agg')
import json


def get_image_values(image_path):
    try:
        img = Image.open(image_path)
        width, height = img.size
        pixel_values = []

        for y in range(height):
            for x in range(width):
                pixel = img.getpixel((x, y))
                pixel_values.append(pixel)

        return pixel_values, width, height
    except Exception as e:
        print(f"Error: {e}")
        return [], None, None

def create_puzzle(size):
    # Load the image
    img_path = "static/img/img_now.jpg"
    img = Image.open(img_path)

    # Get image dimensions
    width, height = img.size

    # Calculate dimensions for each puzzle piece
    piece_width = width // size
    piece_height = height // size

    # Initialize an array to store the puzzle pieces
    puzzle_pieces = []

    for i in range(size):
        for j in range(size):
            # Crop a piece from the image
            left = j * piece_width
            upper = i * piece_height
            right = left + piece_width
            lower = upper + piece_height

            puzzle_piece = img.crop((left, upper, right, lower))
            puzzle_pieces.append(puzzle_piece)

            # Save the puzzle piece
            puzzle_piece.save(f"static/img/puzzle_piece_{i}_{j}.jpg")

    return puzzle_pieces


def grayscale():
    if not is_grey_scale("static/img/img_now.jpg"):
        img = Image.open("static/img/img_now.jpg")
        img_arr = np.asarray(img)
        r = img_arr[:, :, 0]
        g = img_arr[:, :, 1]
        b = img_arr[:, :, 2]
        new_arr = r.astype(int) + g.astype(int) + b.astype(int)
        new_arr = (new_arr/3).astype('uint8')
        new_img = Image.fromarray(new_arr)
        new_img.save("static/img/img_now.jpg")


def is_grey_scale(img_path):
    im = Image.open(img_path).convert('RGB')
    w, h = im.size
    for i in range(w):
        for j in range(h):
            r, g, b = im.getpixel((i, j))
            if r != g != b:
                return False
    return True

def zoomin():
    if is_grey_scale("static/img/img_now.jpg"):
        img = Image.open("static/img/img_now.jpg")
        img_arr = np.asarray(img)
        
        if len(img_arr.shape) == 3:
            # Convert grayscale to 2D array
            img_arr = img_arr[:, :]

        new_size = (img_arr.shape[0] * 2, img_arr.shape[1] * 2)
        new_arr = np.full(new_size, 255, dtype=np.uint8)

        for i in range(len(img_arr)):
            for j in range(len(img_arr[i])):
                new_arr[2*i, 2*j] = img_arr[i, j]
                new_arr[2*i, 2*j+1] = img_arr[i, j]
                new_arr[2*i+1, 2*j] = img_arr[i, j]
                new_arr[2*i+1, 2*j+1] = img_arr[i, j]

        new_img = Image.fromarray(new_arr)
        new_img.save("static/img/img_now.jpg")
    else:
        img = Image.open("static/img/img_now.jpg")
        img = img.convert("RGB")
        img_arr = np.asarray(img)
        new_size = ((img_arr.shape[0] * 2),
                    (img_arr.shape[1] * 2), img_arr.shape[2])
        new_arr = np.full(new_size, 255)
        new_arr.setflags(write=1)

        r = img_arr[:, :, 0]
        g = img_arr[:, :, 1]
        b = img_arr[:, :, 2]

        new_r = []
        new_g = []
        new_b = []

        for row in range(len(r)):
            temp_r = []
            temp_g = []
            temp_b = []
            for i in r[row]:
                temp_r.extend([i, i])
            for j in g[row]:
                temp_g.extend([j, j])
            for k in b[row]:
                temp_b.extend([k, k])
            for _ in (0, 1):
                new_r.append(temp_r)
                new_g.append(temp_g)
                new_b.append(temp_b)

        for i in range(len(new_arr)):
            for j in range(len(new_arr[i])):
                new_arr[i, j, 0] = new_r[i][j]
                new_arr[i, j, 1] = new_g[i][j]
                new_arr[i, j, 2] = new_b[i][j]

        new_arr = new_arr.astype('uint8')
        new_img = Image.fromarray(new_arr)
        new_img.save("static/img/img_now.jpg")


def zoomout():
    if is_grey_scale("static/img/img_now.jpg"):
        img = Image.open("static/img/img_now.jpg")
        img_arr = np.asarray(img)
        
        if len(img_arr.shape) == 3:
            # Convert grayscale to 2D array
            img_arr = img_arr[:, :]

        x, y = img_arr.shape
        new_arr = np.zeros((int(x / 2), int(y / 2)), dtype=np.uint8)

        for i in range(0, int(x/2)):
            for j in range(0, int(y/2)):
                new_arr[i, j] = np.mean(img_arr[2*i:2*i+2, 2*j:2*j+2])

        new_img = Image.fromarray(new_arr)
        new_img.save("static/img/img_now.jpg")
    else:
        img = Image.open("static/img/img_now.jpg")
        img = img.convert("RGB")
        x, y = img.size
        new_arr = Image.new("RGB", (int(x / 2), int(y / 2)))
        r = [0, 0, 0, 0]
        g = [0, 0, 0, 0]
        b = [0, 0, 0, 0]

        for i in range(0, int(x/2)):
            for j in range(0, int(y/2)):
                r[0], g[0], b[0] = img.getpixel((2 * i, 2 * j))
                r[1], g[1], b[1] = img.getpixel((2 * i + 1, 2 * j))
                r[2], g[2], b[2] = img.getpixel((2 * i, 2 * j + 1))
                r[3], g[3], b[3] = img.getpixel((2 * i + 1, 2 * j + 1))
                new_arr.putpixel((int(i), int(j)), (int((r[0] + r[1] + r[2] + r[3]) / 4), int(
                    (g[0] + g[1] + g[2] + g[3]) / 4), int((b[0] + b[1] + b[2] + b[3]) / 4)))
        new_arr = np.uint8(new_arr)
        new_img = Image.fromarray(new_arr)
        new_img.save("static/img/img_now.jpg")


def move_left():
    img_path = "static/img/img_now.jpg"
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    if is_grey_scale(img_path):
        r, g, b = img_arr[:, :], img_arr[:, :], img_arr[:, :]
        r = np.pad(r, ((0, 0), (0, 50)), 'constant')[:, 50:]
        new_arr = r
    else:
        r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
        r = np.pad(r, ((0, 0), (0, 50)), 'constant')[:, 50:]
        g = np.pad(g, ((0, 0), (0, 50)), 'constant')[:, 50:]
        b = np.pad(b, ((0, 0), (0, 50)), 'constant')[:, 50:]
        new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_right():
    img_path = "static/img/img_now.jpg"
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    if is_grey_scale(img_path):
        r, g, b = img_arr[:, :], img_arr[:, :], img_arr[:, :]
        r = np.pad(r, ((0, 0), (50, 0)), 'constant')[:, :-50]
        new_arr = r
    else:
        r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
        r = np.pad(r, ((0, 0), (50, 0)), 'constant')[:, :-50]
        g = np.pad(g, ((0, 0), (50, 0)), 'constant')[:, :-50]
        b = np.pad(b, ((0, 0), (50, 0)), 'constant')[:, :-50]
        new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_up():
    img_path = "static/img/img_now.jpg"
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    if is_grey_scale(img_path):
        r, g, b = img_arr[:, :], img_arr[:, :], img_arr[:, :]
        r = np.pad(r, ((0, 50), (0, 0)), 'constant')[50:, :]
        new_arr = r
    else:
        r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
        r = np.pad(r, ((0, 50), (0, 0)), 'constant')[50:, :]
        g = np.pad(g, ((0, 50), (0, 0)), 'constant')[50:, :]
        b = np.pad(b, ((0, 50), (0, 0)), 'constant')[50:, :]
        new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_down():
    img_path = "static/img/img_now.jpg"
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    if is_grey_scale(img_path):
        r, g, b = img_arr[:, :], img_arr[:, :], img_arr[:, :]
        r = np.pad(r, ((50, 0), (0, 0)), 'constant')[0:-50, :]
        new_arr = r
    else:
        r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
        r = np.pad(r, ((50, 0), (0, 0)), 'constant')[0:-50, :]
        g = np.pad(g, ((50, 0), (0, 0)), 'constant')[0:-50, :]
        b = np.pad(b, ((50, 0), (0, 0)), 'constant')[0:-50, :]
        new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_addition():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img).astype('uint16')
    img_arr = img_arr+100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_substraction():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img).astype('int16')
    img_arr = img_arr-100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_multiplication():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    img_arr = img_arr*1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_division():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    img_arr = img_arr/1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")

def convolution(img, kernel):
    if len(img.shape) == 3 and img.shape[2] == 3:  # Color image
        h_img, w_img, _ = img.shape
        out = np.zeros((h_img-2, w_img-2), dtype=np.float64)
        new_img = np.zeros((h_img-2, w_img-2, 3))
        if np.array_equal((img[:, :, 1], img[:, :, 0]), img[:, :, 2]) == True:
            array = img[:, :, 0]
            for h in range(h_img-2):
                for w in range(w_img-2):
                    S = np.multiply(array[h:h+3, w:w+3], kernel)
                    out[h, w] = np.sum(S)
            out_ = np.clip(out, 0, 255)
            for channel in range(3):
                new_img[:, :, channel] = out_
        else:
            for channel in range(3):
                array = img[:, :, channel]
                for h in range(h_img-2):
                    for w in range(w_img-2):
                        S = np.multiply(array[h:h+3, w:w+3], kernel)
                        out[h, w] = np.sum(S)
                out_ = np.clip(out, 0, 255)
                new_img[:, :, channel] = out_
        new_img = np.uint8(new_img)
        return new_img

    elif len(img.shape) == 2:  # Grayscale image
        h_img, w_img = img.shape
        out = np.zeros((h_img-2, w_img-2), dtype=np.float64)

        for h in range(h_img-2):
            for w in range(w_img-2):
                S = np.multiply(img[h:h+3, w:w+3], kernel)
                out[h, w] = np.sum(S)

        out_ = np.clip(out, 0, 255)
        new_img = np.uint8(out_)
        return new_img

    else:
        raise ValueError("Unsupported image format")

def edge_detection():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img, dtype=np.int64)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")

def blur():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img, dtype=np.int64)
    kernel = np.array(
        [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def sharpening():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img, dtype=np.int64)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def customBlur(size):
    img = cv2.imread("static/img/img_now.jpg")
    blur = cv2.blur(img, ksize=(size, size))
    cv2.imwrite("static/img/img_now.jpg", blur)


def gaussianBlur(size):
    img = cv2.imread("static/img/img_now.jpg")
    gaussian_blur = cv2.GaussianBlur(src=img, ksize=(size, size), sigmaX=0)
    cv2.imwrite("static/img/img_now.jpg", gaussian_blur)

def gaussianBlur25():
    img = cv2.imread("static/img/img_now.jpg")
    gaussian_blur = cv2.GaussianBlur(src=img, ksize=(25, 25), sigmaX=0)
    cv2.imwrite("static/img/img_now.jpg", gaussian_blur)


def medianBlur(size):
    img = cv2.imread("static/img/img_now.jpg")
    medianblur = cv2.medianBlur(src=img, ksize=size)
    cv2.imwrite("static/img/img_now.jpg", medianblur)


def medianBlur25():
    img = cv2.imread("static/img/img_now.jpg")
    medianblur = cv2.medianBlur(src=img, ksize=(25))
    cv2.imwrite("static/img/img_now.jpg", medianblur)


def bilateralBlur():
    img = cv2.imread("static/img/img_now.jpg")
    bbf = cv2.bilateralFilter(src=img,d=9,sigmaColor=75,sigmaSpace=75)
    cv2.imwrite("static/img/img_now.jpg", bbf)


def zero_padding():
    img = cv2.imread("static/img/img_now.jpg")
    padded_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.imwrite("static/img/img_now.jpg", padded_img)

def lowFilterPass(size):
    img = cv2.imread("static/img/img_now.jpg")

    kernel_size = size
    random_kernel = np.random.rand(kernel_size, kernel_size)
    random_kernel /= np.sum(random_kernel)  

    lowFilterImage = cv2.filter2D(img, -1, random_kernel)
    cv2.imwrite("static/img/img_now.jpg", lowFilterImage)


def lowPassFilter():
    img = cv2.imread("static/img/img_now.jpg")
    # create the low pass filter
    lowFilter = np.ones((3,3),np.float32)/9
    # apply the low pass filter to the image
    lowFilterImage = cv2.filter2D(img,-1,lowFilter)
    cv2.imwrite("static/img/img_now.jpg", lowFilterImage)


def highFilterPass(size):
    img = cv2.imread("static/img/img_now.jpg")

    kernel_size = size
    random_kernel = np.random.randint(-1, 2, size=(kernel_size, kernel_size))
    random_kernel[1, 1] = (
        -np.sum(random_kernel) + 1
    )  

    highFilterImage = cv2.filter2D(img, -1, random_kernel)
    cv2.imwrite("static/img/img_now.jpg", highFilterImage)


def highPassFilter():
    img = cv2.imread("static/img/img_now.jpg")
    # create the high pass filter
    highFilter = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    # apply the high pass filter to the image
    highFilterImage = cv2.filter2D(img,-1,highFilter)
    cv2.imwrite("static/img/img_now.jpg", highFilterImage)


def bandFilterPass(size):
    img = cv2.imread("static/img/img_now.jpg")

    kernel_size = size
    random_kernel = np.random.randint(-1, 2, size=(kernel_size, kernel_size))

    while np.sum(random_kernel) == 0:
        random_kernel = np.random.randint(-1, 2, size=(kernel_size, kernel_size))

    bandFilterImage = cv2.filter2D(img, -1, random_kernel)
    cv2.imwrite("static/img/img_now.jpg", bandFilterImage)


def bandPassFilter():
    img = cv2.imread("static/img/img_now.jpg")
    # create the band pass filter
    bandFilter = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    # apply the band pass filter to the image
    bandFilterImage = cv2.filter2D(img,-1,bandFilter)
    cv2.imwrite("static/img/img_now.jpg", bandFilterImage)


def histogram_rgb():
    img_path = "static/img/img_now.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    if is_grey_scale(img_path):
        g = img_arr[:, :].flatten()
        data_g = Counter(g)
        plt.bar(list(data_g.keys()), data_g.values(), color='black')
        plt.savefig(f'static/img/grey_histogram.jpg', dpi=300)
        plt.clf()
    else:
        r = img_arr[:, :, 0].flatten()
        g = img_arr[:, :, 1].flatten()
        b = img_arr[:, :, 2].flatten()
        data_r = Counter(r)
        data_g = Counter(g)
        data_b = Counter(b)
        data_rgb = [data_r, data_g, data_b]
        warna = ['red', 'green', 'blue']
        data_hist = list(zip(warna, data_rgb))
        for data in data_hist:
            plt.bar(list(data[1].keys()), data[1].values(), color=f'{data[0]}')
            plt.savefig(f'static/img/{data[0]}_histogram.jpg', dpi=300)
            plt.clf()


def df(img):  # to make a histogram (count distribution frequency)
    values = [0]*256
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            values[img[i, j]] += 1
    return values


def cdf(hist):  # cumulative distribution frequency
    cdf = [0] * len(hist)  # len(hist) is 256
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i-1]+hist[i]
    # Now we normalize the histogram
    # What your function h was doing before
    cdf = [ele*255/cdf[-1] for ele in cdf]
    return cdf


def histogram_equalizer():
    img = cv2.imread('static\img\img_now.jpg', 0)
    my_cdf = cdf(df(img))
    # use linear interpolation of cdf to find new pixel values. Scipy alternative exists
    image_equalized = np.interp(img, range(0, 256), my_cdf)
    cv2.imwrite('static/img/img_now.jpg', image_equalized)


def threshold(lower_thres, upper_thres):
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    condition = np.logical_and(np.greater_equal(img_arr, lower_thres),
                               np.less_equal(img_arr, upper_thres))
    print(lower_thres, upper_thres)
    img_arr = np.asarray(img).copy()
    img_arr[condition] = 255
    new_img = Image.fromarray(img_arr)
    new_img.save("static/img/img_now.jpg")


def inverseFilter():
    # Load the input image
    img = cv2.imread("static/img/img_now.jpg")

    # Invert the colors by subtracting each channel from 255
    inverse_image = 255 - img

    # Save the inverse image
    cv2.imwrite("static/img/img_now.jpg", inverse_image)


# =================================================

# Freeman Chain Code

def calculate_similarity(chain_code1, chain_code2):
    # Calculate the similarity between two FCCs
    length1 = len(chain_code1)
    length2 = len(chain_code2)

    # Pad the shorter chain code with zeros to match the length of the longer chain code
    if length1 < length2:
        chain_code1 = chain_code1 + [0] * (length2 - length1)
    elif length1 > length2:
        chain_code2 = chain_code2 + [0] * (length1 - length2)

    return np.sum(np.array(chain_code1) == np.array(chain_code2))

def freeman_chain_code2(contour):
    chain_code = []
    prev_point = contour[0][0]
    
    for point in contour[1:]:
        x, y = point[0]
        dx = x - prev_point[0]
        dy = y - prev_point[1]
        
        # Calculate the angle in degrees
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Convert the angle to Freeman Chain Code (numerical values)
        code = round((angle % 360) / 45) % 8
        chain_code.append(code)
        
        prev_point = point[0]
    
    return chain_code

def goofy(lst):
    # Add 1 to each element
    for i in range(len(lst)):
        lst[i] += 1
        # Check if any element becomes 10 and change it to 0
        if lst[i] == 10:
            lst[i] = 0

def create_image_from_chain_code(chain_code, image_size):
    # Initialize a white canvas
    image = np.zeros(image_size, dtype=np.uint8)
    x, y = image_size[0] // 2, image_size[1] // 2  # Start from the center

    # Define the Freeman Chain Code directional offsets
    offsets = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

    for code in chain_code:
        dx, dy = offsets[code]
        x, y = x + dx, y + dy
        # Ensure the coordinates are within the image bounds
        x = max(0, min(x, image_size[0] - 1))
        y = max(0, min(y, image_size[1] - 1))
        # Set the pixel at the new position to white
        image[x, y] = 255

    return image

def read_binary(input_image_path):
    # Load the image
    image = input_image_path
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    if image is None:
        print("Error: Unable to read the image.")
        return

    # Apply Otsu's thresholding
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    return binary_image

def get_contour(binary_image):
    
    contours, _ = cv2.findContours(
        binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def FCC(binary_image):
    contours = get_contour(binary_image)
    for contour in contours:
        result = freeman_chain_code2(contour)
        print(f'result {result}')
        return result
        
def invert_binary_image(binary_image):
    # Invert the binary image using the NOT operation
    inverted_image = cv2.bitwise_not(binary_image)
    return inverted_image

def recognize_digit(input_chain_code, knowledge_base):
    best_match = None
    best_similarity = 0

    for digit, data in knowledge_base.items():
        for stored_chain_code in data["FCC_8_Directions"]:
            for input_code in input_chain_code:
                similarity = calculate_similarity(input_code, stored_chain_code)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = digit

    return best_match

def recognize_digits_in_images():
    
    img = cv2.imread("static/img/img_now.jpg")

    # Load the knowledge base from the JSON file
    with open("static/data/knowledge_base.json", "r") as file:
        knowledge_base = json.load(file)

    # Read and process the image
    binary_image = invert_binary_image(read_binary(img))
    contours = get_contour(binary_image)

    recognized_digits = []

    # Create a dictionary to map the position of contours to their FCC
    contour_position_mapping = {}

    for contour in contours:
        # Calculate FCC in 8 directions for each contour
        fcc_8_directions = freeman_chain_code2(contour)

        # Determine the position (e.g., x-coordinate) of the contour
        position = contour[0][0][0]  # Assuming x-coordinate is at index 0

        contour_position_mapping[position] = fcc_8_directions

    # Sort the contours based on their positions
    sorted_contours = sorted(contour_position_mapping.items())

    for _, fcc_8_directions in sorted_contours:
        recognized_digit = recognize_digit([fcc_8_directions], knowledge_base)

        if recognized_digit is not None:
            recognized_digits.append(recognized_digit)

    if recognized_digits:
        result = f"{''.join(map(str, recognized_digits))}"
    else:
        result = f"Digit recognition failed for {img}. No matching digits found in the knowledge base."

    return result