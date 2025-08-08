import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

# read image
# imread, imshow, imwrite

def read_images():
    imgPath = "/Users/riyaraut/Downloads/IMG_3473.JPG"
    img = cv.imread(imgPath) # read
    cv.imshow("img", img) # show
    cv.waitKey(0) # exit with any key

def writeImage():
    imgPath = "/Users/riyaraut/Downloads/IMG_3473.JPG"
    img = cv.imread(imgPath) # read
    op = "/Users/riyaraut/Downloads/outputed_now.JPG"
    cv.imwrite(op,img)
    
    
    
    
## DEALING WITH VIDEOS - webcam and then files

def videoFromWebcam():
    capture = cv.VideoCapture(0) # use the default camera by saying zero
    
    if not capture.isOpened():
        print("stopped by some other program")
        exit()
    
    while True:
        returnn,  frame = capture.read()
        if returnn:
            cv.imshow("wecammm", frame)
        
        # Exit the loop and close window if 'q' is pressed
        if cv.waitKey(1) == ord("q"):
            break

    # Release the webcam and close all OpenCV windows
    capture.release()
    cv.destroyAllWindows()
    
# read from videos
def videoFromFile():
    video_path = "videoFromWebcam"
    capture = cv.VideoCapture(video_path)
    
    while capture.isOpened():
        returnn, frame = capture.read()
        cv.imshow("video",frame )
        delay = int(1000/60)
        if cv.waitKey(delay)== ord("q"):
            break

# read from the webcam and save
def writeVideoToFile():
    capture = cv.VideoCapture(0) # use the default camera by saying zero
    
    fourcc = cv.VideoWriter_fourcc(*"XVID")
    outputPath = "/Users/riyaraut/Downloads/webcam.avi"
    out = cv.VideoWriter(outputPath, fourcc, 20.0,(640,480))
    
    if not capture.isOpened():
        print("stopped by some other program")
        exit()
    
    while True:
        returnn,  frame = capture.read()
        if returnn:
            out.write(frame)
            cv.imshow("wecammm", frame)
        
        # Exit the loop and close window if 'q' is pressed
        if cv.waitKey(1) == ord("q"):
            break

    # Release the webcam and close all OpenCV windows
    capture.release()
    cv.destroyAllWindows()


def seePixelPlot():
    imgPath = "/Users/riyaraut/Downloads/IMG_3473.JPG"
    img = cv.imread(imgPath)
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)    # when opencv reads it reads in BGR so lets convert to rgb for matplotlib
    
    plt.figure()
    plt.imshow(imgRGB)
    plt.show()


def readAndWritePixel():
    imgPath = "/Users/riyaraut/Downloads/IMG_3473.JPG"
    img = cv.imread(imgPath)


    # Print original BGR pixel at (100, 100)
    print("Original BGR pixel at (100, 100):", img[100, 100])

    # Change pixel at (100, 100) to pure red in BGR format
    img[100, 100] = [0, 0, 255]  # Blue=0, Green=0, Red=255

    # Convert to RGB for matplotlib
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Show the modified image
    plt.figure()
    plt.imshow(imgRGB)
    plt.title("Pixel at (100,100) changed to red")
    plt.show()

    
    
# rgb channels

def pureColors():
    zeros = np.zeros((100,100)) # zero matrix
    ones = np.ones((100,100)) # one matrix
    
    bgImg = cv.merge((255*ones,zeros, zeros))
    
    # plt.figure()
    # plt.subplot(231)
    plt.imshow(bgImg)
    
    plt.show()

def bgrChannelGrayScale():
    imgPath = "/Users/riyaraut/Downloads/IMG_3473.JPG"
    img = cv.imread(imgPath)
    
    b,g,r = cv.split(img)
    
    plt.figure()
    plt.imshow(b, cmap="gray") # colour map is gray
    plt.show()

def bgrColour():
    imgPath = "/Users/riyaraut/Downloads/IMG_3473.JPG"
    img = cv.imread(imgPath)
    b,g,r = cv.split(img)
    zeros = np.zeros_like(b)
    
    zeros = np.zeros_like(b)
    b_img = cv.merge((b, zeros,zeros))
    plt.figure()
    plt.imshow(b_img)
    plt.show()
    
    
    
    

def shapes_colours():
    imgPath = "/Users/riyaraut/Desktop/all-shapes-and-colors-v-2/test_dataset/img_7.png"
    img = cv.imread(imgPath)

    b, g, r = cv.split(img)
    zeros = np.zeros_like(b)

    # Blue image: Blue channel + zeros for green and red
    b_img = cv.merge((b, zeros, zeros))

    # Convert BGR to RGB for matplotlib display
    b_img_rgb = cv.cvtColor(b_img, cv.COLOR_BGR2RGB)

    # Display
    plt.figure()
    plt.imshow(b_img_rgb)
    plt.title("Only Blue Channel Visualized")
    plt.axis("off")
    plt.show()

        
    
    
    
    
    
    
    
# gray scale images 

# WHY DO WE NEED GRAY SCALE IMAGES?
# - REDUCE THE AMOUNT OF DATA FROM 3 CHANNLES TO JUST ONE CHANNLE
# - PREPROCESSING TECHNIQUE

# HOW TO MAKE GRAY SCALE IMAGES
# WE TAKE THE SCALE VALUE OF THE BLUE CHANNEL - 0.114 AND GREEN - 0.587, RED - 0.299
# SCALE VALUES ABOVE ARE FOR OPENCV

def grayscale():
    root = os.getcwd()
    imgPath = "/Users/riyaraut/Desktop/all-shapes-and-colors-v-2/test_dataset/img_7.png"
    img = cv.imread(imgPath)
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    cv.imshow("gray",imgGray)
    cv.waitKey(0)
    
    
"""
HSV Color Space — Used for Color-Based Segmentation

H → Hue (type of color)
    - Represents the actual color (e.g., red, green, blue)
    - Range in OpenCV: 0 to 179 (circular scale)
    - Think of it as the angle around a color wheel or cone

S → Saturation (intensity of color)
    - 0 = gray (washed out), 255 = full vivid color
    - Represents how pure or intense the color is

V → Value (brightness) ---> darkness of the coloiur
    - 0 = black (no brightness), 255 = full brightness
    - Controls how light or dark the color appears

HSV can be visualized as a cone or cylinder:
- Hue is the angle (around the base of the cone)
- Saturation is the distance from the center (radius)
- Value is the height (brightness)

HSV is preferred over RGB for color detection tasks because:
- It separates color from lighting (COLOUR THAT IS HUE FROM LIGHTING )
- It makes it easier to define and segment specific colors


# here they normalize the value by 255
and compute the cmax and cmin and difference


"""


def hsvimg():
    imgPath = "/Users/riyaraut/Desktop/all-shapes-and-colors-v-2/test_dataset/img_7.png"
    img = cv.imread(imgPath) # read in bgr format
    
    # BGR--> RGB--> HSV ---> bounds --> mask(inrange)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
    
    lower_bounds = np.array([100, 90, 40])
    upper_bounds = np.array([130,255, 255])
    
    mask = cv.inRange(img_hsv,lower_bounds, upper_bounds)
    
    cv.imshow("hw",mask)
    cv.waitKey(0)
    
    

# image resizing 
# reduce the data and prepocessing --> impove resoluition
# -->  work?
# Interpolation is the process of estimating unknown pixel values when resizing an image.
# Nearest Neighbor -->cv.INTER_NEAREST	Picks the nearest pixel. Fastest, blocky edges. Not smooth.
# Bilinear --> cv.INTER_LINEAR	Uses the 4 nearest neighbors in a grid. Smoother than nearest.
# Bicubic---> cv.INTER_CUBIC	Uses 16 neighbors. Smoother than bilinear. Slower.
# Area ---> cv.INTER_AREA	Averages pixels in an area. Best for shrinking images.
# 1d nearest neighbour, 2d nearest neighbour , linear, bilinear, cubic, bicubic 

# nearest(rough)--> linear--> cubic ---> area --> lanzoc (worst to betetr)

def resize():
    img = cv.imread("/Users/riyaraut/Desktop/all-shapes-and-colors-v-2/test_dataset/img_7.png")
    
    resized_area = cv.resize(img,(200,200),interpolation=cv.INTER_AREA)
    resized_linear= cv.resize(img,(200,200),interpolation=cv.INTER_LINEAR)
    
    cv.imshow("jsj", resized_area)
    cv.waitKey(0)
    
    
    
import cv2 as cv
import matplotlib.pyplot as plt

"""
📊 HISTOGRAM EQUALIZATION - EXPLAINED

Histogram equalization is a contrast enhancement technique that redistributes pixel intensities
in an image to use the full dynamic range (0–255 in 8-bit images).

Why we use it:
- To make dark areas darker and bright areas brighter
- To improve visibility of structures in low-contrast images
- To help ML models learn better by enhancing features

TYPES:
1. Global Histogram Equalization (GHE)
   - Equalizes contrast across the **entire image**
   - May over-enhance noise or wash out local details

2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Divides image into **tiles**
   - Applies equalization **locally**
   - **Limits contrast** using a `clipLimit`
   - **Interpolates between tiles** to avoid visible boundaries
   - Best for medical images like X-rays
"""
def histogram_equalization():
    # Step 1: Load a grayscale image
    img = cv.imread("/Users/riyaraut/Desktop/potrait.png", cv.IMREAD_GRAYSCALE)

    # 2. GLOBAL HISTOGRAM EQUALIZATION
    # Spreads pixel values across the entire image uniformly
    global_eq = cv.equalizeHist(img)

    # 3. CLAHE HISTOGRAM EQUALIZATION
    # Applies histogram equalization in small tiles + limits over-contrast
   # SYTNTAXXX is differnet here
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_eq = clahe.apply(img)

    # 4. Show all three side by side
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(global_eq, cmap='gray')
    plt.title("Global Histogram Equalization")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(clahe_eq, cmap='gray')
    plt.title("CLAHE Equalization")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    
 # 2D convolution like convolutional neural network is about-> feature extratcion
 
"""
📌 2D Convolution — Everything You Need to Know

🔹 What is 2D Convolution?
2D Convolution is a fundamental image processing operation used to extract features,
detect edges, blur, sharpen, and more. It involves sliding a small matrix (called a kernel or filter)
across the image and computing a dot product at each location.

🔹 How does it work?
1. Take a small square matrix (e.g., 3x3 or 5x5), called a kernel.
2. Place the kernel on top of the image at a specific position.
3. Multiply each value in the kernel with the corresponding pixel value under it.
4. Sum all the multiplied values — this is the dot product.
5. Replace the center pixel of that region in the **output image** with this sum.
6. Slide the kernel across the image and repeat the process.
Image patch:
[  4,  6,  5  ]
[  2, 10,  8  ]
[  1,  7,  3  ]

Kernel:
[  0, -1,  0 ]
[ -1,  5, -1 ]
[  0, -1,  0 ]

Dot product = (0*4) + (-1*6) + (0*5) + (-1*2) + (5*10) + (-1*8) + (0*1) + (-1*7) + (0*3)
            = 0 -6 + 0 -2 + 50 -8 + 0 -7 + 0 = 27

So, the center pixel (10) in the **output image** becomes 27.

🔹 Why do we do this?
Because different kernels perform different tasks:
- Blur/Smooth → Averages surrounding pixels to reduce noise.
- Sharpen → Enhances edges by emphasizing intensity changes.
- Edge Detection → Highlights regions where pixel values change rapidly.
- Emboss → Creates 3D-like shadow effects.
- Feature Extraction (CNNs) → Learns patterns like edges, curves, textures, etc.

🔹 Important Concepts:
- **Stride**: How many pixels the kernel moves at each step (usually 1).
- **Padding**: Adding borders around the image to control output size. 
  ('valid' = no padding, 'same' = output same size as input)
- **Output Size**: Depends on image size, kernel size, stride, and padding.

🔹 In Libraries (like OpenCV or PyTorch):
Convolution is often implemented using optimized matrix operations.
You can apply your own kernels using:
    - `cv2.filter2D()`
    - `scipy.signal.convolve2d()`
    - Or define custom filters in deep learning models (CNNs).

"""

    
def covolution():

    # Load image in grayscale for simplicity (you can use color too)
    img = cv.imread("/Users/riyaraut/Desktop/potrait.png")

    # Define a 3x3 sharpening kernel (you can try others below)
    # kernel = np.array([[0, -1, 0],
    #                 [-1, 5, -1],
    #                 [0, -1, 0]])

    # 2. Edge Detection (Sobel-like)
    kernel = np.array([[-1, -1, -1],
                    [-1,  8, -1],
                    [-1, -1, -1]])
    """
    # Other sample kernels:

    # 1. Blur (Averaging) Kernel
    kernel = np.ones((3, 3), np.float32) / 9

    # 2. Edge Detection (Sobel-like)
    kernel = np.array([[-1, -1, -1],
                    [-1,  8, -1],
                    [-1, -1, -1]])

    # 3. Emboss
    kernel = np.array([[-2, -1, 0],
                    [-1,  1, 1],
                    [ 0,  1, 2]])
    """

    # Apply the convolution using OpenCV
    convolved = cv.filter2D(img, -1, kernel) # one line of code
    # This means the output image will have the same depth as the input. For example, if the input is 8-bit (uint8), the output will also be 8-bit.
    plt.imshow(convolved)
    plt.show()


"""
🧠 Median Filtering (Non-linear filter):

- Instead of averaging pixel values like in average filtering,
  it replaces the central pixel with the **median** of the surrounding pixels.
- EDGE PRSERVATOION
- Best for removing **salt-and-pepper noise** (random black and white dots). the shapes task had dots in it which were salt and pepper noises
- Preserves edges better than average filtering, which tends to blur them.
✅ Average filtering → Linear, because it’s a weighted sum of neighbors.
❌ Median filtering → Non-linear, because it uses the median, not a sum.
"""


def medianBlur():
    img_path = "/Users/riyaraut/Desktop/all-shapes-and-colors-v-2/test_dataset/img_51.png"
    img = cv.imread(img_path)
    

# Apply Median Filtering with kernel size 5
# Kernel size must be an odd number: 3, 5, 7, etc.
    median_img = cv.medianBlur(img,5)
    
    plt.imshow(img)
    plt.title("orignal")
    
    plt.imshow(median_img)
    plt.title("median_blur")
    
    plt.show()

    
    
# gaussian filtering follows a gaussian distribution curve a bell curve
# sigma value controls how tall or short the kernal or the bell how fat or skinny it is going to be 

    """
    🧠 Gaussian Filtering:
    - Applies a Gaussian blur to reduce noise and detail in the image.
    - Uses a kernel that follows a 2D Gaussian distribution.
    - Pixels near the center of the kernel are given more weight (smoother blur).
    - More effective and natural than average blur.

    Syntax:
    cv.GaussianBlur(src, ksize, sigmaX)

    🔹 src: input image
    🔹 ksize: kernel size, must be odd and positive (e.g., (3,3), (5,5))
    🔹 sigmaX: standard deviation in X direction (sigmaY auto-computed if 0)
    """

def gausssian_blur():
    img_path = "/Users/riyaraut/Desktop/all-shapes-and-colors-v-2/test_dataset/img_51.png"
    img = cv.imread(img_path)
    
    g_b = cv.GaussianBlur(img, (5,5), 9)
    
    plt.imshow(g_b)
    plt.show()    
    
    
"""
🧠 What is Image Thresholding?

- Thresholding is a **segmentation technique** that turns a colour image into a binary image (black & white).
- It checks each pixel value: converts it to either 0 or 255
    - If the pixel intensity is **greater than a threshold**, set it to **white (255)**.
    - Otherwise, set it to **black (0)**.

📌 Why is it used?
- Simplifies the image for analysis, object detection, OCR, etc.
- Reduces computational complexity by converting to binary.
- Useful in situations where the **foreground and background** have clear intensity differences.

🔍 How does it work?
- You choose a **threshold value (T)**.
- For each pixel:
    - If pixel > T → pixel = 255 (white)
    - Else → pixel = 0 (black)


🏥 Where is it used? - segmentation, extraction of features, object detection
- OCR (Optical Character Recognition) for scanned text
- Tumor or cell segmentation in medical imaging
- Industrial defect detection
- Barcode/QR code scanning
- Background removal for object detection
- Simplifying input for shape analysis, counting, etc.


USE HISTOGRAM TO CHOOSE THE THRESHOLD VALUE
If there are two distinct peaks, choose a value between them (this separates background and foreground).
Plot the grayscale histogram.

If there are two distinct peaks, choose a value between them (this separates background and foreground).

"""



def threshold():
    img_path = "/Users/riyaraut/Desktop/all-shapes-and-colors-v-2/test_dataset/img_51.png"

    # Read the image in color
    img = cv.imread(img_path)

    # 🎨 Step 2: Convert the image to grayscale
    # Thresholding requires a single channel (grayscale) image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 🚪 Step 3: Apply Binary Thresholding
    # Any pixel > 127 becomes 255 (white), else 0 (black)
    thresh_val, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    
    # 🖼️ Step 4: Display the thresholded image
    plt.imshow(binary, cmap='gray')  # Show as grayscale
    plt.title(f"Binary Thresholding at 127")
    plt.axis('off')
    plt.show()


"""
🧠 Image Gradient Methods: Tkes 
An image gradient is a **measure of change in pixel intensity**
— basically, how fast and in which direction the colors or brightness in an image are changing.


📸 Imagine a grayscale image:
- Black (0) → White (255)
- If the brightness suddenly jumps from dark to light, that’s a **strong edge**
- The gradient helps detect those changes.

🔁 Technically:
- A gradient is a **derivative** — it calculates the rate of change of pixel values.
- In 2D images, we compute this change along:
    ➤ X-axis (left ↔ right)
    ➤ Y-axis (top ↕ bottom)

1. Sobel X:
   - Detects vertical edges
   - Measures intensity change in the horizontal direction (left ↔ right)

2. Sobel Y:
   - Detects horizontal edges
   - Measures intensity change in the vertical direction (top ↕ bottom)

3. Laplacian:
   - Second-order derivative (d²)
   - Detects all edges regardless of direction (more sensitive)
   - Combines horizontal and vertical gradients automatically

These are used in:
✅ Edge detection
✅ Feature extraction
✅ Preprocessing before object detection, OCR, etc.
"""


def gradient():
    # Load image and convert to grayscale
    img_path = "/Users/riyaraut/Desktop/all-shapes-and-colors-v-2/test_dataset/img_51.png"
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # ➤ Sobel X: detects vertical edges (horizontal intensity change)
    sobel_x = cv.Sobel(gray, cv.CV_64F, dx=1, dy=0, ksize=3)

    # ➤ Sobel Y: detects horizontal edges (vertical intensity change)
    sobel_y = cv.Sobel(gray, cv.CV_64F, dx=0, dy=1, ksize=3)

    # ➤ Laplacian: second-order edge detector, detects in all directions
    laplacian = cv.Laplacian(gray, cv.CV_64F)
    

    


"""
🔪 Canny Edge Detection

🧠 What is it?
Canny Edge Detection is a **multi-stage algorithm** used to detect sharp changes in intensity—aka edges—in an image.

It's often used **instead of basic gradient methods** (like Sobel or Laplacian) because:
✅ It's more accurate
✅ It reduces noise before detecting edges
✅ It results in **thin, well-defined edges**
✅ It minimizes false positives (detects fewer wrong edges)

📍 Where is it used?
- Lane detection in self-driving cars
- Object detection and segmentation
- Barcode or QR code detection
- Preprocessing for OCR (text recognition)
- Medical image analysis (detecting boundaries)

⚙️ How does it work? (5-step pipeline)

1️⃣ **Noise Reduction**: Apply **Gaussian Blur** to reduce noise before detecting edges.
2️⃣ **Gradient Calculation**: Use **Sobel filters** to compute gradient magnitude and direction.
3️⃣ **Non-Maximum Suppression**: Thin out the edges by suppressing non-maximum pixels.
4️⃣ **Double Thresholding**: Classify pixels into strong, weak, or non-edges.
5️⃣ **Edge Tracking by Hysteresis**: Keep weak edges only if they're connected to strong edges.

✏️ TL;DR:
Canny = Gaussian Blur ➜ Gradient ➜ Thin ➜ Threshold ➜ Final clean edges

🎛️ You control:
- `low_threshold` and `high_threshold`
  → Lower = more edges but possibly more noise
  → Higher = cleaner, but may miss subtle edges
"""

# gaussian blur --> canny 

def cannyedge():
    img_path = "/Users/riyaraut/Desktop/all-shapes-and-colors-v-2/test_dataset/img_51.png"
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    
    blur_gaussian = cv.GaussianBlur(img, (5,5), 0) 
    
    canny = cv.Canny(blur_gaussian,10, 250)
    plt.imshow(canny)
    plt.show()
    
    
"""
🧠 What is Hough Line Transform?

It's a technique used to detect **straight lines** in an image.
It works by transforming points in the image space (x, y) into curves in **Hough space (ρ, θ)**.

📌 Why use it?
- In the Cartesian system (y = mx + b), vertical lines break things (infinite slope).
- But in **polar coordinates**, every line is represented by a unique (ρ, θ), so even vertical lines are handled properly.

🧩 Core Principle:
- Every point (x, y) in the image can lie on **many possible lines**.
- So we transform this point into a **sinusoidal curve** in Hough space: ρ = x·cos(θ) + y·sin(θ)
- When **multiple curves intersect** at the same (ρ, θ), it means those image points lie on the **same line** → BINGO! That's our detected line.e.
- Where many such curves intersect → those points lie on the **same straight line** in the image.


🚗 Example Use Case:
- Lane detection in self-driving cars.
- Detecting lines in scanned documents.
- Detecting edges of objects or buildings in aerial images.

"""

def hough_line_detection():
    # Load and preprocess image
    img = cv.imread("/Users/riyaraut/Desktop/all-shapes-and-colors-v-2/test_dataset/img_51.png")       
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert to grayscale

    # Step 1: Detect edges using Canny
    edges = cv.Canny(gray, 50, 150)

    # Step 2: Detect lines using Standard Hough Line Transform
    lines = cv.HoughLines(edges, 1, np.pi / 180, 100)

    # Step 3: Draw the lines on the original image (if any detected)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]  # ρ (distance from origin), θ (angle in radians)

            # Convert polar coords to Cartesian for drawing
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            # Choose two far points on the line to ensure visibility
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # Draw the line on the original image (green color, thickness 2)
            cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
    # Show the final result
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title("Hough Line Detection")
    plt.axis("off")
    plt.show()

"""
🧠 What is Harris Corner Detection?

Harris Corner Detection is a computer vision technique used to identify corners in an image—
points where the intensity changes significantly in all directions. It’s one of the earliest 
and most popular methods for feature detection.

🔧 How it Works:
- Calculates image gradients in both x and y directions.
- Constructs a structure tensor (matrix) for each pixel using those gradients.
- Computes a "corner response function" (R) using the formula:
      R = det(M) - k * (trace(M))^2
  where:
      - M is a matrix summarizing local gradient information
      - det(M) = strength of the signal in both directions
      - trace(M) = how much variation there is in general
      - k is an empirically chosen constant (commonly 0.04 - 0.06)

- Pixels with high R values are marked as corners.

📌 Why Use It?
- Corners are stable, easily trackable features in an image.
- More informative than just edges or flat regions.

🎯 Where It's Used:
- Feature matching in image stitching (panoramas)
- Object detection and tracking
- Motion tracking in videos
- Camera calibration
- SLAM (Simultaneous Localization and Mapping)
- Robot navigation

"""

def harris_corner():
    # Step 1: Read the input image
    img_path = "/Users/riyaraut/Desktop/all-shapes-and-colors-v-2/train_dataset/img_8.png"
    img = cv.imread(img_path)  # Load in color
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert to grayscale

    # Step 2: Convert grayscale image to float32 (required for Harris)
    gray = np.float32(gray)

    # Step 3: Apply Harris Corner Detection
    harris_response = cv.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    # blockSize: Neighborhood size considered for corner detection
    # ksize: Aperture parameter of the Sobel derivative used
    # k: Harris detector free parameter in the equation

    # Step 4: Dilate detected corners to make them more visible
    harris_response = cv.dilate(harris_response, None)

    # Step 5: Threshold and mark corners on the original image
    img[harris_response > 0.50 * harris_response.max()] = [0, 0, 255]
    # If pixel has strong corner response → color it red (BGR: [0,0,255])

    # Step 6: Display result using matplotlib
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title("Harris Corners")
    plt.axis("off")
    plt.show()
"""
🔍 SIFT: Scale-Invariant Feature Transform

🧠 What is it?
- An algorithm to detect and describe **keypoints** in images.
- Invented by David Lowe in 1999.
- Keypoints are selected based on **scale-space extrema** — stable and repeatable points even under transformations.

🎯 What makes SIFT powerful?
- **Scale-invariant** → works if the image is zoomed in or out.
- **Rotation-invariant** → detects the same keypoint even if the image is rotated.
- **Robust to lighting changes**, noise, and small viewpoint shifts.

🪄 How does it work? (Simplified)
1. Build a **scale-space** using Gaussian blurs.
2. Find **keypoints** by detecting extrema in the Difference of Gaussians (DoG).
3. Assign an **orientation** to each keypoint (based on gradient directions).
4. Extract a **descriptor vector** (usually 128D) summarizing local image gradients.
5. Use these descriptors for **matching**, **recognition**, or **tracking**.

📦 Where is SIFT used?
- Image stitching (panorama)
- Object recognition
- Tracking
- Robot vision
"""

def sift():
    """
    🔍 SIFT (Scale-Invariant Feature Transform) Keypoint Detection
    
    This function detects and visualizes SIFT keypoints in an image.
    
    📌 What it does:
    - Converts image to grayscale
    - Detects keypoints and descriptors using SIFT
    - Draws keypoints (with scale and orientation) on the original image
    - Displays the result using matplotlib
    """

    # Load the image
    img_path = "/Users/riyaraut/Desktop/potrait.png"
    img = cv.imread(img_path)

    # Convert to grayscale (SIFT works on intensity)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Create SIFT object
    sift = cv.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Draw keypoints on the grayscale image
    # Flags explain:
    # DRAW_RICH_KEYPOINTS shows the size and orientation of keypoints as well
    sift_img = cv.drawKeypoints(gray, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display the result
    plt.figure(figsize=(8, 6))
    plt.imshow(sift_img, cmap="gray")
    plt.title("SIFT Keypoints")
    plt.axis("off")
    plt.show()

"""
🧠 What is Optical Flow?
Optical Flow is the pattern of apparent motion of objects in a video or between two consecutive frames due to movement. It estimates how pixels move over time.

🔍 Why use Optical Flow?
Track motion in videos

Detect moving objects

Use in robotics, surveillance, gesture recognition, etc.

Input for higher-level applications like activity recognition, motion-based segmentation, or object tracking

📌 Two Common Methods in OpenCV:
Dense Optical Flow – calculates flow for all pixels

Sparse Optical Flow – tracks only selected feature points (e.g., corners)

We'll focus on Lucas-Kanade Sparse Optical Flow, which is fast and great for tracking objects.

"""











"""🎯 What is Camera Calibration?
Camera calibration is the process of estimating the parameters of a camera — basically teaching the computer how your camera "sees" the world, so it can map 3D real-world points to 2D image coordinates accurately.

🧠 Why do we need it?
Raw images from cameras are distorted due to lens imperfections. Calibration corrects for this and helps:

Remove lens distortion (barrel, pincushion, etc.)

Estimate camera position and orientation in space

Use the camera for 3D reconstruction, augmented reality, robotics, etc.

🔍 What are we estimating?
Two types of parameters:

1. Intrinsic Parameters (camera-specific)
Focal length (fx, fy)

Optical center (cx, cy)

Skew (usually 0)

Lens distortion coefficients (radial and tangential distortion)

2. Extrinsic Parameters (pose of camera w.r.t the world)
Rotation (R) and Translation (T) vectors

⚙️ How does it work?
We show the camera a known pattern (usually a checkerboard) at different angles and positions.

Steps:

Take multiple images of a chessboard grid.

For each image, detect corners of the chessboard.

Map known 3D world coordinates of the corners to 2D image points.

Use OpenCV’s cv.calibrateCamera() to estimate parameters.

Use the result to undistort future images.

📸 Common calibration object:
Checkerboard (black and white grid)

You know exactly where each corner lies in 3D.

🧪 Once calibrated, you can:
Undistort images (cv.undistort)

Estimate camera pose (cv.solvePnP)

Project 3D points (cv.projectPoints)

"""




"""🎯 What is Pose Estimation?
Pose estimation is figuring out the position and orientation of an object (or the camera) in 3D space relative to the camera or world.

In simple terms:

Where is the object located?

How is it rotated?

It gives you:

Rotation Vector (rvec) → orientation

Translation Vector (tvec) → position

🤔 Why is this useful?
Augmented Reality: Placing virtual objects correctly on real-world surfaces

Robotics: Knowing where objects are in 3D

3D reconstruction: Mapping 2D images to 3D models

🛠️ How does it work?
You need:

3D coordinates of some object points (e.g. corners of a checkerboard)

2D coordinates of those points in the image

Camera intrinsic parameters (from calibration)

Then use:

cv.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)
This gives:

rvec (rotation vector)

tvec (translation vector)

You can convert rvec into a rotation matrix using cv.Rodrigues() if needed.

"""



"""🎯 What is Depth Estimation?
Depth estimation means figuring out how far objects in an image are from the camera. It’s like giving 3D understanding to a 2D image.

🧠 Why is this important?
Self-driving cars: Understand how far the next car or pedestrian is.

AR/VR: Place objects realistically in your environment.

3D reconstruction: Build real-world environments from images.

Robotics: Help robots navigate and interact with objects.

📌 Types of Depth Estimation
Monocular Depth Estimation

Input: A single image.

Output: A depth map (each pixel has a depth value).

Usually done with deep learning models (e.g., MiDaS).

No geometry involved, purely data-learned.

Stereo Depth Estimation

Input: Two images from slightly different views (like human eyes).

Find matching points between the two → compute disparity.

Depth ∝ 1 / disparity.

Needs calibrated cameras.

Structure from Motion (SfM)

Uses multiple images from different views.

Tracks points and reconstructs the scene in 3D.

Needs camera pose estimation too.

Depth sensors (hardware-based)

Devices like LiDAR, Kinect, or iPhone’s TrueDepth sensor.

Hardware gives direct depth measurements."""
if __name__ =="__main__":
    sift()
    
    