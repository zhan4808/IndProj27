import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def read_image(file_path):
    while True:
        try:
            if file_path == 'mnms':
                file_path = '/Users/robertzhang/Documents/GitHub/IndProj27/mnms.jpg'
            elif file_path == 'blur':
                file_path = '/Users/robertzhang/Documents/GitHub/IndProj27/mnmsBlurred.jpg'
            elif file_path == 'rob':
                file_path = '/Users/robertzhang/Documents/GitHub/IndProj27/rob1.png'
            elif file_path == 'tokyo':
                file_path = '/Users/robertzhang/Documents/GitHub/IndProj27/tokyo3.jpg'
            image = cv2.imread(file_path)
            if image is not None:
                image_rgb = Image.open(file_path)
                return image_rgb
            else:
                print("Error: Image not found or invalid format.")
                file_path = input("Enter a valid image file path: ")
        except Exception as e:
            print(f"Error reading the image: {e}")

def show_images(filter_name, img, gray_img=[[0]]):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax[0].imshow(np.array(img, dtype=np.uint8))
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(np.array(gray_img), cmap='gray', vmin=0, vmax=255)
    ax[1].set_title(f'{filter_name} Filter')
    ax[1].axis('off')
    plt.show()

def grayscale(img):
    img = np.array(img)
    vector1 = [0.2989, 0.5870, 0.1140]
    grayscale_img = np.dot(img[..., :3], vector1).astype('uint8')
    return Image.fromarray(grayscale_img)

#gradient represents rate of change of intensity (brightness) of the image at each pixel in both x and y directions
def compute_gradient_magnitude(image):
    dx = np.gradient(image, axis=0)
    dy = np.gradient(image, axis=1) 
    gradient_magnitude = np.sqrt(dx**2 + dy**2) 
    return gradient_magnitude

def edge_detection(img, threshold):
    grayscale_img = grayscale(img)
    gradient_magnitude = compute_gradient_magnitude(np.array(grayscale_img))
    threshold_fraction = threshold
    threshold = threshold_fraction * np.max(gradient_magnitude) 
    print(f"Chosen threshold: {threshold}")
    edge_image = np.where(gradient_magnitude > threshold, 255, 0).astype(np.uint8)
    return edge_image

def sepia(img):
    img_array = np.array(img)
    sepia_matrix = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    sepia_image = cv2.transform(img_array, sepia_matrix)
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
    return Image.fromarray(sepia_image)

def is_kernel_valid(img, kernel_size):
    img_array = np.array(img)
    height, width = img_array.shape[:2]

    if kernel_size >= height or kernel_size >= width:
        print("Error: Kernel size is larger than the image dimensions.")
        return False
    elif kernel_size % 2 == 0 or kernel_size < 0:
        print("Error: Kernel size must be an odd positive number")
    else:
        return True

#Each pixel's new value is set to a weighted average of that pixel's neighborhood
#Original pixel's value receives the heaviest weight (having the highest Gaussian value) 
# and neighboring pixels receive smaller weights as their distance to the original pixel increases.
def gaussian(img, size):
    img_array = np.array(img)
    blur = cv2.GaussianBlur(img_array, (size, size), 0)
    return Image.fromarray(blur)

#this kernel enhances the contrast of the pixel
def sharpen(img):
    img_array = np.array(img)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(img_array, -1, kernel)
    return Image.fromarray(sharpened)

#uses uniform quantization to simplify an entire grayscale image into 'k' number shades of gray
#min-max grayscale intensities with step sizes depending on k
#as k gets higher you get more detail
def quantize_img_gray(img, k):
    img = np.array(img)
    grayscale_img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype('uint8')
    min_gray = np.min(grayscale_img)
    max_gray = np.max(grayscale_img)
    range_gray = max_gray - min_gray + 1
    step_size = range_gray / k
    quantized_img = np.floor((grayscale_img - min_gray) / step_size) * step_size + min_gray
    quantized_img = np.uint8(quantized_img)
    return quantized_img
    
#Quantizes based on euclidean distance to the different colors black green yellow brown
def quantize_img(img_path):
    color_palette = np.array([(0, 0, 10), (41, 200, 0), (36, 0, 85), (40, 40, 40)])
    img_array = np.array(img_path)
    quantized_img = np.zeros_like(img_array)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            pixel = img_array[i, j]
            distances = [np.linalg.norm(pixel - color) for color in color_palette]
            closest_color_index = np.argmin(distances)
            quantized_img[i, j] = color_palette[closest_color_index]
    quantized_img = Image.fromarray(quantized_img)
    return quantized_img


def rgb_to_lab(rgb):
    lab = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0][0].astype(float)
    lab[0] *= 100 / 255
    lab[1] -= 128
    lab[2] -= 128
    return lab

def lab_distance(lab1, lab2):
    return np.linalg.norm(lab1 - lab2)

#CIE LAB color profile is most color accurate to human eye
def quantize_img_lab(img_path):
    color_palette = np.array([(0, 0, 10), (41, 200, 0), (36, 0, 85), (40, 40, 40)])
    img_array = np.array(img_path)
    quantized_img = np.zeros_like(img_array)
    
    palette_lab = [rgb_to_lab(color) for color in color_palette]
    
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            pixel = img_array[i, j]
            pixel_lab = rgb_to_lab(pixel)
            closest_color = None
            min_distance = float('inf')
            for idx, palette_color_lab in enumerate(palette_lab):
                distance = lab_distance(pixel_lab, palette_color_lab)
                if distance < min_distance:
                    min_distance = distance
                    closest_color = color_palette[idx]
            quantized_img[i, j] = closest_color
    
    quantized_img = Image.fromarray(quantized_img.astype('uint8'))
    return quantized_img

def valid_adaptive(num):
    if num < 1 or num > 100:
        print("Error: Number of colors needs to be between 1-100 inclusive.")
        return False
    else:   
        return True
    
def adaptive_quant(img, num):
    filtered_image = img.quantize(colors=num)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6)) 
    ax[0].imshow(img)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(filtered_image)
    ax[1].set_title(f'Adapative Quantization with {num} Colors')
    ax[1].axis('off')
    plt.show()


def main():
    file_path = input("Enter the path to the image file: ")
    img = read_image(file_path)
    run = True

    while run:
        print("Select a filter:")
        print("1. Grayscale")
        print("2. Edge Detection")
        print("3. Sepia")
        print("4. Gaussian")
        print("5. Sharpen")
        print("6. Quantization - Grayscale")
        print("7. Quantization - RGB")
        print("8. Quantization - LAB")
        print("9. Adapative Color Quantization")
        print("10. Exit Program")

        choice = input("Enter your choice (#): ")

        if choice == '1':
            filtered_image = grayscale(img)
            show_images("Grayscale", img, filtered_image)
        elif choice == '2':
            thresh = float(input("Input Threshold for Gradient Magnitude (0-1): "))
            filtered_image = edge_detection(img, thresh)
            show_images(f"Edge Detection with Threshold {thresh}", img, filtered_image)
        elif choice == '3':
            filtered_image = sepia(img)
            show_images("Sepia", img, filtered_image)
        elif choice == '4':
            while True:
                kernelSize = int(input("Input Kernel Size for Blur Filter: "))
                if is_kernel_valid(img, kernelSize):
                    filtered_image = gaussian(img, kernelSize)
                    show_images("Gaussian Blur", img, filtered_image)
                    break
        elif choice == '5':
            filtered_image = sharpen(img)
            show_images("Sharpen", img, filtered_image)
        elif choice == '6':
                k = int(input("Input 'k' number of Gray Levels: "))
                filtered_image = quantize_img_gray(img, k)
                show_images(f"Quantized Grayscale With k = {k}", img, filtered_image)
        elif choice == '7':
            filtered_image = quantize_img(img)
            show_images("Quantization RGB", img, filtered_image)
        elif choice == '8':
            filtered_image = quantize_img_lab(img)
            show_images("Quantization LAB", img, filtered_image)
        elif choice == '9':
            while True:
                num = int(input("Select how many colors to be quantized (1-100): "))
                if valid_adaptive(num):
                    adaptive_quant(img, num)
                    break  
        elif choice == '10':
            print("Exiting...")
            run = False
            break
        else:
            print("Invalid choice. Please enter numbers from 1-10.")

if __name__ == "__main__":
    main()
