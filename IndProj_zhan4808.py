import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#https://towardsdatascience.com/image-filters-in-python-26ee938e57d2
#https://www.google.com/search?q=uploading+an+image+filter+and+applying+it+python&oq=uploading+an+image+filter+and+applying+it+python&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQIRigAdIBCTI5MjQ0ajBqN6gCALACAA&sourceid=chrome&ie=UTF-8
#https://github.com/m4nv1r/medium_articles/blob/master/Image_Filters_in_Python.ipynb
#https://www.askpython.com/python/examples/filters-to-images

#/Users/robertzhang/Downloads/mnms.jpg
def read_image(file_path):
    while True:
        try:
            image = cv2.imread(file_path)
            if image is not None:
                image_rgb = Image.open(file_path)
                return image_rgb
            else:
                print("Error: Image not found or invalid format.")
                file_path = input("Enter a valid image file path: ")
        except Exception as e:
            print(f"Error reading the image: {e}")

def show_images(filter_name, img, gray_img = [[0]]):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6)) 
    ax[0].imshow(np.array(img,dtype=np.uint8))
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
    
    #img2 = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #return img2

def edge_detection(img):
    return cv2.Canny(img, 100, 200)

def sepia(img):
    sepia_matrix = np.array([[0.393, 0.769, 0.189],
                            [0.349, 0.686, 0.168],
                            [0.272, 0.534, 0.131]])

    sepia_image = cv2.transform(img, sepia_matrix)
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)

    return sepia_image

def gaussian(img):
    blur = cv2.GaussianBlur(img, (35, 35), 0)
    return blur

def emboss(img):
    Emboss_Kernel = np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
    embossed = cv2.filter2D(src=img, kernel=Emboss_Kernel, ddepth=-1)
    return embossed

def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

def mean(img):
    figure_size = 9 # the dimension of the x and y axis of the kernal.
    meaned = cv2.blur(img,(figure_size, figure_size))
    return meaned

def median(img):
    figure_size = 9
    medianed = cv2.medianBlur(img, figure_size)
    return medianed

# first a conservative filter for grayscale images will be defined.
def conservative_smoothing_gray(data, filter_size):
    temp = []
    indexer = filter_size // 2
    new_image = data.copy()
    nrow, ncol = data.shape
    for i in range(nrow):  
        for j in range(ncol):    
            for k in range(i-indexer, i+indexer+1):  
                for m in range(j-indexer, j+indexer+1):    
                    if (k > -1) and (k < nrow):       
                        if (m > -1) and (m < ncol):                
                            temp.append(data[k,m])                   
            temp.remove(data[i,j])     
            max_value = max(temp)
            min_value = min(temp)
            if data[i,j] > max_value:
                new_image[i,j] = max_value
            elif data[i,j] < min_value:    
                new_image[i,j] = min_value
            temp =[]
    return new_image.copy()

def conservative(img):
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    smoothed = conservative_smoothing_gray(img2,5)
    return smoothed

def laplacian(img):
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
    lap = cv2.Laplacian(img2, cv2.CV_64F)
    return img2 + lap 

def crimminsCulling(data):
    new_image = data.copy()
    nrow = len(data)
    ncol = len(data[0])
    
    # Dark pixel adjustment
    
    # First Step
    # N-S
    for i in range(1, nrow):
        for j in range(ncol):
            if data[i-1,j] >= (data[i,j] + 2):
                new_image[i,j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(ncol-1):
            if data[i,j+1] >= (data[i,j] + 2):
                new_image[i,j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow):
        for j in range(1, ncol):
            if data[i-1,j-1] >= (data[i,j] + 2):
                new_image[i,j] += 1
    data = new_image
    #NE-SW
    for i in range(1, nrow):
        for j in range(ncol-1):
            if data[i-1,j+1] >= (data[i,j] + 2):
                new_image[i,j] += 1
    data = new_image
    # Second Step
    # N-S
    for i in range(1, nrow-1):
        for j in range(ncol):
            if (data[i-1,j] > data[i,j]) and (data[i,j] <= data[i+1,j]):
                new_image[i,j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol-1):
            if (data[i,j+1] > data[i,j]) and (data[i,j] <= data[i,j-1]):
                new_image[i,j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow-1):
        for j in range(1, ncol-1):
            if (data[i-1,j-1] > data[i,j]) and (data[i,j] <= data[i+1,j+1]):
                new_image[i,j] += 1
    data = new_image
    # NE-SW
    for i in range(1, nrow-1):
        for j in range(1, ncol-1):
            if (data[i-1,j+1] > data[i,j]) and (data[i,j] <= data[i+1,j-1]):
                new_image[i,j] += 1
    data = new_image
    #Third Step
    # N-S
    for i in range(1, nrow-1):
        for j in range(ncol):
            if (data[i+1,j] > data[i,j]) and (data[i,j] <= data[i-1,j]):
                new_image[i,j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol-1):
            if (data[i,j-1] > data[i,j]) and (data[i,j] <= data[i,j+1]):
                new_image[i,j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow-1):
        for j in range(1, ncol-1):
            if (data[i+1,j+1] > data[i,j]) and (data[i,j] <= data[i-1,j-1]):
                new_image[i,j] += 1
    data = new_image
    # NE-SW
    for i in range(1, nrow-1):
        for j in range(1, ncol-1):
            if (data[i+1,j-1] > data[i,j]) and (data[i,j] <= data[i-1,j+1]):
                new_image[i,j] += 1
    data = new_image
    # Fourth Step
    # N-S
    for i in range(nrow-1):
        for j in range(ncol):
            if (data[i+1,j] >= (data[i,j]+2)):
                new_image[i,j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1,ncol):
            if (data[i,j-1] >= (data[i,j]+2)):
                new_image[i,j] += 1
    data = new_image
    # NW-SE
    for i in range(nrow-1):
        for j in range(ncol-1):
            if (data[i+1,j+1] >= (data[i,j]+2)):
                new_image[i,j] += 1
    data = new_image
    # NE-SW
    for i in range(nrow-1):
        for j in range(1,ncol):
            if (data[i+1,j-1] >= (data[i,j]+2)):
                new_image[i,j] += 1
    data = new_image
    
    # Light pixel adjustment
    
    # First Step
    # N-S
    for i in range(1,nrow):
        for j in range(ncol):
            if (data[i-1,j] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(ncol-1):
            if (data[i,j+1] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # NW-SE
    for i in range(1,nrow):
        for j in range(1,ncol):
            if (data[i-1,j-1] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # NE-SW
    for i in range(1,nrow):
        for j in range(ncol-1):
            if (data[i-1,j+1] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # Second Step
    # N-S
    for i in range(1,nrow-1):
        for j in range(ncol):
            if (data[i-1,j] < data[i,j]) and (data[i,j] >= data[i+1,j]):
                new_image[i,j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol-1):
            if (data[i,j+1] < data[i,j]) and (data[i,j] >= data[i,j-1]):
                new_image[i,j] -= 1
    data = new_image
    # NW-SE
    for i in range(1,nrow-1):
        for j in range(1,ncol-1):
            if (data[i-1,j-1] < data[i,j]) and (data[i,j] >= data[i+1,j+1]):
                new_image[i,j] -= 1
    data = new_image
    # NE-SW
    for i in range(1,nrow-1):
        for j in range(1,ncol-1):
            if (data[i-1,j+1] < data[i,j]) and (data[i,j] >= data[i+1,j-1]):
                new_image[i,j] -= 1
    data = new_image
    # Third Step
    # N-S
    for i in range(1,nrow-1):
        for j in range(ncol):
            if (data[i+1,j] < data[i,j]) and (data[i,j] >= data[i-1,j]):
                new_image[i,j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1,ncol-1):
            if (data[i,j-1] < data[i,j]) and (data[i,j] >= data[i,j+1]):
                new_image[i,j] -= 1
    data = new_image
    # NW-SE
    for i in range(1,nrow-1):
        for j in range(1,ncol-1):
            if (data[i+1,j+1] < data[i,j]) and (data[i,j] >= data[i-1,j-1]):
                new_image[i,j] -= 1
    data = new_image
    # NE-SW
    for i in range(1,nrow-1):
        for j in range(1,ncol-1):
            if (data[i+1,j-1] < data[i,j]) and (data[i,j] >= data[i-1,j+1]):
                new_image[i,j] -= 1
    data = new_image
    # Fourth Step
    # N-S
    for i in range(nrow-1):
        for j in range(ncol):
            if (data[i+1,j] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1,ncol):
            if (data[i,j-1] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # NW-SE
    for i in range(nrow-1):
        for j in range(ncol-1):
            if (data[i+1,j+1] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # NE-SW
    for i in range(nrow-1):
        for j in range(1,ncol):
            if (data[i+1,j-1] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    return new_image.copy()

def crimmins(img):
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    crim = crimminsCulling(img2)
    return crim


def main():
    file_path = input("Enter the path to the image file: ")
    if file_path == 'mnms':
        file_path = '/Users/robertzhang/Downloads/mnms.jpg'
    img = read_image(file_path)
    run = True

    while(run):

        print("Select a filter:")
        print("1. Grayscale")
        print("2. Edge Detection")
        print("3. Sepia")
        print("4. Gaussian")
        print("5. Emboss")
        print("6. Sharpen")
        print("7. Mean")
        print("8. Median")
        print("9. Laplacian")
        print("10. Conservative")
        print("11. Exit Program")

        choice = input("Enter your choice (#): ")

        if choice == '1':
            filtered_image = grayscale(img)
            show_images("Grayscale", img, filtered_image)
        elif choice == '2':
            filtered_image = edge_detection(img)
            show_images( "Edge Detection", img, filtered_image)
        elif choice == '3':
            filtered_image = sepia(img)
            show_images("Sepia", img, filtered_image)
        elif choice == '4':
            filtered_image = gaussian(img)
            show_images("Gaussian Blur", img, filtered_image)
        elif choice == '5':
            filtered_image = sepia(img)
            show_images("Emboss", img, filtered_image)
        elif choice == '6':
            filtered_image = sharpen(img)
            show_images("Sharpen", img, filtered_image)
        elif choice == '7':
            filtered_image = mean(img)
            show_images("Mean", img, filtered_image)
        elif choice == '8':
            filtered_image = median(img)
            show_images("Median", img, filtered_image)
        elif choice == '9':
            filtered_image = laplacian(img)
            show_images("Laplacian", img, filtered_image)
        elif choice == '10':
            filtered_image = conservative(img)
            show_images("Conservative", img, filtered_image)
        elif choice == '11':
                print("Exiting...")
                run = False
                break
        else:
            print("Invalid choice. Please enter numbers from 1-11.")

if __name__ == "__main__":
    main()
