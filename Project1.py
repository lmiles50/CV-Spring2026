from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np


#NOTES:
#tried using cv2.imshow but im on windows and theres a backend GUI problem so i resorted to matplotlib
#next step 1D filter 0.5[-1, 0, 1]

#to avoid redoing path every time one of us works on it 
images_dir = Path(__file__).parent / "Office" / "Office"
print(f"counting images in: {images_dir}")

def box_filter(grayscale_image): #untested for now
    filter = 0.5 * np.array([-1,0,1])
    filtered_img = cv2.filter2D(grayscale_image, kernel=filter)
    return filtered_img

def gaus_filter(grayscale_image):


    return filtered_img

try:
    image_files = sorted(images_dir.glob("*.jpg"))

    for img_path in image_files:
        print(img_path)
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        boxfiltered_img = box_filter(img)




        #t = input("input gaussian standard deviation value: ")

except FileNotFoundError:
    print("Ensure Office folder is in the same directory as Project1.py!")
    raise
