from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


#NOTES:
#tried using cv2.imshow but im on windows and theres a backend GUI problem so i resorted to matplotlib
#next step 1D filter 0.5[-1, 0, 1]
# have to store at least three frames???
# pixel coords (i, j)

#to avoid redoing path every time one of us works on it 
images_dir = Path(__file__).parent / "Office" / "Office"
print(f"counting images in: {images_dir}")

# gaus_var1 = input("enter gaussian variation 1: ")
# gaus_var2 = input("enter gaussian variation 2: ")
# gaus_var3 = input("enter gaussian variation 3: ")

plt.ion()
fig, ax = plt.subplots()
image_files = sorted(images_dir.glob("*.jpg"))
first_img = cv2.imread(str(image_files[0]), cv2.IMREAD_GRAYSCALE)
img_display = ax.imshow(np.zeros_like(first_img), cmap='gray')
img_display.set_clim(0, 20)  # threshold for what clips to white 0,80 is essentially no noise?

for i in range(1, len(image_files)-1):
    #print(image_files[i])
    prev_img = cv2.imread(str(image_files[i-1]), cv2.IMREAD_GRAYSCALE) # take three images 
    curr_img = cv2.imread(str(image_files[i]), cv2.IMREAD_GRAYSCALE)
    next_img = cv2.imread(str(image_files[i+1]), cv2.IMREAD_GRAYSCALE)

    temporal_derivative = 0.5 * (next_img.astype(np.float32) - prev_img.astype(np.float32)) # thank you numpy for making this easy !!!


    img_display.set_data(np.abs(temporal_derivative))
    ax.set_title(f"Frame {i}")
    ax.axis('off')
    plt.pause(0.01) # change how fast the images are displaying, dont go much faster seems like cpu does not like. 

