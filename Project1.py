from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


#NOTES:
#tried using cv2.imshow but im on windows and theres a backend GUI problem so i resorted to matplotlib
#next step gaussian filters dont show up - threshold or std dev or calculation problem?

#to avoid redoing path every time one of us works on it 
images_dir = Path(__file__).parent / "EnterExitCrossingPaths2cor" / "EnterExitCrossingPaths2cor"
print(f"counting images in: {images_dir}")

print("Please enter three desired standard deviations for a 1D gaussian derivative filter in increasing order. ")
sigmaA = float(input("enter first std deviation value: "))
sigmaB = float(input("enter second std deviation value: "))
sigmaC = float(input("enter third std deviation value: "))

def Gaussian_Derivative(std_deviation):
    sigma = std_deviation
    mu = 0
    x = curr_img.astype(np.float32)
    gauss = gauss = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-1*(x-mu)**2/(2*sigma**2))
    gauss_derivative=-x/sigma**2*gauss
    return gauss_derivative

plt.ion()
image_files = sorted(images_dir.glob("*.jpg"))
first_img = cv2.imread(str(image_files[0]), cv2.IMREAD_GRAYSCALE)
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

titles = [
    "Temporal Derivative",
    f"Gaussian Derivative σ={sigmaA}",
    f"Gaussian Derivative σ={sigmaB}",
    f"Gaussian Derivative σ={sigmaC}"
]

displays = []
for ax, title in zip(axes.flat, titles):
    im = ax.imshow(np.zeros_like(first_img), cmap='gray')
    im.set_clim(0, 160)  #THRESHOLD - 80 is basically no noise 
    ax.set_title(title)
    ax.axis('off')
    displays.append(im)

for i in range(1, len(image_files)-1):
    #print(image_files[i])
    prev_img = cv2.imread(str(image_files[i-1]), cv2.IMREAD_GRAYSCALE) # loads in three sequential images 
    curr_img = cv2.imread(str(image_files[i]), cv2.IMREAD_GRAYSCALE)
    next_img = cv2.imread(str(image_files[i+1]), cv2.IMREAD_GRAYSCALE)

    temporal_derivative = 0.5 * (next_img.astype(np.float32) - prev_img.astype(np.float32)) # thank you numpy for making this easy !!!
    gdA = Gaussian_Derivative(sigmaA)
    gdB = Gaussian_Derivative(sigmaB)
    gdC = Gaussian_Derivative(sigmaC)

    displays[0].set_data(np.abs(temporal_derivative))
    displays[1].set_data(np.abs(gdA))
    displays[2].set_data(np.abs(gdB))
    displays[3].set_data(np.abs(gdC))

    fig.suptitle(f"Frame {i}", fontsize=14)
    plt.pause(0.01) # change how fast the images are displaying, dont go much faster seems like cpu does not like. 

