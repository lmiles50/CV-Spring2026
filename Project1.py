from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


# - SET PATH - 
# please replace with path to folder of images! 
# Currently set up to run as long as the zip folder was extracted in the same directory as the python file exists in 

images_dir = Path(__file__).parent / "Office" / "Office"
image_files = sorted(images_dir.glob("*.jpg"))
print("number of images found:", len(image_files))

# - USER INPUT - 
print("\nPlease enter three standard deviations (tσ) for the temporal 1D Gaussian Derivative filter, in INCREASING order:")
t1 = float(input("tσ 1: "))
t2 = float(input("tσ 2: "))
t3 = float(input("tσ 3: "))

# - FUNCTION DEFINITIONS - 

# 1D gaussian
def GaussianDeriv_1D(t_sigma):
    r = int(np.ceil(3 * t_sigma))
    t = np.arange(-r, r + 1, dtype=np.float32)
    filt = -(t / (t_sigma ** 2)) * np.exp(-(t ** 2) / (2 * t_sigma ** 2))
    filt = filt / (np.sum(np.abs(filt)) + 1e-12)
    return filt.astype(np.float32), r

def apply_Gaus1D(center_idx, filt, r):
    acc = None
    for offset in range(-r, r + 1):
        img = cv2.imread(str(image_files[center_idx + offset]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        w = filt[offset + r]
        if acc is None:
            acc = w * img  
        else:
            acc += w * img
    return acc

# - PART 1 -

print("PART 1 – Temporal Filter Comparison")

f1, r1 = GaussianDeriv_1D(t1)
f2, r2 = GaussianDeriv_1D(t2)
f3, r3 = GaussianDeriv_1D(t3)

max_r = max(r1, r2, r3, 1)

plt.figure("PART 1 – Temporal Filter Comparison", figsize=(15, 6))
plt.ion()

for i in range(max_r, len(image_files) - max_r):

    original = cv2.imread(str(image_files[i]), cv2.IMREAD_GRAYSCALE) #reading in grayscale 
    original_f = original.astype(np.float32)

    # 0.5[-1 0 1] temporal filter 
    prev = cv2.imread(str(image_files[i - 1]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    next = cv2.imread(str(image_files[i + 1]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    simple = 0.5 * (next - prev)

    # 1D gaussian derivative results
    d1 = apply_Gaus1D(i, f1, r1)
    d2 = apply_Gaus1D(i, f2, r2)
    d3 = apply_Gaus1D(i, f3, r3)

    plt.clf() # 0              1                    2                                     3                                       4
    titles = ["Original", "0.5[-1 0 1]", f"1D Gaussian Derivative \n  tσ={t1}", f"1D Gaussian Derivative \n  tσ={t2}", f"1D Gaussian Derivative \n tσ={t3}"]
           #    0           1                2           3         4
    images = [original, np.abs(simple), np.abs(d1), np.abs(d2), np.abs(d3)]

    for j in range(5):
        plt.subplot(1, 5, j + 1)
        plt.imshow(images[j], cmap="gray")
        plt.title(titles[j])
        plt.axis("off")

    plt.pause(0.01)

plt.ioff()
plt.show()
