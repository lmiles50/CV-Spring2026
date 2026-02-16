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

print("\nPlease enter three standard deviations (sσ) for the spatial smoothing 2D Gaussian filter, in INCREASING order:")
s1 = float(input("sσ 1: "))
s2 = float(input("sσ 2: "))
s3 = float(input("sσ 3: "))

# threshold multipliers for Part 3
k_values = [2.5, 4.0, 6.0]


# - FUNCTION DEFINITIONS - 

# 1D gaussian derivative 
def GaussianDeriv_1D(t_sigma):
    r = int(np.ceil(3 * t_sigma))
    t = np.arange(-r, r + 1, dtype=np.float32)
    filt = -(t / (t_sigma ** 2)) * np.exp(-(t ** 2) / (2 * t_sigma ** 2))
    filt = filt / (np.sum(np.abs(filt)) + 1e-12)
    return filt.astype(np.float32), r

def apply_GausDeriv1D_stack(stack, filt):
    acc = None
    for idx, img in enumerate(stack):
        w = filt[idx]
        if acc is None:
            acc = w * img
        else:
            acc += w * img
    return acc

# 2D spatial filtering ALL  
def spatial_filtering(gray, mode, sigma=1.0):
    if mode == "box3x3":
        return cv2.blur(gray, (3, 3))
    if mode == "box5x5":
        return cv2.blur(gray, (5, 5))
    if mode == "gauss":
        return cv2.GaussianBlur(gray, (0, 0), sigma)
    return gray


#threshholding
def threshold(grad_abs, k):
    sigma_est = np.median(grad_abs) / 0.67449
    return k * sigma_est

# loading all images into memory to try to make it faster
all_images = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float32) for f in image_files]


# - PART 1 - ---------------------------------------------------------------------------------------------------------------------------------------------

print("PART 1 – Temporal Filter Comparison")

f1, r1 = GaussianDeriv_1D(t1)
f2, r2 = GaussianDeriv_1D(t2)
f3, r3 = GaussianDeriv_1D(t3)

max_r = max(r1, r2, r3, 1)

fig1 = plt.figure("PART 1 – Temporal Filter Comparison", figsize=(18, 6))
plt.ion()

for i in range(max_r, len(all_images) - max_r):
    if not plt.fignum_exists(fig1.number):  # stop if user hits x 
        break

    original = all_images[i]

    # 0.5[-1 0 1] temporal filter 
    prev = all_images[i - 1]
    next = all_images[i + 1]
    simple = 0.5 * (next - prev)

    # 1D gaussian derivative results
    d1 = apply_GausDeriv1D_stack(all_images[i - r1:i + r1 + 1], f1)
    d2 = apply_GausDeriv1D_stack(all_images[i - r2:i + r2 + 1], f2)
    d3 = apply_GausDeriv1D_stack(all_images[i - r3:i + r3 + 1], f3)

    plt.clf()
    titles = ["Original", "0.5[-1 0 1]", f"1D Gaussian Derivative \n  tσ={t1}", f"1D Gaussian Derivative \n  tσ={t2}", f"1D Gaussian Derivative \n tσ={t3}"]
    images = [original, np.abs(simple), np.abs(d1), np.abs(d2), np.abs(d3)]

    for j in range(2):
        plt.subplot(2, 3, j + 1)
        plt.imshow(images[j], cmap="gray")
        plt.title(titles[j])
        plt.axis("off")
    for j in range(3):
        plt.subplot(2, 3, j + 4)
        plt.imshow(images[j + 2], cmap="gray")
        plt.title(titles[j + 2])
        plt.axis("off")

    plt.pause(0.01)

plt.ioff()
plt.show()


# - PART TWO - -----------------------------------------------------------------------------------------------------------------------------------------------

print("PART 2 – Spatial + Temporal Filter Comparison")

fig2 = plt.figure("PART 2 – Spatial + Temporal Filter Comparison", figsize=(20, 15))
plt.ion()

# computing spatially filtered version of all versions of the image created in part 1
spatial_modes = ["none", "box3x3", "box5x5", "gauss_s1", "gauss_s2", "gauss_s3"]
pre_spatial = []
for img in all_images:
    pre_spatial.append([
        img,
        spatial_filtering(img, "box3x3"),
        spatial_filtering(img, "box5x5"),
        spatial_filtering(img, "gauss", s1),
        spatial_filtering(img, "gauss", s2),
        spatial_filtering(img, "gauss", s3)
    ])

temporal_filters = [("Original", None, 0), ("0.5[-1 0 1]", None, 1), (f"tσ={t1}", f1, r1), (f"tσ={t2}", f2, r2), (f"tσ={t3}", f3, r3)]
n_temporal = len(temporal_filters)
n_spatial = len(spatial_modes)

for i in range(max_r, len(all_images) - max_r):
    if not plt.fignum_exists(fig2.number):  # stop if user hits x
        break

    results = []

    for temp_name, filt, r in temporal_filters:
        for s_idx in range(n_spatial):
    # select prefiltered image stack
            if temp_name == "Original":
                results.append(pre_spatial[i][s_idx])
            elif temp_name == "0.5[-1 0 1]":
                prev_s = pre_spatial[i - 1][s_idx]
                next_s = pre_spatial[i + 1][s_idx]
                results.append(0.5 * (next_s - prev_s))
            else:
                stack = [pre_spatial[i + offset][s_idx] for offset in range(-r, r + 1)]
                results.append(apply_GausDeriv1D_stack(stack, filt))

    plt.clf()
    
    for idx, res in enumerate(results):
        plt.subplot(n_temporal, n_spatial, idx + 1)
        plt.imshow(np.abs(res), cmap="gray")
        plt.axis("off")
        t_idx = idx // n_spatial
        s_idx = idx % n_spatial
        if t_idx == 0:
            plt.title(["Orig","3x3 Box","5x5 Box",f"Gauss s1={s1}",f"Gauss s2={s2}",f"Gauss s3={s3}"][s_idx], fontsize=8)
        if s_idx == 0:
            plt.ylabel(temporal_filters[t_idx][0], fontsize=8)
    
    plt.pause(0.01)

plt.ioff()
plt.show()

# - PART THREE - -----------------------------------------------------------------------------------------------------------------------------------------------

print("PART 3 – Threshold Comparison (3x3 Box + 0.5[-1,0,1])")

fig3 = plt.figure("PART 3 – Threshold Comparison", figsize=(20, 8))
plt.ion()

for i in range(1, len(all_images) - 1):

    if not plt.fignum_exists(fig3.number):  # stop if user hits x
        break

    original_color = cv2.imread(str(image_files[i]))
    
    # reapply 3x3 box filter
    prev = spatial_filtering(all_images[i - 1], "box3x3")
    next = spatial_filtering(all_images[i + 1], "box3x3")
    
    # temporal derivative, simple filter 
    grad = np.abs(0.5 * (next - prev))

    # threshhold masks 
    masks = [(grad >= threshold(grad, k)).astype(np.uint8) * 255 for k in k_values]

    # Make overlay!!
    T_overlay = threshold(grad, 4.0)
    overlay_mask = (grad >= T_overlay).astype(np.uint8)
    overlay_img = original_color.copy()
    overlay_img[overlay_mask == 1] = [0, 0, 255]  # mark changes in red

    plt.clf()

    # show original image
    plt.subplot(1, 5, 1)
    plt.imshow(cv2.cvtColor(original_color, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    # show threshold masks
    for j, k in enumerate(k_values):
        plt.subplot(1, 5, j + 2)
        plt.imshow(masks[j], cmap="gray")
        plt.title(f"k = {k}")
        plt.axis("off")

    # show overlay with k=4
    plt.subplot(1, 5, 5)
    plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    plt.title("Overlay k=4")
    plt.axis("off")

    plt.pause(0.01)

plt.ioff()
plt.show()
