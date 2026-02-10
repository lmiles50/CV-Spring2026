from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np


# NOTES
# - Tried using cv2.imshow but im on windows and theres a backend GUI problem so i resorted to matplotlib
# - Motion detection idea:
#   Background pixels change little over time, moving objects create large temporal changes.
# - We compute temporal derivative filters at each pixel,
#   threshold |derivative| to get a motion mask, and overlay it on the frame.


# PATH SETUP
# to avoid redoing path every time one of us works on it 
images_dir = Path(__file__).parent / "Office" / "Office"
print(f"counting images in: {images_dir}")

image_files = sorted(images_dir.glob("*.jpg"))
print("Number of images found:", len(image_files))


# USER INPUTS

print("\nEnter three standard deviations for a 1D Gaussian DERIVATIVE TEMPORAL filter (tσ), in increasing order.")
sigmaA = float(input("enter first tσ: "))
sigmaB = float(input("enter second tσ: "))
sigmaC = float(input("enter third tσ: "))

print("\nEnter three standard deviations for SPATIAL Gaussian smoothing (sσ), in increasing order.")
sA = float(input("enter first sσ: "))
sB = float(input("enter second sσ: "))
sC = float(input("enter third sσ: "))

# threshold multiplier for adaptive thresholding
k_thresh = 4.0

# choose which temporal filter output to use for mask/overlay
# options: "simple", "A", "B", "C"
mask_source = "B"

pause_time = 0.01


# SPATIAL FILTERS

def spatial_filter(gray, mode, s_sigma):
    # gray should be float32
    if mode == "none":
        return gray
    if mode == "box3":
        return cv2.blur(gray, (3, 3))
    if mode == "box5":
        return cv2.blur(gray, (5, 5))
    if mode == "gauss":
        return cv2.GaussianBlur(gray, (0, 0), s_sigma)


# TEMPORAL FILTERS

def dog_temporal_filter(t_sigma):
    # 1D derivative-of-Gaussian temporal filter
    # filter is defined over time indices t = -r..r
    r = int(np.ceil(3 * t_sigma))
    t = np.arange(-r, r + 1, dtype=np.float32)
    filt = -(t / (t_sigma ** 2)) * np.exp(-(t ** 2) / (2 * (t_sigma ** 2)))
    filt = filt / (np.sum(np.abs(filt)) + 1e-12)  # normalize
    return filt.astype(np.float32), r

# simple temporal derivative filter: 0.5[-1, 0, 1]
simple_temporal_filter = 0.5 * np.array([-1.0, 0.0, 1.0], dtype=np.float32)

filtA, rA = dog_temporal_filter(sigmaA)
filtB, rB = dog_temporal_filter(sigmaB)
filtC, rC = dog_temporal_filter(sigmaC)

# need valid frame indices for largest temporal window
max_r = max(rA, rB, rC, 1)


# APPLY TEMPORAL FILTER (WITH SPATIAL FILTERING INSIDE)

def apply_temporal_filter(center_idx, temporal_filter, radius, spatial_mode, s_sigma):
    # computes sum_{offset=-r..r} temporal_filter[offset] * I(center_idx + offset)
    acc = None
    for offset in range(-radius, radius + 1):
        img = cv2.imread(str(image_files[center_idx + offset]),
                          cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img = spatial_filter(img, spatial_mode, s_sigma)
        w = temporal_filter[offset + radius]
        if acc is None:
            acc = w * img
        else:
            acc += w * img
    return acc


# DISPLAY SETUP

plt.ion()

first_img_gray = cv2.imread(str(image_files[0]), cv2.IMREAD_GRAYSCALE)
H, W = first_img_gray.shape

fig, axes = plt.subplots(2, 3, figsize=(13, 7))

titles = [
    "Simple Temporal Derivative",
    f"DoG Temporal Filter (tσ={sigmaA})",
    f"DoG Temporal Filter (tσ={sigmaB})",
    f"DoG Temporal Filter (tσ={sigmaC})",
]

# derivative displays (2x2 in columns 0 and 1)
disp_simple = axes[0, 0].imshow(np.zeros_like(first_img_gray), cmap="gray", vmin=0, vmax=80)
axes[0, 0].set_title(titles[0]); axes[0, 0].axis("off")

disp_A = axes[0, 1].imshow(np.zeros_like(first_img_gray), cmap="gray", vmin=0, vmax=80)
axes[0, 1].set_title(titles[1]); axes[0, 1].axis("off")

disp_B = axes[1, 0].imshow(np.zeros_like(first_img_gray), cmap="gray", vmin=0, vmax=80)
axes[1, 0].set_title(titles[2]); axes[1, 0].axis("off")

disp_C = axes[1, 1].imshow(np.zeros_like(first_img_gray), cmap="gray", vmin=0, vmax=80)
axes[1, 1].set_title(titles[3]); axes[1, 1].axis("off")

# mask (top-right)
mask_im = axes[0, 2].imshow(np.zeros_like(first_img_gray), cmap="gray", vmin=0, vmax=255)
axes[0, 2].set_title("Motion Mask")
axes[0, 2].axis("off")

# overlay (bottom-right)
dummy_rgb = np.zeros((H, W, 3), dtype=np.uint8)
overlay_im = axes[1, 2].imshow(dummy_rgb)
axes[1, 2].set_title("Overlay Result")
axes[1, 2].axis("off")



# SPATIAL FILTER RUNS (PART 2 OF ASSIGNMENT)

spatial_runs = [
    ("box3", 1.0),
    ("box5", 1.0),
    ("gauss", sA),
    ("gauss", sB),
    ("gauss", sC),
]


# MAIN LOOP

for spatial_mode, s_sigma in spatial_runs:
    print("\n---------------------------------------")
    print(f"Spatial filter: {spatial_mode}   sσ={s_sigma}")
    print("---------------------------------------")

    for i in range(max_r, len(image_files) - max_r):

        curr_color = cv2.imread(str(image_files[i]), cv2.IMREAD_COLOR)

        # SIMPLE temporal derivative
        prev_img = cv2.imread(str(image_files[i - 1]),
                               cv2.IMREAD_GRAYSCALE).astype(np.float32)
        next_img = cv2.imread(str(image_files[i + 1]),
                               cv2.IMREAD_GRAYSCALE).astype(np.float32)

        prev_img = spatial_filter(prev_img, spatial_mode, s_sigma)
        next_img = spatial_filter(next_img, spatial_mode, s_sigma)

        temporal_simple = 0.5 * (next_img - prev_img)

        # DoG temporal filters
        gdA = apply_temporal_filter(i, filtA, rA, spatial_mode, s_sigma)
        gdB = apply_temporal_filter(i, filtB, rB, spatial_mode, s_sigma)
        gdC = apply_temporal_filter(i, filtC, rC, spatial_mode, s_sigma)

        # update derivative displays
        disp_simple.set_data(np.abs(temporal_simple))
        disp_A.set_data(np.abs(gdA))
        disp_B.set_data(np.abs(gdB))
        disp_C.set_data(np.abs(gdC))

        # MOTION MASK + OVERLAY

        if mask_source == "simple":
            grad_abs = np.abs(temporal_simple)
        elif mask_source == "A":
            grad_abs = np.abs(gdA)
        elif mask_source == "B":
            grad_abs = np.abs(gdB)
        else:
            grad_abs = np.abs(gdC)

        sigma_est = np.median(grad_abs) / 0.67449
        T = k_thresh * sigma_est

        motion_mask = (grad_abs >= T).astype(np.uint8)

        overlay = curr_color.copy()
        overlay[motion_mask == 1] = [0, 0, 255]

        # update mask + overlay displays
        mask_im.set_data(motion_mask * 255)
        overlay_im.set_data(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

        fig.suptitle(
            f"Frame {i} | spatial={spatial_mode}, sσ={s_sigma} | source={mask_source}, k={k_thresh}",
            fontsize=14
        )

        plt.pause(pause_time)


plt.ioff()
plt.show()
