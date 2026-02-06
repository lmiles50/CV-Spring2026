from pathlib import Path
import cv2
import matplotlib.pyplot as plt


#NOTES:
#tried using cv2.imshow but im on windows and theres a backend GUI problem so i resorted to matplotlib
#next step 1D filter 0.5[-1, 0, 1]

#to avoid redoing path every time one of us works on it 
images_dir = Path(__file__).parent / "Office" / "Office"

print(f"counting images in: {images_dir}")

try:
    image_files = sorted(images_dir.glob("*.jpg"))

    for img_path in image_files:
        print(img_path)
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #-----------if you want to see that its converting to grayscale uncomment these guys: 
        #plt.imshow(gray, cmap="gray")
        #plt.title(img_path.name)
        #plt.axis("off")
        #plt.show()


        #t = input("input gaussian standard deviation value: ")

except FileNotFoundError:
    print("Ensure Office folder is in the same directory as Project1.py!")
    raise
