# PatchNet
# Prerequisites

Make sure you have the following libraries installed with the specified versions:

- **torch**: 2.7.1+cu118  
- **numpy**: 2.3.0  
- **pandas**: 2.3.0  
- **networkx**: 3.5.1  
- **torchvision**: 0.22.1+cu118  
- **transformers**: 4.53.2  
- **clip**  
- **PIL**: 11.2.1  
- **scikit-learn**: 1.7.0  
- **torch-geometric**: 2.6.1  
- **scipy**: 1.15.3  


⚠️ **Note on SURF:**  
The SURF (Speeded-Up Robust Features) algorithm is patented, so it is **not included** in the default OpenCV or `opencv-contrib-python` packages.  
To use SURF (`cv2.xfeatures2d.SURF_create()`), you must **build OpenCV from source with contrib modules** using **CMake**.  

Steps (summary):
1. Clone OpenCV and opencv_contrib repositories.  
2. Run CMake with `-DOPENCV_ENABLE_NONFREE=ON` and set the contrib path.  
3. Build and install OpenCV.  

# Prepare Data

To run the program, organize your dataset in the following format:

dataset/
├── data.csv
├── Images/
│ ├──Image1
│ ├──Image2
│ ├──Image3

- The **CSV file (`data.csv`)** should contain at all the following columns:
  - **Image** → name of the image file (must match the files in `Images/`)
  - **Quality** → quality score for the image
  - **Consistency** → consistency score for the image