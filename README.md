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
```
dataset/
├── data.csv
├── Images/
│ ├──Image1
│ ├──Image2
│ ├──Image3
```
- The **CSV file (`data.csv`)** should contain at all the following columns:
  - **Image** → name of the image file (must match the files in `Images/`)
  - **Quality** → quality score for the image
  - **Consistency** → consistency score for the image

# Training
## Required Arguments

| Argument | Type | Description | Example |
|----------|------|-------------|---------|
| `--image_folder` | `str` | Path to the folder containing all images for training/testing. | `"./dataset/Images/"` |
| `--csv_path` | `str` | Path to the CSV or Excel file containing image names and corresponding quality scores (MOS). | `"./dataset/data.csv"` |

---

## Optional Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--patch_size` | `int` | `128` | Size (width and height) of square patches cropped from each image for SURF feature extraction. |
| `--num_patches` | `int` | `30` | Maximum number of SURF patches to extract from each image. |
| `--overlap` | `float` | `0.3` | Maximum allowed overlap ratio between two patches (e.g., `0.3` means 30% overlap). |
| `--hessian_thresh` | `int` | `400` | SURF detector parameter. Higher values detect fewer, stronger keypoints; lower values detect more, including weaker ones. |

---

## Training Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--patience` | `int` | `20` | Number of epochs with no improvement on validation loss before early stopping. |
| `--model_path` | `str` | `"best_model_quality"` | Path to save the best-performing model (lowest validation loss). |
| `--learning_rate` | `float` | `1e-3` | Learning rate for the optimizer. Controls how fast the model updates weights. |

---

## Usage Example

For Quality Score:
```bash
python main_quality.py --mode "train" --image_folder "./dataset/Images" --csv_path "./dataset/data.csv"
```

For Consistency Score:
```bash
python main_consistency.py --mode "train" --image_folder "./dataset/Images" --csv_path "./dataset/data.csv"
```

# Testing

For Quality Score:
```bash
python main_quality.py --mode "test" --image_folder "./dataset/Images" --csv_path "./dataset/data.csv"
```

For Consistency Score:
```bash
python main_consistency.py --mode "test" --image_folder "./dataset/Images" --csv_path "./dataset/data.csv"
```
