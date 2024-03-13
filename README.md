# OCR Model for Digit Classification using MINST Dataset

This project aims to design an OCR (Optical Character Recognition) model to classify digits (from 0-9) using the MINST dataset. The MINST dataset consists of 28x28 grayscale images of handwritten digits.

## Dataset Description
- Dataset: [MINST Dataset](http://yann.lecun.com/exdb/mnist/)
- Classes: Digits from 0 to 9
- Image Size: 28x28 pixels
- Total Images: 70,000 (60,000 training images and 10,000 testing images)

## Model Components
- **Feature Extraction**: Localized Centroid Features (LCF)
- **Classifier**: KNN (K-Nearest Neighbors) classifier with k=1.

## Project Structure
- `data/`: Folder containing the MINST dataset.
- `src/`: Source code files for feature extraction and model training.
- `README.md`: Overview of the project and instructions for replicating the experiment.

## Usage
1. Download the MINST dataset from the provided link and organize it into the `data/` folder.
2. Split the dataset into training and testing sets, using 60,000 images for training and 10,000 images for testing.
3. Implement LCF to extract features from the images.
4. Train the KNN classifier with k=1 using the extracted features.
5. Evaluate the trained model using the testing set and calculate performance metrics.

## Dependencies
- Python 3.x
- Required libraries: NumPy, OpenCV, scikit-image, scikit-learn

## Contributors
- [Ahmed Eltaify](https://github.com/yourusername)
