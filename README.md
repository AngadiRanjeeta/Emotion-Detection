# Emotion Detection from Facial Images

This repository contains the code for an emotion detection model using Convolutional Neural Networks (CNNs) in TensorFlow/Keras. The model is trained and evaluated on a dataset of facial images labeled with emotions (e.g., happy, sad, angry, neutral).

## Setup

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/AngadiRanjeeta/Emotion-Detection.git
    cd Emotion-Detection
    ```

2. **Open the Colab Notebook**:
    - Open `emotion_detection.ipynb` in Google Colab.

3. **Install Dependencies**:
    - Install the required dependencies by running:
      ```sh
      !pip install tensorflow numpy pandas matplotlib
      ```

4. **Upload Dataset**:
    - [Download the dataset](https://drive.google.com/file/d/1G36Lpdgq3ha1vbMcI4yOFLGNhVZ_6glG/view).
    - Upload the downloaded dataset to your Google Drive and mount it in the Colab notebook as shown in the code.
  
5. **Run the Notebook**:
    - Execute the cells in the notebook sequentially.

## Results

The model achieves the following performance:
- **Validation Accuracy**: X.XX
- **Test Accuracy**: X.XX

## Visualizations

Training and validation accuracy and loss curves are provided to visualize the training process.

## Dependencies

- TensorFlow
- NumPy
- Pandas
- Matplotlib

## **Summary of Approach and Methodology**
    
 **Data Loading and Preprocessing:**

- Used ImageDataGenerator to load and preprocess images for training, validation, and testing.
- Applied data augmentation techniques like rescaling and horizontal flipping to increase the diversity of the training set.

**Model Architecture:**

- Designed a CNN with three convolutional layers followed by max-pooling and dropout layers to reduce overfitting.
- Added dense layers for classification with a final softmax layer for outputting probabilities for each emotion class.

**Training and Optimization:**

- Used Adam optimizer for training and categorical cross-entropy as the loss function.
- Implemented callbacks such as ReduceLROnPlateau, ModelCheckpoint, and EarlyStopping for better training management and optimization.

**Evaluation and Testing:**

- Evaluated the model on validation and test sets, reporting accuracy.
- Generated and visualized a confusion matrix and classification report to understand the performance and misclassifications.

## License

This project is licensed under the MIT License.
