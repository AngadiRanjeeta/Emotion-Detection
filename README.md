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
- **Validation Accuracy**: 0.5634
- **Test Accuracy**: 0.4947

## Visualizations

Training and validation accuracy and loss curves are provided to visualize the training process.

## Dependencies

- TensorFlow
- Keras
- Matplotlib
- Numpy
- Sklearn
- Seaborn

## Approach and Methodology

1. **Data Preparation:**
    - Defined dataset paths for training, validation, and test sets.
    - Utilized `ImageDataGenerator` for data augmentation (horizontal flipping) and normalization.

2. **Model Architecture:**
    - Built a Convolutional Neural Network (CNN) with three convolutional layers followed by max-pooling and dropout layers to prevent overfitting.
    - Used a fully connected layer with dropout before the final softmax layer to classify emotions into 7 categories.

3. **Model Compilation:**
    - Compiled the model using the Adam optimizer and categorical cross-entropy loss, with accuracy as the evaluation metric.

4. **Training:**
    - Set up callbacks: ReduceLROnPlateau to adjust the learning rate, ModelCheckpoint to save the best model, and EarlyStopping to stop training if the model's performance didn't improve for a specified number of epochs.
    - Trained the model on the training data while validating it on the validation set.

5. **Evaluation:**
    - Evaluated the model's performance on both the validation and test sets to calculate accuracy.
    - Plotted the training history to visualize accuracy and loss over epochs.

6. **Predictions and Analysis:**
    - Predicted emotions on the test set and calculated the accuracy.
    - Generated a confusion matrix and classification report to evaluate performance across all classes.
    - Displayed some misclassified examples for further analysis.

## License

This project is licensed under the MIT License.
