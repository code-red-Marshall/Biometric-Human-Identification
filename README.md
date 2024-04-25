# Biometric Human Identification for Patient Classification using ANN

This project involves the development of a machine learning model for patient classification. The model is built using Python, TensorFlow, and Keras. The dataset used in this project is a CSV file containing patient data that is being prepared from the public free source dataset named "PTB Diagnostic ECG Database". We have used Pan-Tompkins Algo for R peak Detection using the Moving Integrated Window.

## Pan-Tompkins Plot
![Pan-Tompkins Algo](https://github.com/code-red-Marshall/Biometric-Human-Identification-/assets/82904501/6646b850-d883-49d5-a87b-6878f463e95f)

##Now detecting the R-peaks using the same algo
![R detection Peaks](https://github.com/code-red-Marshall/Biometric-Human-Identification-/assets/82904501/2f733196-91da-49f6-8644-295c8c50e588)

## Project Structure

The project is divided into several sections. Below is a flow chart of the entire project: 

![Block diagram of PM](https://github.com/code-red-Marshall/Biometric-Human-Identification-/assets/82904501/50591d2d-3dcc-40db-ab89-bc7a9d6e3987)


1. **Importing Libraries**: This section involves importing necessary libraries such as numpy, pandas, tensorflow, keras, sklearn, and matplotlib.

2. **Data Preprocessing**: In this section, the dataset is imported and preprocessed. The preprocessing steps include dropping unnecessary columns, label encoding, and splitting the dataset into training and testing sets.

3. **Building the Model**: This section involves building the machine learning model using TensorFlow and Keras. The model is an ANN model with multiple dense layers.

4. **Training the Model**: In this section, the model is trained using the training data. The model's performance is evaluated using accuracy and loss metrics for both training and validation data.

5. **Evaluating the Model**: This section involves evaluating the model's performance using various metrics such as F1-score, precision, recall, and accuracy. The model's performance is also visualized using accuracy and loss plots.

6. **Making Predictions**: In this section, the model is used to make predictions on the training data.

7. **Model Performance Evaluation**: This section involves evaluating the model's performance using a confusion matrix, F1-score, precision, recall, and accuracy. The performance is also visualized using ROC curves.

8. **Visualizing the Data**: This section involves visualizing the features in the dataset using scatter plots.

## Usage

To use this project, follow these steps:

1. Clone the repository to your local machine.
2. Install the necessary libraries mentioned in the 'Importing Libraries' section.
3. Run the Python script to train the model and evaluate its performance.

## Contributing

Contributions to this project are welcome. Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details. 

## Contact

For any queries or suggestions, please open an issue on GitHub.
