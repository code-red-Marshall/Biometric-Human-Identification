# Section 1: Importing Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from keras.utils.vis_utils import plot_model

# Section 2: Data Preprocessing
# Importing the dataset
dataset = pd.read_csv('dataset.csv')
X=dataset.drop(labels=['PATIENT_NAME'],axis=1)
Y=dataset['PATIENT_NAME']

# Encoding the labels
from sklearn.preprocessing import LabelBinarizer
le = LabelBinarizer()
Y=le.fit_transform(Y)

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# Section 3: Building the Model
model= Sequential()
model.add(Dense(X.shape[1],activation='relu',input_dim=X.shape[1]))
model.add(Dense(150, activation='relu'))
model.add(Dense(Y.shape[1],activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Section 4: Training the Model
history=model.fit(X_train,Y_train,batch_size=10,epochs=42,verbose=1,validation_data=(X_train,Y_train))

# Section 5: Evaluating the Model
# Plotting the model accuracy and loss
hist_df = pd.DataFrame(history.history)
plt.plot(hist_df['accuracy'])
plt.plot(hist_df['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='upper left')
plt.show()

plt.figure()
plt.plot(hist_df['loss'])
plt.plot(hist_df['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='upper left')
plt.show()

# Plotting the model structure
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Section 6: Making Predictions
Y_pred=model.predict(X_train)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_test_classes = np.argmax(Y_train, axis=1)

# Section 7: Model Performance Evaluation
# Confusion Matrix
cm = confusion_matrix(Y_test_classes, Y_pred_classes)
print(cm)

# F1-score, Precision, Recall, Accuracy
f1 = f1_score(Y_test_classes, Y_pred_classes, average='macro')
precision = precision_score(Y_test_classes, Y_pred_classes, average='macro')
recall = recall_score(Y_test_classes, Y_pred_classes, average='macro')
accuracy = accuracy_score(Y_test_classes, Y_pred_classes)
print("F1-score:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)

# ROC Curve
Y_test_bin = label_binarize(Y_test_classes, classes=np.unique(Y_test_classes))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(Y_test_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(Y_test_bin[:, i], Y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
for i in range(Y_test_bin.shape[1]):
    plt.plot(fpr[i], tpr[i], label='Class {} (AUC = {:.2f})'.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Section 8: Visualizing the Data
feature1 = X_train[:, 0]
feature2 = X_train[:, 1]
plt.scatter(feature1, feature2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
