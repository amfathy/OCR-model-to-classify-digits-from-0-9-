import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
X_train_flat = X_train_flat / 255.0
X_test_flat = X_test_flat / 255.0

def extract_centroid_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroid_x, centroid_y = 0, 0

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])

    x_coords, y_coords = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    distances = np.sqrt((x_coords - centroid_x)**2 + (y_coords - centroid_y)**2)

    return centroid_x, centroid_y, distances

image_example = cv2.imread('download.png')
centroid_x, centroid_y, distances = extract_centroid_features(image_example)

X_train, X_test, y_train, y_test = train_test_split(X_train_flat, y_train, test_size=0.2, random_state=42)

knn_classifier = KNeighborsClassifier(n_neighbors=1)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)
