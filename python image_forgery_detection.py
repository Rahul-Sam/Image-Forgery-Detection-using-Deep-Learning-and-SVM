 Install Requirements First

pip install numpy opencv-python matplotlib scikit-learn tensorflow

 FULL WORKING CODE

python

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------
IMG_SIZE = 128
DATASET_PATH = "dataset"
EPOCHS = 10
BATCH_SIZE = 16

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------
def load_dataset():
    X = []
    y = []

    print("Loading images...")

    for label, folder in enumerate(["Non_Forged", "Forged"]):
        folder_path = os.path.join(DATASET_PATH, folder)

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"{folder_path} not found!")

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label)

    X = np.array(X, dtype="float32") / 255.0
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)

    print("Total Images Loaded:", len(X))

    return X, y

# ---------------------------------------------------
# BUILD CNN MODEL (Feature Extractor)
# ---------------------------------------------------
def build_cnn():
    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))

    model.compile(
        optimizer=Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ---------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------
if __name__ == "__main__":

    # 1️ Load Dataset
    X, y = load_dataset()

    # 2️ Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3️ Train CNN
    print("\nTraining CNN...")
    cnn = build_cnn()

    cnn.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )

    # 4️ Feature Extraction (Remove last classification layer)
    print("\nExtracting Features...")
    feature_extractor = Sequential(cnn.layers[:-1])

    train_features = feature_extractor.predict(X_train)
    test_features = feature_extractor.predict(X_test)

    # 5️ Train SVM
    print("\nTraining SVM...")
    svm = SVC(kernel='linear')
    svm.fit(train_features, y_train)

    predictions = svm.predict(test_features)

    # ---------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\n==========================")
    print("     FINAL RESULTS")
    print("==========================")
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1 Score : {f1 * 100:.2f}%")

    # ---------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.show()
```

---

#  IMPORTANT – Dataset Folder Structure

Make sure this exists in same folder:

```
dataset/
│
├── Forged/
├── Non_Forged/
```

---

#  How To Run

```bash
python image_forgery_detection.py
```

