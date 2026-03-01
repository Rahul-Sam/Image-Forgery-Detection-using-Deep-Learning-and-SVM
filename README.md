# Image Forgery Detection using Deep Learning and SVM

---

## 1️ Introduction

With the rapid growth of digital media and social platforms like Instagram and Facebook, image manipulation has become very common. Forged or tampered images can spread misinformation, cause legal issues, and create security threats.

This project aims to build an intelligent system that can automatically detect whether an image is **forged** or **non-forged** using Deep Learning and Machine Learning techniques.

---

## 2️ Problem Statement

Manual verification of image authenticity is time-consuming and unreliable. Therefore, we need an automated system that:

* Accepts digital images
* Extracts meaningful deep features
* Classifies images as Forged or Non-Forged
* Provides performance metrics

---

## 3️ Objectives

* To build a CNN model for feature extraction
* To use SVM for final classification
* To evaluate model performance using standard metrics
* To visualize results using a confusion matrix

---

## 4️ System Architecture

###  Workflow

1. Dataset Collection
2. Image Preprocessing
3. CNN Feature Extraction
4. SVM Classification
5. Performance Evaluation

###  Architecture Explanation

Input Image
→ Preprocessing (Resize, Normalize)
→ CNN (Deep Feature Extraction)
→ SVM Classifier
→ Output (Forged / Non-Forged)

---

## 5️ Technologies Used

| Technology         | Purpose                  |
| ------------------ | ------------------------ |
| Python             | Programming language     |
| OpenCV             | Image processing         |
| NumPy              | Numerical operations     |
| TensorFlow / Keras | CNN model building       |
| Scikit-learn       | SVM & evaluation metrics |
| Matplotlib         | Visualization            |

---

## 6️ Dataset Structure

```
dataset/
│
├── Forged/
│   ├── img1.jpg
│   ├── img2.jpg
│
├── Non_Forged/
│   ├── img1.jpg
│   ├── img2.jpg
```

* Forged → Tampered images
* Non_Forged → Authentic images

---

## 7️ Methodology

###  Step 1: Data Loading

* Images are read in grayscale format.
* Resized to 128x128 pixels.
* Normalized between 0 and 1.
* Labels assigned:

  * 0 → Non_Forged
  * 1 → Forged

---

###  Step 2: CNN Feature Extraction

A Convolutional Neural Network is built with:

* Conv2D layers for feature detection
* MaxPooling for dimensionality reduction
* Dense layer for learning high-level features
* Dropout for overfitting prevention

CNN learns:

* Texture inconsistencies
* Pixel-level anomalies
* Tampering artifacts

Instead of using CNN for final classification, we use it only as a **feature extractor**.

---

###  Step 3: SVM Classification

Extracted features are passed to Support Vector Machine (SVM).

Why SVM?

* Works well on small datasets
* Good margin-based classifier
* High performance in binary classification

Kernel used: Linear

---

###  Step 4: Model Evaluation

The system evaluates performance using:

###  Accuracy

Percentage of correctly classified images.

###  Precision

How many predicted forged images are actually forged.

###  Recall

How many actual forged images were correctly identified.

###  F1 Score

Harmonic mean of Precision and Recall.

---

## 8️ Confusion Matrix

Confusion matrix shows:

|                   | Predicted Non-Forged | Predicted Forged |
| ----------------- | -------------------- | ---------------- |
| Actual Non-Forged | True Negative        | False Positive   |
| Actual Forged     | False Negative       | True Positive    |

This helps in understanding model mistakes.

---

## 9️ Why CNN + SVM Hybrid Approach?

Instead of pure CNN classification:

* CNN extracts deep spatial features.
* SVM performs robust classification.
* Hybrid models often perform better on medium-sized datasets.

Advantages:

* Better generalization
* Reduced overfitting
* Higher accuracy stability

---

##  Applications

* Digital Forensics
* Media Verification
* Cyber Security
* News Authenticity Verification
* Legal Evidence Validation

---

## 1️1 Advantages

* Automated detection
* Good accuracy
* Scalable for larger datasets
* Hybrid approach improves robustness

---

## 1️2️ Limitations

* Requires labeled dataset
* Performance depends on dataset size
* Cannot detect very advanced AI-generated forgeries without large training data

---

## 1️3️ Future Enhancements

* Add GUI using Tkinter
* Use Transfer Learning (e.g., Google pretrained models like ResNet)
* Deploy as Web Application
* Add real-time detection
* Compare with Random Forest and Deep CNN

---

## 1️4️ Conclusion

This project successfully demonstrates an intelligent image forgery detection system using a hybrid Deep Learning + Machine Learning approach.

The CNN extracts meaningful features from images, and SVM classifies them effectively. The system provides strong evaluation metrics including accuracy, precision, recall, and F1 score, making it suitable for academic and practical applications.

---

