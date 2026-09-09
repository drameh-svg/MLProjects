'''
Jordan Rivera
ACAD 222, Spring 2026
jrivera@usc.edu
Final Project Part 2 - SVM (Chest X-Ray Pneumonia Classifier)
'''

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# Load and flatten images into pixel vectors for SVM
# SVM cannot work on raw image tensors - we resize to 32x32 and flatten
# to keep the feature vector manageable (32*32 = 1024 features per image)
def load_flat_images(path, n_samples=1000):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, batch_size=n_samples, shuffle=True)
    imgs, labels = next(iter(loader))
    X = imgs.numpy().reshape(len(imgs), -1)  # flatten to (n_samples, 1024)
    y = labels.numpy().astype(np.float64)  # 0=NORMAL, 1=PNEUMONIA
    return X, y


image_path = './chest_xray'
X_train, y_train = load_flat_images(image_path + '/train', n_samples=1000)
X_test, y_test = load_flat_images(image_path + '/test', n_samples=500)

print('Train set shape:', X_train.shape)
print('Test set shape: ', X_test.shape)
print('Classes: 0=NORMAL, 1=PNEUMONIA')

# start of first example
# Linear SVM - equivalent to the Iris LinearSVC example
# C=1 with hinge loss, same as chapter 5
svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge", max_iter=5000)),
])
svm_clf.fit(X_train, y_train)
print('Linear SVM accuracy:', svm_clf.score(X_test, y_test))
# end of first example


'''
#Start of second example
# Polynomial features + LinearSVC - equivalent to the Moons example
# Adds polynomial combinations of pixel features before training
polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=2)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", max_iter=5000))
    ])
polynomial_svm_clf.fit(X_train, y_train)
print('Polynomial SVM accuracy:', polynomial_svm_clf.score(X_test, y_test))
#end of second example
'''

'''
#start of third example
# Polynomial kernel SVM - uses kernel trick instead of explicit poly features
# Much faster than second example for high-dimensional image data
poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])
poly_kernel_svm_clf.fit(X_train, y_train)
print('Poly kernel SVM accuracy:', poly_kernel_svm_clf.score(X_test, y_test))
#end of third example
'''

# start of fourth example
# RBF (Gaussian) kernel SVM - best general-purpose kernel for image data
# gamma controls how tightly each training example influences the boundary
# C controls the margin - same tradeoff as in the chapter 5 slides
# probability=True enables predict_proba() which is required for ROC curve
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=0.001, C=10.0, probability=True))
])
rbf_kernel_svm_clf.fit(X_train, y_train)
print('RBF kernel SVM accuracy:', rbf_kernel_svm_clf.score(X_test, y_test))

RocCurveDisplay.from_estimator(rbf_kernel_svm_clf, X_test, y_test)
plt.title('ROC Curve - RBF Kernel SVM')
plt.show()
# end of fourth example
