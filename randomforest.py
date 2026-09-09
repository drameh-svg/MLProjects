'''
Jordan Rivera
ACAD 222, Spring 2026
jrivera@usc.edu
Final Project Part 2 - Decision Tree / Random Forest (Chest X-Ray Pneumonia Classifier)
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# Load and flatten images into pixel vectors
def load_flat_images(path, n_samples=1000):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, batch_size=n_samples, shuffle=True)
    imgs, labels = next(iter(loader))
    X = imgs.numpy().reshape(len(imgs), -1)
    y = labels.numpy()
    return X, y


image_path = './chest_xray'
X_train, y_train = load_flat_images(image_path + '/train', n_samples=1000)

print('Train set shape:', X_train.shape)
print('Classes: 0=NORMAL, 1=PNEUMONIA')

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

pipe_dt = make_pipeline(
    DecisionTreeClassifier(random_state=1)
)
'''
# ============================================================
# start of first example - Decision Tree baseline (CV)
pipe_dt = make_pipeline(
    DecisionTreeClassifier(random_state=1)
)

scores = cross_val_score(estimator=pipe_dt,
                         X=X_train,
                         y=y_train,
                         cv=cv,
                         n_jobs=-1)

print(f'DT CV accuracy scores: {scores}')
print(f'DT CV accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
# end of first example
# ============================================================
'''

'''
# ============================================================
# start of second example - RandomizedSearchCV (Decision Tree)

param_dist = {
    'decisiontreeclassifier__max_depth': [3, 5, 10, 20, None],
    'decisiontreeclassifier__min_samples_split': [2, 5, 10],
    'decisiontreeclassifier__min_samples_leaf': [1, 2, 4],
    'decisiontreeclassifier__criterion': ['gini', 'entropy']
}

rs = RandomizedSearchCV(estimator=pipe_dt,
                        param_distributions=param_dist,
                        scoring='accuracy',
                        refit=True,
                        n_iter=20,
                        cv=cv,
                        random_state=1,
                        n_jobs=-1)

rs.fit(X_train, y_train)

print('Best score:', rs.best_score_)
print('Best params:', rs.best_params_)
# end of second example
# ============================================================
'''

'''
# ============================================================
# start of third example - Random Forest (CV)

pipe_rf = make_pipeline(
    RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=1,
        n_jobs=-1
    )
)

scores = cross_val_score(estimator=pipe_rf,
                         X=X_train,
                         y=y_train,
                         cv=cv,
                         n_jobs=-1)

print(f'RF CV accuracy scores: {scores}')
print(f'RF CV accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
# end of third example
# ============================================================
'''


'''
# ============================================================
# start of fourth example - ROC curve (cross-validation)
pipe_rf = make_pipeline(
    RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=1,
        n_jobs=-1
    )
)
from sklearn.model_selection import cross_val_predict

y_proba = cross_val_predict(pipe_rf,
                           X_train,
                           y_train,
                           cv=cv,
                           method='predict_proba',
                           n_jobs=-1)[:, 1]

fpr, tpr, thresholds = roc_curve(y_train, y_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr,
         label=f'Random Forest (area = {roc_auc:.2f})')

plt.plot([0, 1],
         [0, 1],
         linestyle='--',
         color=(0.6, 0.6, 0.6),
         label='Random guessing (area = 0.5)')

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='lower right')
plt.title('ROC Curve - Random Forest')

plt.tight_layout()
plt.show()
# end of fourth example
# ============================================================
'''

'''
pipe_rf = make_pipeline(
    RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=1,
        n_jobs=-1
    ))
# ============================================================
# start of fifth example - Learning curve (Random Forest)

train_sizes, train_scores, test_scores = learning_curve(
    estimator=pipe_rf,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=cv,
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         marker='o',
         label='Training accuracy')

plt.plot(train_sizes, test_mean,
         marker='s',
         linestyle='--',
         label='Validation accuracy')

plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Learning Curve - Random Forest')

plt.tight_layout()
plt.show()
# end of fifth example
# ============================================================
'''


# start of sixth example - Learning curve (Random Forest, regularized)

pipe_rf = make_pipeline(
    RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=1,
        n_jobs=-1
    )
)

train_sizes, train_scores, test_scores = learning_curve(
    estimator=pipe_rf,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=cv,
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         marker='o',
         label='Training accuracy')

plt.plot(train_sizes, test_mean,
         marker='s',
         linestyle='--',
         label='Validation accuracy')

plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Learning Curve - Random Forest (Regularized)')

plt.tight_layout()
plt.show()
# end of sixth example


