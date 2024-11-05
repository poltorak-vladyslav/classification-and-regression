import numpy as np

from sklearn import linear_model
from utilities import visualize_classifier

# Ознаки: [години навчання, попередня оцінка]
X = np.array([
    [2.0, 70], [3.5, 65], [5.1, 80],
    [4.0, 75], [6.0, 85], [7.2, 88],
    [1.5, 60], [2.8, 67], [6.5, 82],
    [8.0, 90], [9.0, 93], [10.5, 95]
])

# Ціль: оцінка на тесті
y = np.array([72, 75, 82, 78, 88, 90, 68, 74, 85, 92, 95, 98])

# Створити класифікатор логістичної регресії
classifier = linear_model.LogisticRegression(solver='liblinear', C=1)

# Навчити класифікатор
classifier.fit(X, y)

# Візуалізувати ефективність класифікатора
visualize_classifier(classifier, X, y)