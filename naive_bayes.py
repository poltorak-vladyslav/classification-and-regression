import numpy as np
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB

from utilities import visualize_classifier

# Вхідний файл, що містить дані
input_file = 'data_multivar_nb.txt'

# Завантаження даних з вхідного файлу
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Створення класифікатора на основі наївного Байєса
classifier = GaussianNB()

# Навчання класифікатора
classifier.fit(X, y)

# Прогнозування значень для навчальних даних
y_pred = classifier.predict(X)

# Обчислення точності
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Точність класифікатора Наївного Байєса =", round(accuracy, 2), "%")

# Візуалізація ефективності класифікатора
visualize_classifier(classifier, X, y)

###############################################
# Перехресна перевірка

# Поділ даних на навчальні та тестові
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=3)
classifier_new = GaussianNB()
classifier_new.fit(X_train, y_train)
y_test_pred = classifier_new.predict(X_test)

# обчислення точності класифікатора
accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Точність нового класифікатора =", round(accuracy, 2), "%")

# Візуалізація ефективності класифікатора
visualize_classifier(classifier_new, X_test, y_test)

###############################################
# Функції оцінки

num_folds = 3
accuracy_values = model_selection.cross_val_score(classifier, X, y, scoring='accuracy', cv=num_folds)
print("Точність: " + str(round(100*accuracy_values.mean(), 2)) + "%")

precision_values = model_selection.cross_val_score(classifier, X, y, scoring='precision_weighted', cv=num_folds)
print("Достовірність: " + str(round(100*precision_values.mean(), 2)) + "%")

recall_values = model_selection.cross_val_score(classifier, X, y, scoring='recall_weighted', cv=num_folds)
print("Повнота: " + str(round(100*recall_values.mean(), 2)) + "%")

f1_values = model_selection.cross_val_score(classifier, X, y, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100*f1_values.mean(), 2)) + "%")