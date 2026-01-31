import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

column_names = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
    "ring-type", "spore-print-color", "population", "habitat"
]

def save_confusion_matrix(model, x_test, y_test, title, filename):
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice di Confusione - {title}')
    plt.xlabel('Predetto')
    plt.ylabel('Reale')
    plt.savefig(filename)
    plt.close()
    return cm

def cross_validation_score(model, x_data, y_data, cv=10):
    scores = cross_val_score(model, x_data, y_data, cv=cv)
    return scores.mean(), scores.std()

def model_accuracy(model, x_test, y_test):
    return model.score(x_test, y_test)

def train_tree_model(x_train, y_train):
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(x_train, y_train)
    return model

def train_knn_model(x_train, y_train):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train, y_train)
    return model

def train_rf_model(x_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(x_train, y_train)
    return model

def preprocces_data(input_data):
    
    input_data['class'] = LabelEncoder().fit_transform(input_data['class'])
    
    input_data.drop('veil-type', axis=1, inplace=True)  # Rimuoviamo la colonna 'veil-type' poiché ha un solo valore
    
    # Remove or handle other columns if necessary
    # input_data.drop('odor', axis=1, inplace=True)
    # input_data.drop('gill-color', axis=1, inplace=True)
    # input_data.drop('gill-size', axis=1, inplace=True)
    # input_data.drop('spore-print-color', axis=1, inplace=True)
    # input_data.drop('ring-type', axis=1, inplace=True)
    # input_data.drop('stalk-root', axis=1, inplace=True)
    # input_data.drop('population', axis=1, inplace=True)
    # input_data.drop('bruises', axis=1, inplace=True)
    
    input_data['stalk-root'] = input_data['stalk-root'].replace('?', 'missing')  # Gestiamo i valori mancanti
    
    x = pd.get_dummies(input_data.drop('class', axis=1))
    y = input_data['class']
    x_scaled = StandardScaler().fit_transform(x)
    
    return x_scaled, y

def main():
    # Caricamento del dataset
    df = pd.read_csv('mushroom/mushroom.csv', header=None, names=column_names)
    
    # Preprocessing dei dati e suddivisione in train e test set
    print("\nPreprocessing dei dati...")
    x_scaled, y = preprocces_data(df)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
    
    print("\nAddestramento dei modelli...\n")
    
    # Decision Tree training with time measurement
    start = time.time()
    dt = train_tree_model(x_train, y_train)
    dt_time = time.time() - start
    print(f"Decision Tree training time: {dt_time:.4f}s")

    # k-NN training with time measurement
    start = time.time()
    knn = train_knn_model(x_train, y_train)
    knn_time = time.time() - start
    print(f"k-NN training time: {knn_time:.4f}s")

    # Random Forest training with time measurement
    start = time.time()
    rf = train_rf_model(x_train, y_train)
    rf_time = time.time() - start
    print(f"Random Forest training time: {rf_time:.4f}s")
    print("\nAddestramento completato.\n")
    
    # Model evaluation on test set
    dt_accuracy = model_accuracy(dt, x_test, y_test)
    knn_accuracy = model_accuracy(knn, x_test, y_test)
    rf_accuracy = model_accuracy(rf, x_test, y_test)
    
    print("Valutazione dei modelli sul Test set...\n")
    print(f"Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")
    print(f"k-NN Accuracy: {knn_accuracy * 100:.2f}%")
    print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
    
    # Cross-validation scores for each model
    print("\nCross-Validation dei modelli...\n")
    dt_cv_mean, dt_cv_std = cross_validation_score(dt, x_scaled, y)
    knn_cv_mean, knn_cv_std = cross_validation_score(knn, x_scaled, y)
    rf_cv_mean, rf_cv_std = cross_validation_score(rf, x_scaled, y)
    
    print(f"Decision Tree CV Accuracy (Media): {dt_cv_mean * 100:.2f}%")
    print(f"Decision Tree CV Accuracy (Deviazione Standard): {dt_cv_std * 100:.2f}%\n")
    
    print(f"k-NN CV Accuracy (Media): {knn_cv_mean * 100:.2f}%")
    print(f"k-NN CV Accuracy (Deviazione Standard): {knn_cv_std * 100:.2f}%\n")
    
    print(f"Random Forest CV Accuracy (Media): {rf_cv_mean * 100:.2f}%")
    print(f"Random Forest CV Accuracy (Deviazione Standard): {rf_cv_std * 100:.2f}%\n")
    
    print("\nSalvataggio matrici di confusione...")
    save_confusion_matrix(dt, x_test, y_test, "Decision Tree", "confusion_matrix_dt.png")
    save_confusion_matrix(knn, x_test, y_test, "k-NN", "confusion_matrix_knn.png")
    save_confusion_matrix(rf, x_test, y_test, "Random Forest", "confusion_matrix_rf.png")
    print("\nMatrici salvate come immagini PNG.")
    
if __name__ == "__main__":
    main()
    
    
# 2. Matrice di Confusione per il k-NN (Esempio)
# y_pred_knn = knn.predict(X_test)
# cm = confusion_matrix(y_test, y_pred_knn)

# plt.figure(figsize=(6,4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Matrice di Confusione - k-NN')
# plt.xlabel('Predetto')
# plt.ylabel('Reale')
# plt.show()