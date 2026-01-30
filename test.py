#%%
import pandas as pd
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

df = pd.read_csv('mushroom/mushroom.csv', header=None, names=column_names)

#print(df.head())

#print(f"\nIl dataset ha {df.shape[0]} righe e {df.shape[1]} colonne.")


#Trasformiamo la colonna 'class' in 0 e 1
df['class'] = LabelEncoder().fit_transform(df['class'])

df.drop('veil-type', axis=1, inplace=True)  # Rimuoviamo la colonna 'veil-type' poiché ha un solo valore
df.drop('odor', axis=1, inplace=True)  # Rimuoviamo la colonna 'odor'
df.drop('spore-print-color', axis=1, inplace=True)  # Rimuoviamo la colonna 'spore-print-color'
df.drop('gill-size', axis=1, inplace=True)  # Rimuoviamo la colonna 'gill-size'
df.drop('stalk-root', axis=1, inplace=True)  # Rimuoviamo la colonna 'stalk-root'
#df['stalk-root'] = df['stalk-root'].replace('?', 'missing')  # Gestiamo i valori mancanti

#Trasformiamo il resto delle colonne in numeri (One-Hot Encoding)
X = pd.get_dummies(df.drop('class', axis=1))
y = df['class']

#print(X)

#print(y)


#%%
#Standardizziamo le feature
X_scaled = StandardScaler().fit_transform(X)

#print(X_scaled)



#print("\nPreprocessing completato.")

#%%
print("Modelli in fase di addestramento...")

# Dividiamo il dataset in Training set e Test set (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Creiamo i modelli
dt = DecisionTreeClassifier(max_depth=5) # Albero (limitiamo la profondità per l'overfitting)
knn = KNeighborsClassifier(n_neighbors=5) # k-NN
rf = RandomForestClassifier(n_estimators=100) # Random Forest (100 alberi)

# Li addestriamo (usando i dati divisi precedentemente)
dt.fit(X_train, y_train)
knn.fit(X_train, y_train)
rf.fit(X_train, y_train)

print("Addestramento completato.")
print("Valutazione dei modelli sul Test set...")
# Valutiamo i modelli
dt_accuracy = dt.score(X_test, y_test)
knn_accuracy = knn.score(X_test, y_test)
rf_accuracy = rf.score(X_test, y_test)
print(f"Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")
print(f"k-NN Accuracy: {knn_accuracy * 100:.2f}%")
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
print("Valutazione completata.")

#%%
# 1. 10-Fold Cross Validation per l'Albero di Decisione
cv_scores = cross_val_score(dt, X_scaled, y, cv=10)
print(f"\nDecision Tree CV Accuracy (Media): {cv_scores.mean()*100:.2f}%")
print(f"Decision Tree CV Accuracy (Deviazione Standard): {cv_scores.std()*100:.2f}%")

cv_scores_knn = cross_val_score(knn, X_scaled, y, cv=10)
print(f"KNN CV Accuracy (Media): {cv_scores_knn.mean()*100:.2f}%")
print(f"KNN CV Accuracy (Deviazione Standard): {cv_scores_knn.std()*100:.2f}%")

cv_scores_rf = cross_val_score(rf, X_scaled, y, cv=10)
print(f"Random Forest CV Accuracy (Media): {cv_scores_rf.mean()*100:.2f}%")
print(f"Random Forest CV Accuracy (Deviazione Standard): {cv_scores_rf.std()*100:.2f}%")

# 2. Matrice di Confusione per il k-NN (Esempio)
y_pred_knn = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred_knn)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice di Confusione - k-NN')
plt.xlabel('Predetto')
plt.ylabel('Reale')
plt.show()
# plt.savefig('confusion_matrix_knn.png') # Salviamo l'immagine su file
# print("\nGrafico della matrice di confusione salvato come 'confusion_matrix_knn.png'")

# 3. Report completo (Precision, Recall, F1 - PDF pag. 124)
print("\nReport finale k-NN:")
print(classification_report(y_test, y_pred_knn))