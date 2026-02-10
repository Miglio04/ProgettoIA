import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from matplotlib.ticker import AutoMinorLocator
from sklearn.inspection import permutation_importance

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
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

# Salva la matrice di confusione come un heatmap con annotazioni personalizzate per classificazione binaria
def save_confusion_matrix(model, x_test, y_test, class_names, title, filename):
    # Prevede le etichette per il set di test
    y_pred = model.predict(x_test)
    # Calcola la matrice di confusione
    cm = confusion_matrix(y_test, y_pred)
    
    # Crea etichette personalizzate per le celle se è una classificazione binaria
    if len(class_names) == 2:
        names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        # matrice "schicciata" in una lista piatta di 4 numeri: TN, FP, FN, TP
        counts = cm.flatten()
        # calcola le percentuali per ogni cella, dividendo ogni conteggio per la somma totale dei campioni nella matrice
        percentages = counts / np.sum(cm)
        # crea etichette che mostrano il nome, il conteggio e la percentuale per ogni cella
        labels = [f"{n}\n{c:.0f}\n{p:.2%}" for n, c, p in zip(names, counts, percentages)]
        # prende la lista appena creata e la trasforma in una matrice 2x2 per poterla usare come annotazione nella heatmap
        annot = np.asarray(labels).reshape(2, 2)
        # fmt è una stringa che indica il formato dei numeri nelle annotazioni. Se è vuota, i 
        # numeri vengono visualizzati come sono (ad esempio, "10").
        fmt = ''
    else:
        #scrivi i numeri nelle celle
        annot = True
        #Se è 'd', i numeri vengono formattati come interi (ad esempio, "10" invece di "10.0").
        fmt = 'd'

    plt.figure(figsize=(8, 6))
    # qui viene creata la heatmap della matrice di confusione, con le annotazioni dentro ogni cella e la mappa dei colori 'Blues'. 
    # Gli xticklabels e yticklabels sono impostati sui nomi delle classi per rendere più chiara la visualizzazione.
    sns.heatmap(cm, annot=annot, fmt=fmt, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matrice di Confusione - {title}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(filename)
    plt.close()
    return cm

# utilizzo del n-fold cross-validation per valutare il modello
def cross_validation_score(model, x_data, y_data, cv=10):
    scores = cross_val_score(model, x_data, y_data, cv=cv)
    # restituisce la media e la deviazione standard dei punteggi di accuratezza ottenuti
    return scores.mean(), scores.std()

# Calcolo accuratezza del modello
def model_accuracy(model, x_test, y_test):
    return model.score(x_test, y_test)

# Calcola Precision e Recall
def calculate_and_print_metrics(model, x_test, y_test, title):
    y_pred = model.predict(x_test)
    
    # Precision: Tra tutti quelli predetti positivi, quanti lo erano davvero?
    precision = precision_score(y_test, y_pred)
    # Recall (Sensitivity): Tra tutti i positivi reali, quanti ne ho trovati?
    recall = recall_score(y_test, y_pred)
        
    print(f"\n--- Metriche dettagliate per {title} ---")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    
    return precision, recall

# Train Decision Tree model
def train_tree_model(x_train, y_train):
    # max_depth è un iperparametro che limita la profondità massima dell'albero di decisione. 
    # Impostandolo a 5, stiamo dicendo al modello di non creare alberi troppo profondi, il che può 
    # aiutare a prevenire l'overfitting e migliorare la generalizzazione del modello sui dati di test.
    model = DecisionTreeClassifier(max_depth=5)
    # fit addestra il modello sui dati di addestramento (x_train e y_train)
    model.fit(x_train, y_train)
    return model

# Train k-NN model
def train_knn_model(x_train, y_train):
    # n_neighbors è un iperparametro che specifica il numero di vicini da considerare per fare una previsione.
    # Impostando n_neighbors=5, stiamo dicendo al modello di guardare i 5 campioni più vicini nel set di addestramento per determinare la classe di un nuovo campione.
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train, y_train)
    return model

# Train Random Forest model
def train_rf_model(x_train, y_train):
    # n_estimators è un iperparametro che specifica il numero di alberi nella foresta.
    model = RandomForestClassifier(n_estimators=10)
    model.fit(x_train, y_train)
    return model

# Preprocess data
def preprocces_data(input_data):
    
    # la LabelEncoder trasforma le etichette di classe da stringhe a numeri interi (0 e 1)
    le = LabelEncoder()
    # qui viene applicato il LabelEncoder alla colonna 'class' del dataset
    input_data['class'] = le.fit_transform(input_data['class'])
    
    # Recupera i nomi originali delle classi prima della codifica
    original_classes = le.classes_
    class_map = {'e': 'Edible', 'p': 'Poisonous'}
    # mappa i nomi originali alle etichette desiderate
    class_names = [class_map.get(str(c), str(c)) for c in original_classes]
    
    # Rimuoviamo la colonna 'veil-type' poiché ha un solo valore
    input_data.drop('veil-type', axis=1, inplace=True)  

    # Colonne rimosse per testare l'impatto sulla performance e sulla precisione
    # input_data.drop('odor', axis=1, inplace=True)
    # input_data.drop('gill-color', axis=1, inplace=True)
    # input_data.drop('gill-size', axis=1, inplace=True)
    # input_data.drop('spore-print-color', axis=1, inplace=True)
    # input_data.drop('ring-type', axis=1, inplace=True)
    # input_data.drop('stalk-root', axis=1, inplace=True)
    # input_data.drop('population', axis=1, inplace=True)
    # input_data.drop('bruises', axis=1, inplace=True)
    
    input_data['stalk-root'] = input_data['stalk-root'].replace('?', 'missing')  # Gestiamo i valori mancanti
    
    # Creiamo le variabili dummy per tutte le colonne tranne 'class'
    x = pd.get_dummies(input_data.drop('class', axis=1))
    # Recuperiamo i nomi delle feature dopo la codifica one-hot
    feature_names = x.columns.tolist()
    # La variabile target y è la colonna 'class' del dataset, che contiene le etichette di classe codificate come numeri interi
    y = input_data['class']
    # Questo di preciso è il passaggio in cui i dati vengono scalati usando lo StandardScaler,
    # che normalizza le feature per avere media 0 e deviazione standard 1.
    x_scaled = StandardScaler().fit_transform(x)
    
    return x_scaled, y, feature_names, class_names

# Visualizza l'albero di decisione
def visualize_decision_tree(model, feature_names, class_names, filename):
    # Imposta la dimensione della figura per la visualizzazione dell'albero
    plt.figure(figsize=(20, 10))
    # plot_tree crea la rappresentazione grafica dell'albero
    plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=10)
    plt.title("Visualizzazione Albero di Decisione")
    plt.savefig(filename)
    plt.close()

# Funzione per visualizzare i confini decisionali del modello k-NN dopo aver ridotto le dimensioni a 2 usando PCA
def visualize_knn_boundaries(x_train, y_train, k, class_names, filename='knn_decision_boundaries.png'):
    # Riduciamo le dimensioni a 2 usando PCA per poter visualizzare i confini decisionali in un piano 2D
    pca = PCA(n_components=2)
    x_reduced = pca.fit_transform(x_train)

    # Addestriamo un nuovo KNN sui dati ridotti a 2D (perdendo informazioni di tutte le altre feature, quindi questo è solo per scopi di visualizzazione e non rappresenta la performance reale del modello)
    knn_2d = KNeighborsClassifier(n_neighbors=k)
    knn_2d.fit(x_reduced, y_train)
    
    # Calcoliamo lo score del modello 2D per mostrarlo
    score_2d = knn_2d.score(x_reduced, y_train)

    # Creiamo una griglia di punti che copre l'intero spazio 2D per visualizzare i confini decisionali
    h = .02  # step size nella griglia
    x_min, x_max = x_reduced[:, 0].min() - 1, x_reduced[:, 0].max() + 1
    y_min, y_max = x_reduced[:, 1].min() - 1, x_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Prevediamo le classi per ogni punto nella griglia usando il modello k-NN addestrato sui dati 2D
    Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Convertiamo y_train in un array numpy se è un pandas Series, altrimenti lo lasciamo com'è (ad esempio, se è già un array numpy)
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train

    # Creiamo le mappe di colori per le regioni e i punti
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA']) # colori chiari per le regioni decisionali (rosso chiaro per una classe, verde chiaro per l'altra)
    cmap_bold = ListedColormap(['#FF0000', '#00FF00']) # colori più forti per i punti di addestramento (rosso per una classe, verde per l'altra)

    plt.figure(figsize=(12, 10))
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.3) # contourf disegna le regioni colorate che rappresentano i confini decisionali del modello k-NN nel piano 2D
    
    # Scatter plot dei punti di addestramento
    scatter = plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=y_train_np, 
                          cmap=cmap_bold, edgecolor='k', s=20)
    
    plt.legend(handles=scatter.legend_elements()[0], labels=class_names, title="Classes")
    plt.title(f"KNN Boundaries (2D PCA) - Acc. Visiva: {score_2d:.2%}\n(Il modello reale usa tutte le feature ed è più accurato)")
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    # Miglioramento della scala e della griglia
    plt.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.8)
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    
    # Aumenta la frequenza dei tick sugli assi se necessario (opzionale, ma aiuta la precisione)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())

    plt.savefig(filename, dpi=300)
    plt.close()

# Funzione per calcolare e visualizzare l'importanza delle feature per i modelli che la supportano direttamente (come Decision Tree e Random Forest) o tramite Permutation Importance (come k-NN)
def plot_feature_importance(model, feature_names, title, filename, x_data=None, y_data=None, top_n=20):
    
    if hasattr(model, 'feature_importances_'):
        # Se il modello supporta direttamente la feature importance (come Decision Tree e Random Forest), usiamo quella
        importances = model.feature_importances_
    elif x_data is not None and y_data is not None:
        # Se il modello non supporta la feature importance diretta (come k-NN), ma sono stati forniti i dati, calcoliamo la Permutation Importance
        print(f"Calcolo Permutation Importance per {title}")
        # La Permutation Importance misura l'importanza di una feature valutando quanto peggiora la performance del modello quando i valori di quella feature vengono mescolati casualmente.
        # n_repeats determina quante volte mescolare ogni feature. Più è alto, più è stabile ma lento.
        result = permutation_importance(model, x_data, y_data, n_repeats=10, random_state=42, n_jobs=-1)
        importances = result.importances_mean
    else:
        print(f"Il modello {title} non supporta la feature importance diretta e non sono stati forniti dati per la permutation importance.")
        return

    # Ordina le feature in base all'importanza e prendi solo le top_n più importanti
    indices = np.argsort(importances)[::-1][:top_n]
    
    # Creazione del grafico a barre per visualizzare l'importanza delle feature
    plt.figure(figsize=(12, 8))
    plt.title(f"Top {top_n} Feature Importance - {title}")
    plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, len(indices)])
    plt.tight_layout()
    plt.ylabel("Importanza")
    plt.savefig(filename)
    plt.close()

def main():

    inizio = time.time()

    # Caricamento del dataset
    df = pd.read_csv('datasets/mushroom/mushroom.csv', header=None, names=column_names)
    
    # Preprocessing dei dati e suddivisione in train e test set
    print("\nPreprocessing dei dati...")
    # i valori che ritorniamo da questa funzione sono: x_scaled (le feature scalate),
    # y (le etichette di classe), feature_names (i nomi delle feature dopo la codifica one-hot) e 
    # class_names (i nomi delle classi originali mappati a 'Edible' e 'Poisonous').
    x_scaled, y, feature_names, class_names = preprocces_data(df)
    # suddividiamo i dati in un set di addestramento (80%) e un set di test (20%) usando la funzione train_test_split di scikit-learn, 
    # con un random_state fisso per garantire la riproducibilità.
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
    print("-------------------------------------------------------------------")
    print("\nAddestramento dei modelli...\n")
    print("Addestramento del Decision Tree...\n")
    # addestramento Decision Tree
    start = time.time()
    dt = train_tree_model(x_train, y_train)
    dt_time = time.time() - start
    print(f"Decision Tree training time: {dt_time:.4f}s")

    print("\nAddestramento del k-NN...\n")
    # addestramento k-NN
    start = time.time()
    knn = train_knn_model(x_train, y_train)
    knn_time = time.time() - start
    print(f"k-NN training time: {knn_time:.4f}s")

    print("\nAddestramento del Random Forest...\n")
    # addestramento Random Forest 
    start = time.time()
    rf = train_rf_model(x_train, y_train)
    rf_time = time.time() - start
    print(f"Random Forest training time: {rf_time:.4f}s")
    print("\nAddestramento completato.\n")
    print("-------------------------------------------------------------------")

    # Calcolo e stampa delle accuratezze sui test set
    dt_accuracy = model_accuracy(dt, x_test, y_test)
    knn_accuracy = model_accuracy(knn, x_test, y_test)
    rf_accuracy = model_accuracy(rf, x_test, y_test)
    
    print("Valutazione dei modelli sul Test set...\n")
    print(f"Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")
    print(f"k-NN Accuracy: {knn_accuracy * 100:.2f}%")
    print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

    print("\nCalcolo di Precision e Recall per ogni modello...")
    calculate_and_print_metrics(dt, x_test, y_test, "Decision Tree")
    calculate_and_print_metrics(knn, x_test, y_test, "k-NN")
    calculate_and_print_metrics(rf, x_test, y_test, "Random Forest")
    
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
    print("-------------------------------------------------------------------")


    print("Salvataggio matrici di confusione...")
    save_confusion_matrix(dt, x_test, y_test, class_names, "Decision Tree", "confusion_matrix_dt.png")
    save_confusion_matrix(knn, x_test, y_test, class_names, "k-NN", "confusion_matrix_knn.png")
    save_confusion_matrix(rf, x_test, y_test, class_names, "Random Forest", "confusion_matrix_rf.png")
    print("\nMatrici salvate come immagini PNG.")
    
    print("-------------------------------------------------------------------")

    print("\nGenerazione grafici aggiuntivi...\n")
    visualize_decision_tree(dt, feature_names, class_names, "decision_tree_viz.png")
    visualize_knn_boundaries(x_train, y_train, k=5, class_names=class_names, filename="knn_decision_boundaries.png")
    
    plot_feature_importance(knn, feature_names, "k-NN", "feature_importance_knn.png", x_data=x_test, y_data=y_test)
    plot_feature_importance(dt, feature_names, "Decision Tree", "feature_importance_dt.png")
    plot_feature_importance(rf, feature_names, "Random Forest", "feature_importance_rf.png")
    
    print("-------------------------------------------------------------------")

    fine = time.time() - inizio
    print(f"\nTempo totale di esecuzione: {fine:.4f} secondi\n")
if __name__ == "__main__":
    main()
    