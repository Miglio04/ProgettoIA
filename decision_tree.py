import pandas as pd
import numpy as np
import time
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import recall_score, precision_score
from scipy.io.arff import loadarff

column_names = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
    "ring-type", "spore-print-color", "population", "habitat"
]

# Classe per i nodi dell'albero decisionale
class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        # feature: indice della feature su cui effettuare lo split
        self.feature = feature
        # threshold: valore della feature su cui effettuare lo split
        self.threshold = threshold
        # left: figlio sinistro del nodo
        self.left = left
        # right: figlio destro del nodo
        self.right = right
        # value: etichetta di classe se è un nodo foglia, null altrimenti
        self.value = value

# costruisce l'albero decisionale ricorsivamente
def build_tree(X, y):
    # num_samples contiene il numero di righe (quanti funghi abbiamo) mentre num_features contiene il numero di colonne (quante caratteristiche ha ogni fungo)
    num_samples, num_features = X.shape
    
    # se non ci sono più campioni ritorna none
    if num_samples == 0:
        return None
    
    # se tutti i campioni appartengono alla stessa classe, crea un nodo foglia con quella classe
    if len(set(y)) == 1:
        return DecisionNode(value=y[0])
    
    best_feature = None
    best_threshold = None
    best_gini = float('inf')
    
    # per ogni feature, prova a trovare la soglia che minimizza l'impurità di Gini
    for feature_index in range(num_features):
        # ottieni i valori unici della feature per trovare le possibili soglie di split (ogni valore unico è una possibile soglia, i duplicati vengono eliminati grazie a set())
        thresholds = set(X[:, feature_index])
        # per ogni soglia, calcola l'impurità di Gini per lo split e aggiorna il miglior split se necessario
        for threshold in thresholds:
            # crea due sottoinsiemi di dati in base alla soglia
            left_indices = X[:, feature_index] <= threshold
            right_indices = X[:, feature_index] > threshold

            # se uno dei due sottoinsiemi è vuoto, salta questo split perché non è valido
            if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                continue
            
            # calcola l'impurità di Gini per i due sottoinsiemi e la media pesata dell'impurità totale
            left_gini = gini_impurity(y[left_indices])
            right_gini = gini_impurity(y[right_indices])
            total_gini = (len(y[left_indices]) * left_gini + len(y[right_indices]) * right_gini) / num_samples
            
            # se questo split è migliore del miglior split trovato finora, viene aggiornato il miglior split
            if total_gini < best_gini:
                best_feature = feature_index
                best_threshold = threshold
                best_gini = total_gini
    
    # se non è stato trovato nessuno split valido, crea un nodo foglia con la classe più comune
    if best_feature is None:
            vals, counts = np.unique(y, return_counts=True)
            return DecisionNode(value=vals[np.argmax(counts)])

    # crea i sottoalberi ricorsivamente per i due sottoinsiemi di dati
    left_indices = X[:, best_feature] <= best_threshold
    right_indices = X[:, best_feature] > best_threshold
    
    left_subtree = build_tree(X[left_indices], y[left_indices])
    right_subtree = build_tree(X[right_indices], y[right_indices])
    
    # ritorna un nodo decisionale con la feature, la soglia e i sottoalberi
    return DecisionNode(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

# Calcola l'impurità di Gini
def gini_impurity(y):
    # ottieni le classi uniche presenti nei dati
    classes = set(y)
    impurity = 1
    # per ogni classe, calcola la probabilità di quella classe e sottrai il quadrato di quella probabilità dall'impurità totale per ottenere un grado di disordine nei dati
    for c in classes:
        p = len(y[y == c]) / len(y)
        impurity -= p ** 2
    return impurity

# Funzione per fare previsioni con l'albero decisionale
def decision_tree_predict(tree, sample):
    # se il nodo è una foglia, ritorna il valore della classe
    if tree.value is not None:
        return tree.value
    
    # confronta il valore della feature con la soglia e prosegui nel sottoalbero appropriato
    if sample[tree.feature] <= tree.threshold:
        return decision_tree_predict(tree.left, sample)
    else:
        return decision_tree_predict(tree.right, sample)

# Funzione per eseguire la cross-validation e calcolare l'accuratezza media e la deviazione standard
def cross_validation_score(X, y, cv=10):
    # Utilizza StratifiedKFold per mantenere la proporzione delle classi in ogni fold
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    accuracies = []
    
    # Assicurati che y sia un array numpy
    y = y.values if hasattr(y, 'values') else y
    
    print(f"\nCross-Validation con {cv} fold...")
    
    # Per ogni fold, addestra l'albero decisionale sui dati di training e valuta l'accuratezza sui dati di test
    for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        tree = build_tree(X_train, y_train)
        predictions = [decision_tree_predict(tree, sample) for sample in X_test]
        acc = sum(1 for pred, true in zip(predictions, y_test) if pred == true) / len(y_test)
        accuracies.append(acc)
    # Calcola e ritorna l'accuratezza media e la deviazione standard delle accuratezze ottenute nei diversi fold
    return np.mean(accuracies), np.std(accuracies)

# Funzione per caricare e preprocessare i dati del dataset Mushroom
def load_and_preprocess_data_mushrooms():
    input_data = pd.read_csv('datasets/mushroom/mushroom.csv', header=None, names=column_names)
    # Trasformiamo la colonna 'class' in valori numerici (0 per commestibile, 1 per velenoso)
    input_data['class'] = LabelEncoder().fit_transform(input_data['class'])
    
    input_data.drop('veil-type', axis=1, inplace=True)  # Rimuoviamo la colonna 'veil-type' poiché ha un solo valore
    input_data.drop('stalk-root', axis=1, inplace=True)  # Rimuoviamo la colonna 'stalk-root' poiché ha valori mancanti
    
    # Convertiamo le variabili categoriche in variabili dummy (one-hot encoding) e standardizziamo i dati
    x = pd.get_dummies(input_data.drop('class', axis=1))
    y = input_data['class']
    x_scaled = StandardScaler().fit_transform(x)
    
    return x_scaled, y

# Funzione per caricare e preprocessare i dati del dataset Rice
def load_and_preprocess_data_rice():
    data, meta = loadarff('datasets/rice+cammeo+and+osmancik/Rice_Cammeo_Osmancik.arff')
    input_data = pd.DataFrame(data)# Quando si caricano i dati da un file ARFF, le colonne di tipo stringa vengono spesso interpretate come array di byte (dtype 'object' in pandas).
    # Quindi, se la colonna 'Class' è di tipo object, cerchiamo di decodificarla in stringhe leggibili (utf-8) per poterla poi codificare con LabelEncoder.
    if pd.api.types.is_object_dtype(input_data['Class']):
        try:
             input_data['Class'] = input_data['Class'].str.decode('utf-8')
        except AttributeError:
            pass

    # la LabelEncoder trasforma le etichette di classe da stringhe a numeri interi (0 e 1)
    le = LabelEncoder()
    # qui viene applicato il LabelEncoder alla colonna 'class' del dataset
    input_data['Class'] = le.fit_transform(input_data['Class'])
     
    # La variabile x contiene tutte le feature del dataset, escludendo la colonna 'Class' che è la variabile target.
    x = input_data.drop('Class', axis=1)
    
    # La variabile target y è la colonna 'class' del dataset, che contiene le etichette di classe codificate come numeri interi
    y = input_data['Class']
    
    # Scaliamo le feature usando lo StandardScaler che normalizza le feature per avere media 0 e deviazione standard 1.
    x_scaled = StandardScaler().fit_transform(x)
    
    return x_scaled, y

# Funzione per calcolare le metriche di valutazione (accuratezza, precisione, recall)
def calculate_metrics(predictions, y_true):
    accuracy = sum(1 for pred, true in zip(predictions, y_true) if pred == true) / len(y_true)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    return accuracy, precision, recall

# Funzione per valutare il modello sui dati di test e stampare le metriche di valutazione
def evaluate_model(tree, X_test, y_test):
    predictions = [decision_tree_predict(tree, sample) for sample in X_test]
    accuracy, precision, recall = calculate_metrics(predictions, y_test)
    
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")

def main():
    print("Inserire 1 se si vogliono analizzare i modelli addestrati sul dataset Mushroom, 2 per il dataset Rice:")
    choice = input("Scelta: ")
    
    x_train, x_test, y_train, y_test = None, None, None, None
    inizio = time.time()
    
    print("\nCaricamento e preprocessing dei dati...\n")
    if choice == '1':
        x_scaled, y = load_and_preprocess_data_mushrooms()
    elif choice == '2':
        x_scaled, y = load_and_preprocess_data_rice()
    else:
        print("Scelta non valida. Uscita.")
        return
    
    
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
    
    print("-------------------------------------------------------------------")
    
    print("\nAddestramento dell'albero di decisione...\n")
    start = time.time()
    tree = build_tree(x_train, y_train.values)
    training_time = time.time() - start
    print(f"\nTempo di addestramento: {training_time:.4f} secondi\n")
    
    print("-------------------------------------------------------------------")
    
    print("\nValutazione del modello sui dati di test...\n")
    evaluate_model(tree, x_test, y_test)
    print("\nModello Decision Tree addestrato e valutato.\n")
    
    print("-------------------------------------------------------------------")
    
    mean_acc, std_acc = cross_validation_score(x_scaled, y, cv=10)
    print(f"\nAccuratezza con Cross-Validation: {mean_acc * 100:.2f}% (+/- {std_acc * 100:.2f}%)\n")
    fine = time.time() - inizio
    
    print("-------------------------------------------------------------------")
    
    print(f"\nTempo totale di esecuzione: {fine:.4f} secondi\n")
    
    print("-------------------------------------------------------------------")

if __name__ == "__main__":
    main()