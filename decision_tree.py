import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

column_names = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
    "ring-type", "spore-print-color", "population", "habitat"
]

class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Value represents the class label if the node is a leaf node
        
def decision_tree_train(X, y):
    # Recursively build the decision tree
    def build_tree(X, y):
        # num saples contiene il numero di righe (quanti funghi abbiamo) mentre num_features contiene il numero di colonne (quante caratteristiche ha ogni fungo)
        num_samples, num_features = X.shape
        
        # se non ci sono più campioni ritorna none
        if num_samples == 0:
            return None
        
        # se tutti i campioni appartengono alla stessa classe, crea un nodo foglia 
        if len(set(y)) == 1:
            return DecisionNode(value=y[0])
        
        best_feature = None
        best_threshold = None
        best_gini = float('inf')
        
        for feature_index in range(num_features):
            thresholds = set(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold
                left_gini = gini_impurity(y[left_indices])
                right_gini = gini_impurity(y[right_indices])
                total_gini = (len(y[left_indices]) * left_gini + len(y[right_indices]) * right_gini) / num_samples
                if total_gini < best_gini:
                    best_feature = feature_index
                    best_threshold = threshold
                    best_gini = total_gini
        
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        
        left_subtree = build_tree(X[left_indices], y[left_indices])
        right_subtree = build_tree(X[right_indices], y[right_indices])
        
        return DecisionNode(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)
    
    # Calculate Gini impurity
    def gini_impurity(y):
        classes = set(y)
        impurity = 1
        for c in classes:
            p = len(y[y == c]) / len(y)
            impurity -= p ** 2
        return impurity
    
    return build_tree(X, y)

def decision_tree_predict(tree, sample):
    if tree.value is not None:
        return tree.value
    
    if sample[tree.feature] <= tree.threshold:
        return decision_tree_predict(tree.left, sample)
    else:
        return decision_tree_predict(tree.right, sample)

def main():
    input_data = pd.read_csv('mushroom/mushroom.csv', header=None, names=column_names)
    input_data['class'] = LabelEncoder().fit_transform(input_data['class'])
    input_data.drop('veil-type', axis=1, inplace=True)  # Rimuoviamo la colonna 'veil-type' poiché ha un solo valore
    input_data['stalk-root'] = input_data['stalk-root'].replace('?', 'missing')  # Gestiamo i valori mancanti
    x = pd.get_dummies(input_data.drop('class', axis=1))
    y = input_data['class']
    x_scaled = StandardScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
        

    # Train the decision tree
    tree = decision_tree_train(x_train, y_train.values)
    # Predict on test data
    predictions = [decision_tree_predict(tree, sample) for sample in x_test]

    # Evaluate the predictions
    accuracy = sum(1 for pred, true in zip(predictions, y_test) if pred == true) / len(y_test)
    print("Accuracy:", accuracy)
    
if __name__ == "__main__":
    main()