import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class MushroomClassifier:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'K-Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Neural Network': MLPClassifier(random_state=42, max_iter=1000)
        }
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.results = {}

    def load_data(self):
        """Charge les données depuis le dataset original"""
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
        column_names = [
            'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
            'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
            'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
            'stalk-surface-below-ring', 'stalk-color-above-ring',
            'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
            'ring-type', 'spore-print-color', 'population', 'habitat'
        ]
        self.data = pd.read_csv(url, header=None, names=column_names)
        return self.data

    def preprocess_data(self):
        """Prétraite les données avec encodage correct"""
        self.X = self.data.drop('class', axis=1)
        self.y = self.data['class']

        # Encodage de la variable cible
        self.y = self.y.map({'e': 0, 'p': 1})  # 0=comestible, 1=poisonneux

        # Encodage des features
        for column in self.X.columns:
            le = LabelEncoder()
            self.X[column] = le.fit_transform(self.X[column])
            self.label_encoders[column] = le

            print(f"{column}: {dict(zip(le.classes_, range(len(le.classes_))))}")

        # Séparation des données
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        # Normalisation
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_models(self):
        """Entraîne tous les modèles"""
        print("🔬 Entraînement des modèles...")

        for name, model in self.models.items():
            print(f"Entraînement: {name}")

            if name in ['Logistic Regression', 'SVM', 'Neural Network', 'K-Neighbors']:
                X_train_used = self.X_train_scaled
                X_test_used = self.X_test_scaled
            else:
                X_train_used = self.X_train
                X_test_used = self.X_test

            model.fit(X_train_used, self.y_train)
            y_pred = model.predict(X_test_used)
            accuracy = accuracy_score(self.y_test, y_pred)

            self.results[name] = {
                'model': model,
                'accuracy': accuracy
            }
            print(f"✅ {name} - Accuracy: {accuracy:.4f}")

    def compare_models(self):
        """Compare les performances"""
        comparison = []
        for name, result in self.results.items():
            comparison.append({
                'Modèle': name,
                'Accuracy': result['accuracy']
            })
        return pd.DataFrame(comparison).sort_values('Accuracy', ascending=False)

    def save_models(self):
        """Sauvegarde les modèles"""
        for name, result in self.results.items():
            filename = f"{name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(result['model'], filename)

        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.label_encoders, 'label_encoders.pkl')
        print("✅ Modèles sauvegardés!")

def main():
    classifier = MushroomClassifier()

    print("🍄 Chargement des données...")
    data = classifier.load_data()
    print(f"📊 Dataset: {data.shape[0]} échantillons, {data.shape[1]} caractéristiques")

    print("🔄 Prétraitement...")
    classifier.preprocess_data()

    print("🤖 Entraînement...")
    classifier.train_models()

    print("\n🏆 COMPARAISON:")
    comparison_df = classifier.compare_models()
    print(comparison_df.to_string(index=False))

    classifier.save_models()
    return classifier

if __name__ == "__main__":
    classifier = main()