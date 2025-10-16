import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="🍄 MushroomSafe AI",
    page_icon="🍄",
    layout="wide"
)

# Titre de l'application
st.title("🍄 MushroomSafe AI")
st.markdown("### Système Intelligent de Classification des Champignons")
st.markdown("---")


# Chargement des modèles
@st.cache_resource
def load_models():
    try:
        models = {}
        model_files = {
            'Random Forest': 'random_forest_model.pkl',
            'Logistic Regression': 'logistic_regression_model.pkl',
            'SVM': 'svc_model.pkl',
            'Decision Tree': 'decision_tree_model.pkl',
            'K-Neighbors': 'k-neighbors_model.pkl',
            'Neural Network': 'neural_network_model.pkl'
        }

        for name, filename in model_files.items():
            models[name] = joblib.load(filename)

        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')

        return models, scaler, label_encoders
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {str(e)}")
        return None, None, None


# Dictionnaire de mapping pour l'interface utilisateur
feature_descriptions = {
    'cap-shape': {
        'options': ['bell', 'conical', 'convex', 'flat', 'knobbed', 'sunken'],
        'codes': ['b', 'c', 'x', 'f', 'k', 's'],
        'description': 'Forme du chapeau'
    },
    'cap-surface': {
        'options': ['fibrous', 'grooves', 'scaly', 'smooth'],
        'codes': ['f', 'g', 'y', 's'],
        'description': 'Surface du chapeau'
    },
    'cap-color': {
        'options': ['brown', 'buff', 'cinnamon', 'gray', 'green', 'pink', 'purple', 'red', 'white', 'yellow'],
        'codes': ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'],
        'description': 'Couleur du chapeau'
    },
    'odor': {
        'options': ['almond', 'anise', 'creosote', 'fishy', 'foul', 'musty', 'none', 'pungent', 'spicy'],
        'codes': ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'],
        'description': 'Odeur du champignon'
    },
    'bruises': {
        'options': ['yes', 'no'],
        'codes': ['t', 'f'],
        'description': 'Présence de meurtrissures'
    },
    'gill-size': {
        'options': ['broad', 'narrow'],
        'codes': ['b', 'n'],
        'description': 'Taille des lamelles'
    },
    'gill-color': {
        'options': ['black', 'brown', 'buff', 'chocolate', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white',
                    'yellow'],
        'codes': ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'],
        'description': 'Couleur des lamelles'
    },
    'stalk-shape': {
        'options': ['enlarging', 'tapering'],
        'codes': ['e', 't'],
        'description': 'Forme du pied'
    },
    'stalk-surface-above-ring': {
        'options': ['fibrous', 'scaly', 'silky', 'smooth'],
        'codes': ['f', 'y', 'k', 's'],
        'description': 'Surface du pied au-dessus de l\'anneau'
    },
    'stalk-surface-below-ring': {
        'options': ['fibrous', 'scaly', 'silky', 'smooth'],
        'codes': ['f', 'y', 'k', 's'],
        'description': 'Surface du pied en-dessous de l\'anneau'
    },
    'veil-color': {
        'options': ['brown', 'orange', 'white', 'yellow'],
        'codes': ['n', 'o', 'w', 'y'],
        'description': 'Couleur du voile'
    },
    'ring-type': {
        'options': ['evanescent', 'flaring', 'large', 'none', 'pendant'],
        'codes': ['e', 'f', 'l', 'n', 'p'],
        'description': 'Type d\'anneau'
    },
    'spore-print-color': {
        'options': ['black', 'brown', 'buff', 'chocolate', 'green', 'orange', 'purple', 'white', 'yellow'],
        'codes': ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'],
        'description': 'Couleur des spores'
    },
    'population': {
        'options': ['abundant', 'clustered', 'numerous', 'scattered', 'several', 'solitary'],
        'codes': ['a', 'c', 'n', 's', 'v', 'y'],
        'description': 'Mode de croissance'
    },
    'habitat': {
        'options': ['grasses', 'leaves', 'meadows', 'paths', 'urban', 'waste', 'woods'],
        'codes': ['g', 'l', 'm', 'p', 'u', 'w', 'd'],
        'description': 'Environnement de croissance'
    }
}

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisir une page:", ["🏠 Accueil", "🔍 Classification", "📊 Performance"])

if page == "🏠 Accueil":
    st.header("Bienvenue dans MushroomSafe AI")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 🎯 Objectif
        Ce système utilise l'intelligence artificielle pour classifier les champignons comme **comestibles** ou **poisonneux**.

        ### 📊 Données
        - **8,124** échantillons de champignons
        - **22** caractéristiques morphologiques
        - Dataset de l'UCI Machine Learning Repository

        ### 🤖 Algorithmes Utilisés
        - Random Forest
        - Logistic Regression  
        - SVM (Support Vector Machine)
        - Decision Tree
        - K-Neighbors
        - Neural Network

        ### ⚠️ Attention
        Ce système est une aide à la décision. **Ne consommez jamais un champignon basé uniquement sur cette prédiction!**
        Consultez toujours un expert en mycologie.
        """)

    with col2:
        st.image("https://cdn.pixabay.com/photo/2017/10/15/11/41/mushroom-2853837_960_720.png",
                 width=300, caption="🍄 Classification des Champignons")

        st.info("""
        **Rappel de sécurité:**
        La consommation de champignons sauvages peut être extrêmement dangereuse sans expertise appropriée.
        """)

elif page == "🔍 Classification":
    st.header("🔍 Classification de Champignons")

    models, scaler, label_encoders = load_models()

    if models is not None and label_encoders is not None:
        # Sélection du modèle
        selected_model = st.selectbox(
            "Choisir le modèle de classification:",
            list(models.keys())
        )

        st.markdown("### Caractéristiques du Champignon")

        # Interface utilisateur pour les caractéristiques
        col1, col2, col3 = st.columns(3)

        features = {}

        with col1:
            features['cap-shape'] = st.selectbox(
                "Forme du chapeau",
                options=feature_descriptions['cap-shape']['options'],
                help=feature_descriptions['cap-shape']['description']
            )
            features['cap-surface'] = st.selectbox(
                "Surface du chapeau",
                options=feature_descriptions['cap-surface']['options'],
                help=feature_descriptions['cap-surface']['description']
            )
            features['cap-color'] = st.selectbox(
                "Couleur du chapeau",
                options=feature_descriptions['cap-color']['options'],
                help=feature_descriptions['cap-color']['description']
            )
            features['odor'] = st.selectbox(
                "Odeur",
                options=feature_descriptions['odor']['options'],
                help=feature_descriptions['odor']['description']
            )

        with col2:
            features['bruises'] = st.selectbox(
                "Meurtrissures",
                options=feature_descriptions['bruises']['options'],
                help=feature_descriptions['bruises']['description']
            )
            features['gill-size'] = st.selectbox(
                "Taille des lamelles",
                options=feature_descriptions['gill-size']['options'],
                help=feature_descriptions['gill-size']['description']
            )
            features['gill-color'] = st.selectbox(
                "Couleur des lamelles",
                options=feature_descriptions['gill-color']['options'],
                help=feature_descriptions['gill-color']['description']
            )
            features['stalk-shape'] = st.selectbox(
                "Forme du pied",
                options=feature_descriptions['stalk-shape']['options'],
                help=feature_descriptions['stalk-shape']['description']
            )

        with col3:
            features['stalk-surface-above-ring'] = st.selectbox(
                "Surface du pied au-dessus de l'anneau",
                options=feature_descriptions['stalk-surface-above-ring']['options'],
                help=feature_descriptions['stalk-surface-above-ring']['description']
            )
            features['stalk-surface-below-ring'] = st.selectbox(
                "Surface du pied en-dessous de l'anneau",
                options=feature_descriptions['stalk-surface-below-ring']['options'],
                help=feature_descriptions['stalk-surface-below-ring']['description']
            )
            features['veil-color'] = st.selectbox(
                "Couleur du voile",
                options=feature_descriptions['veil-color']['options'],
                help=feature_descriptions['veil-color']['description']
            )
            features['ring-type'] = st.selectbox(
                "Type d'anneau",
                options=feature_descriptions['ring-type']['options'],
                help=feature_descriptions['ring-type']['description']
            )

        # Caractéristiques supplémentaires
        col4, col5 = st.columns(2)

        with col4:
            features['spore-print-color'] = st.selectbox(
                "Couleur des spores",
                options=feature_descriptions['spore-print-color']['options'],
                help=feature_descriptions['spore-print-color']['description']
            )

        with col5:
            features['population'] = st.selectbox(
                "Population",
                options=feature_descriptions['population']['options'],
                help=feature_descriptions['population']['description']
            )
            features['habitat'] = st.selectbox(
                "Habitat",
                options=feature_descriptions['habitat']['options'],
                help=feature_descriptions['habitat']['description']
            )

        # Bouton de prédiction
        if st.button("🔍 Analyser le Champignon", type="primary"):
            try:
                # Conversion des valeurs d'affichage en codes réels
                input_data = {}
                for feature, display_value in features.items():
                    # Trouver le code correspondant à la valeur d'affichage
                    options = feature_descriptions[feature]['options']
                    codes = feature_descriptions[feature]['codes']
                    code_value = codes[options.index(display_value)]

                    # Encoder avec le LabelEncoder
                    input_data[feature] = label_encoders[feature].transform([code_value])[0]

                # Création du vecteur de features complet
                full_feature_vector = np.zeros(len(label_encoders))
                for i, (feature_name, le) in enumerate(label_encoders.items()):
                    if feature_name in input_data:
                        full_feature_vector[i] = input_data[feature_name]

                # Sélection du modèle
                model = models[selected_model]

                # Appliquer la normalisation si nécessaire
                if selected_model in ['Logistic Regression', 'SVM', 'Neural Network', 'K-Neighbors']:
                    input_scaled = scaler.transform([full_feature_vector])
                    prediction = model.predict(input_scaled)[0]
                    probability = model.predict_proba(input_scaled)[0]
                else:
                    prediction = model.predict([full_feature_vector])[0]
                    probability = model.predict_proba([full_feature_vector])[0]

                # Affichage des résultats
                st.markdown("---")
                st.subheader("📊 Résultats de l'Analyse")

                col_result1, col_result2 = st.columns(2)

                with col_result1:
                    if prediction == 0:
                        st.success("## 🍽️ CHAMPIGNON COMESTIBLE")
                        st.balloons()
                    else:
                        st.error("## ☠️ CHAMPIGNON POISONNEUX")
                        st.warning("⚠️ DANGER - Ne pas consommer!")

                with col_result2:
                    st.metric(
                        label="Confiance de la prédiction",
                        value=f"{max(probability) * 100:.2f}%"
                    )
                    st.progress(float(max(probability)))

                # Détails de la prédiction
                with st.expander("📈 Détails de la Prédiction"):
                    col_prob1, col_prob2 = st.columns(2)
                    with col_prob1:
                        st.metric("Probabilité Comestible", f"{probability[0] * 100:.2f}%")
                    with col_prob2:
                        st.metric("Probabilité Poisonneux", f"{probability[1] * 100:.2f}%")

                    st.info(f"Modèle utilisé: **{selected_model}**")

            except Exception as e:
                st.error(f"Erreur lors de la prédiction: {str(e)}")

    else:
        st.error("Veuillez d'abord entraîner les modèles en exécutant 'mushroom_classifier_final.py'")

elif page == "📊 Performance":
    st.header("📊 Performance des Modèles")

    # Affichage des métriques de performance
    st.markdown("### Comparaison des Algorithmes")

    performance_data = {
        'Modèle': ['Random Forest', 'Decision Tree', 'K-Neighbors', 'Neural Network', 'SVM', 'Logistic Regression'],
        'Accuracy': [1.00, 1.00, 1.00, 1.00, 1.00, 0.95],
        'Precision': [1.00, 1.00, 1.00, 1.00, 1.00, 0.94],
        'Recall': [1.00, 1.00, 1.00, 1.00, 1.00, 0.96],
        'F1-Score': [1.00, 1.00, 1.00, 1.00, 1.00, 0.95]
    }

    df_performance = pd.DataFrame(performance_data)
    st.dataframe(df_performance.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

    # Graphiques
    st.markdown("### Visualisation des Performances")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.bar_chart(df_performance.set_index('Modèle')['Accuracy'])
        st.caption("Accuracy par Modèle")

    with col_chart2:
        st.bar_chart(df_performance.set_index('Modèle')[['Precision', 'Recall', 'F1-Score']])
        st.caption("Métriques Détaillées par Modèle")

    # Informations sur le dataset
    st.markdown("### 📋 Informations sur le Dataset")

    col_info1, col_info2, col_info3 = st.columns(3)

    with col_info1:
        st.metric("Échantillons Totaux", "8,124")

    with col_info2:
        st.metric("Caractéristiques", "22")

    with col_info3:
        st.metric("Classes", "2 (Comestible/Poisonneux)")

# Footer
st.markdown("---")
st.markdown(
    "**⚠️ Disclaimer:** Cette application est à but éducatif. "
    "Consultez toujours un expert avant de consommer des champignons sauvages."
)