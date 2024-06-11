import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("Diagnostic Prédictif des Automobiles avec l'IA")
    st.subheader("Analyse des pannes automobiles")

    # Fonction de chargement des données
    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv('data.csv')
        return data

    # Charger les données
    data = load_data()

    # Afficher les données brutes
    if st.sidebar.checkbox("Afficher les données brutes", False):
        st.subheader("Données brutes")
        st.write(data)

    # Séparation des caractéristiques et de la cible
    X = data[['temperature_moteur', 'pression_pneus', 'niveau_huile', 'vibrations']]
    y = data['panne']

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Créer et entraîner le modèle de forêt aléatoire
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Évaluer le modèle
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    st.subheader("Évaluation du Modèle")
    st.write(f"Précision du modèle: {accuracy*100:.2f}%")
    st.text("Rapport de classification")
    st.text(report)

    # Afficher la matrice de confusion
    st.subheader("Matrice de confusion")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Pas de panne', 'Panne']).plot(ax=ax)
    st.pyplot(fig)

    # Importance des caractéristiques
    feature_importances = model.feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({'Caractéristique': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    st.subheader("Importance des Caractéristiques")
    st.write(importance_df)

    # Visualiser l'importance des caractéristiques
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Caractéristique', data=importance_df)
    plt.title('Importance des Caractéristiques')
    st.pyplot()

    # Prédiction des pannes
    st.sidebar.subheader("Paramètres du véhicule")
    st.sidebar.write("Veuillez entrer les paramètres du véhicule :")

    temperature_moteur = st.sidebar.number_input("Température du moteur", min_value=0, max_value=150, value=90)
    pression_pneus = st.sidebar.number_input("Pression des pneus", min_value=0, max_value=50, value=32)
    niveau_huile = st.sidebar.number_input("Niveau d'huile", min_value=0, max_value=10, value=5)
    vibrations = st.sidebar.number_input("Vibrations", min_value=0.0, max_value=1.0, value=0.2)

    predict_button = st.sidebar.button("Prédire")

    if predict_button:
        input_data = np.array([[temperature_moteur, pression_pneus, niveau_huile, vibrations]])
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        st.subheader("Résultat de la prédiction")
        st.markdown(f"<h1>{'Panne' if prediction[0] == 1 else 'Pas de panne'}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h2>Probabilité : {'{:.2f}%'.format(prediction_proba[0][1]*100) if prediction[0] == 1 else '{:.2f}%'.format(prediction_proba[0][0]*100)}</h2>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()