import pandas as pd
import numpy as np

# Définir le nombre de lignes
num_rows = 150

# Générer des données aléatoires
np.random.seed(42)  # Pour la reproductibilité
temperature_moteur = np.random.randint(60, 130, size=num_rows)
pression_pneus = np.random.randint(20, 50, size=num_rows)
niveau_huile = np.random.randint(1, 10, size=num_rows)
vibrations = np.random.uniform(0.1, 0.6, size=num_rows)
panne = np.random.randint(0, 2, size=num_rows)

# Créer un DataFrame
data = {
    'temperature_moteur': temperature_moteur,
    'pression_pneus': pression_pneus,
    'niveau_huile': niveau_huile,
    'vibrations': vibrations,
    'panne': panne
}
df = pd.DataFrame(data)

# Sauvegarder en fichier CSV
df.to_csv('data.csv', index=False)
