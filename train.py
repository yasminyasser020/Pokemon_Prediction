import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# 1. Load and Augment Data
data = pd.read_csv('Pokemon.csv')
data['Power'] = data['Type 2'].fillna(data['Type 1'])
pre_processed_data = data.drop(columns=['Type 1', 'Type 2'])

stat_columns = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
num_samples_per_pokemon = 100 
augmented_data = []

for index, row in pre_processed_data.iterrows():
    original_row = row.to_dict()
    original_row['Is_Augmented'] = 0 
    augmented_data.append(original_row)
    for _ in range(num_samples_per_pokemon - 1): 
        new_row = row.to_dict()
        for stat in stat_columns:
            base_val = new_row[stat]
            noise_val = np.random.normal(loc=base_val, scale=base_val * 0.05)
            new_row[stat] = max(1, int(round(noise_val)))
        new_row['Total'] = sum(new_row[stat] for stat in stat_columns)
        new_row['Is_Augmented'] = 1
        augmented_data.append(new_row)

augmented_df = pd.DataFrame(augmented_data)
augmented_df['Legendary'] = augmented_df['Legendary'].astype(int)

# 2. Preprocessing
X1 = augmented_df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]
X2 = augmented_df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed' ,'Legendary','Generation', 'Power']]
X2 = pd.get_dummies(X2, columns=['Power'], drop_first=True)

power_encoder = LabelEncoder()
name_encoder = LabelEncoder()
legendary_encoder = LabelEncoder()
generation_encoder = LabelEncoder()

y_power = power_encoder.fit_transform(augmented_df['Power'])
y_name = name_encoder.fit_transform(augmented_df['Name'])
y_legendary = legendary_encoder.fit_transform(augmented_df['Legendary'])
y_generation = generation_encoder.fit_transform(augmented_df['Generation'])

# 3. Train Models
m1 = RandomForestClassifier().fit(X1, y_power)
m2 = RandomForestClassifier().fit(X1, y_legendary)
m3 = RandomForestClassifier().fit(X1, y_generation)
m4 = RandomForestClassifier().fit(X2, y_name)

# 4. Save Everything
joblib.dump(m1, 'PowerPredictor.pkl', compress=3)
joblib.dump(m4, 'NamePredictor.pkl', compress=3)
joblib.dump(m2, 'LegendaryPredictor.pkl', compress=3)
joblib.dump(m3, 'GenerationPredictor.pkl', compress=3)
joblib.dump(power_encoder, 'power_encoder.pkl', compress=3)
joblib.dump(name_encoder, 'name_encoder.pkl', compress=3)
joblib.dump(legendary_encoder, 'legendary_encoder.pkl', compress=3)
joblib.dump(generation_encoder, 'generation_encoder.pkl', compress=3)
print("All models and encoders saved successfully!")