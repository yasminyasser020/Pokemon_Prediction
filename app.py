import pandas as pd
import joblib
import streamlit as st

# --- DASHBOARD SETUP ---
st.set_page_config(page_title="Pokémon Predictor", layout="centered")
st.title("🐾 Pokémon Predictor")

# --- LOAD MODELS (Cached for Speed) ---
@st.cache_resource
def load_assets():
    return {
        "m1": joblib.load('PowerPredictor.pkl'),
        "m2": joblib.load('LegendaryPredictor.pkl'),
        "m3": joblib.load('GenerationPredictor.pkl'),
        "m4": joblib.load('NamePredictor.pkl'),
        "e_p": joblib.load('power_encoder.pkl'),
        "e_n": joblib.load('name_encoder.pkl'),
        "e_l": joblib.load('legendary_encoder.pkl'),
        "e_g": joblib.load('generation_encoder.pkl')
    }

assets = load_assets()
data = pd.read_csv('Pokemon.csv')
name_to_id = dict(zip(data['Name'], data['#']))

# Exactly as your X2_with_power.columns.tolist()
training_columns = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary', 'Generation', 
                    'Power_Dark', 'Power_Dragon', 'Power_Electric', 'Power_Fairy', 'Power_Fighting', 
                    'Power_Fire', 'Power_Flying', 'Power_Ghost', 'Power_Grass', 'Power_Ground', 
                    'Power_Ice', 'Power_Normal', 'Power_Poison', 'Power_Psychic', 'Power_Rock', 
                    'Power_Steel', 'Power_Water']

# --- UI INPUTS ---
st.sidebar.header("Stats")
hp = st.sidebar.slider("HP", 1, 255, 70)
atk = st.sidebar.slider("Attack", 1, 190, 70)
dfe = st.sidebar.slider("Defense", 1, 230, 70)
sa = st.sidebar.slider("Sp. Atk", 1, 194, 70)
sd = st.sidebar.slider("Sp. Def", 1, 230, 70)
spd = st.sidebar.slider("Speed", 1, 180, 70)

if st.button("Predict"):
    # 1. Base Stats
    new_p = pd.DataFrame({'HP':[hp],'Attack':[atk],'Defense':[dfe],'Sp. Atk':[sa],'Sp. Def':[sd],'Speed':[spd]})
    
    # 2. Intermediate Predictions
    p_label = assets["e_p"].inverse_transform(assets["m1"].predict(new_p))[0]
    l_label = assets["e_l"].inverse_transform(assets["m2"].predict(new_p))[0]
    g_label = assets["e_g"].inverse_transform(assets["m3"].predict(new_p))[0]
    
    # 3. Final Name Prediction
    new_p['Legendary'], new_p['Generation'] = l_label, g_label
    new_p[f'Power_{p_label}'] = 1
    new_p = new_p.reindex(columns=training_columns, fill_value=0)
    
    res_name = assets["e_n"].inverse_transform(assets["m4"].predict(new_p))[0]
    
    # --- DISPLAY ---
    st.header(f"Result: {res_name}")
    st.write(f"Type: {p_label} | Gen: {g_label} | Legendary: {bool(l_label)}")
    
    p_id = name_to_id.get(res_name)
    if p_id:
        url = f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{p_id}.png"
        st.image(url, width=300)