import streamlit as st
import pandas as pd
import numpy as np
import urllib.request
import ssl
import certifi
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

st.set_page_config(
    page_title="Deep Learning Solubility Predictor",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 Deep Learning Solubility Predictor (PyTorch)")
st.markdown("Predict aqueous solubility ($\log S$) using a custom **PyTorch Feedforward Neural Network** trained on RDKit molecular descriptors.")

@st.cache_resource
def train_pytorch_model():
    # 1. Load dataset
    url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req, context=ssl_context) as response:
        df = pd.read_csv(response)
        
    target_col = [col for col in df.columns if 'solubility' in col.lower()][0]
    
    # 2. Extract RDKit descriptors
    def extract_features(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
            'MolLogP': Descriptors.MolLogP(mol),
            'MolWt': Descriptors.MolWt(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'FractionCSP3': Descriptors.FractionCSP3(mol),
            'HeavyAtomCount': Descriptors.HeavyAtomCount(mol)
        }

    features_list = []
    valid_indices = []
    for idx, smiles in enumerate(df['smiles']):
        feats = extract_features(smiles)
        if feats is not None:
            features_list.append(feats)
            valid_indices.append(idx)

    X = pd.DataFrame(features_list)
    y = df.loc[valid_indices, target_col].values
    
    # 3. Scale features for neural network stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Convert to PyTorch Tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    # 5. Define PyTorch Neural Network Architecture
    class SolubilityNet(nn.Module):
        def __init__(self, input_dim):
            super(SolubilityNet, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
            
        def forward(self, x):
            return self.network(x)

    model = SolubilityNet(input_dim=X_scaled.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 6. Training Loop
    model.train()
    epochs = 200
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()
        
    model.eval()
    return model, scaler

with st.spinner("Training PyTorch neural network on chemical descriptors..."):
    model, scaler = train_pytorch_model()

st.success("PyTorch model trained and ready!")

st.subheader("Molecule Input")
example_smiles = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Ethanol": "CCO",
    "Cholesterol": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C"
}

selected_example = st.selectbox("Choose a sample molecule or type your own SMILES below:", ["Custom"] + list(example_smiles.keys()))

if selected_example != "Custom":
    default_smiles = example_smiles[selected_example]
else:
    default_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

smiles_input = st.text_input("Enter SMILES string:", value=default_smiles)

if st.button("Predict with Deep Learning", type="primary"):
    mol = Chem.MolFromSmiles(smiles_input)
    if mol is None:
        st.error("Invalid SMILES string. Please check your syntax and try again.")
    else:
        input_features = {
            'MolLogP': Descriptors.MolLogP(mol),
            'MolWt': Descriptors.MolWt(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'FractionCSP3': Descriptors.FractionCSP3(mol),
            'HeavyAtomCount': Descriptors.HeavyAtomCount(mol)
        }
        
        # Scale input features using the same scaler
        input_df = pd.DataFrame([input_features])
        scaled_input = scaler.transform(input_df)
        tensor_input = torch.tensor(scaled_input, dtype=torch.float32)
        
        # Neural network inference
        model.eval()
        with torch.no_grad():
            prediction = model(tensor_input).item()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.metric(label="PyTorch Predicted Log S (mol/L)", value=f"{prediction:.2f}")
            if prediction > -2:
                st.info("💧 **High Solubility**: Dissolves very readily in water.")
            elif prediction > -4:
                st.warning("⚖️ **Moderate Solubility**: Typical aqueous solubility.")
            else:
                st.error("🏜️ **Low Solubility**: Poorly soluble / hydrophobic.")
                
        with col2:
            st.write("**Molecular Structure:**")
            img = Draw.MolToImage(mol, size=(300, 300))
            st.image(img, width='stretch')
            
        with st.expander("View Calculated RDKit Descriptors"):
            st.json(input_features)
