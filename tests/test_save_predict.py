"""Testa salvar modelo + preprocessamento, recarregar e prever em dados novos
(incluindo uma categoria nunca vista, para testar o reindex)."""
import numpy as np
import pandas as pd
import pickle
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

np.random.seed(1)
n = 400
df = pd.DataFrame({
    "idade": np.random.randint(18, 70, n),
    "cidade": np.random.choice(["SP", "RJ", "MG"], n),
})
df["comprou"] = ((df["idade"] > 40) & (df["cidade"] == "SP")).astype(int)

colunas_feature = ["idade", "cidade"]
X = pd.get_dummies(df[colunas_feature], drop_first=False)
colunas_dummies = list(X.columns)
scaler = StandardScaler()
Xn = scaler.fit_transform(X.astype(float))

le = LabelEncoder()
y = le.fit_transform(df["comprou"])

X_train, X_test, y_train, y_test = train_test_split(Xn, y, test_size=0.2, random_state=42)
modelo = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
modelo.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0)

# --- salvar ---
modelo.save("/tmp/modelo_teste.keras")
pacote = {
    "scaler": scaler, "colunas_dummies": colunas_dummies,
    "colunas_feature": colunas_feature, "coluna_rotulo": "comprou",
    "tarefa": "classificacao", "label_encoder": le, "num_classes": 2,
}
with open("/tmp/modelo_teste_preprocessamento.pkl", "wb") as f:
    pickle.dump(pacote, f)

# --- carregar em "outra sessao" ---
modelo2 = keras.models.load_model("/tmp/modelo_teste.keras")
with open("/tmp/modelo_teste_preprocessamento.pkl", "rb") as f:
    pacote2 = pickle.load(f)

# --- novos dados, incluindo uma categoria nunca vista ("BA") ---
df_novo = pd.DataFrame({
    "idade": [45, 25, 60],
    "cidade": ["SP", "BA", "RJ"],  # "BA" nunca apareceu no treino
})

X_novo = df_novo[pacote2["colunas_feature"]]
X_novo_dummies = pd.get_dummies(X_novo, drop_first=False)
X_novo_alinhado = X_novo_dummies.reindex(columns=pacote2["colunas_dummies"], fill_value=0)
X_novo_norm = pacote2["scaler"].transform(X_novo_alinhado.astype(float))

pred = modelo2.predict(X_novo_norm, verbose=0)
classes_idx = (pred.flatten() > 0.5).astype(int)
resultado = pacote2["label_encoder"].inverse_transform(classes_idx)

print("Colunas dummies esperadas:", pacote2["colunas_dummies"])
print("Colunas alinhadas nos dados novos:", list(X_novo_alinhado.columns))
print("Previsoes:", resultado)
assert list(X_novo_alinhado.columns) == pacote2["colunas_dummies"], "alinhamento de colunas falhou"
print("\nOK: salvar/carregar modelo + previsao com categoria nova funcionou corretamente.")
