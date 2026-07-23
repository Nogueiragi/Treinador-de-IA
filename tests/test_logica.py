"""Teste de ponta a ponta da lógica de dados/treino/previsão, sem abrir a GUI."""
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app

np.random.seed(0)
n = 600

# ---------- Teste 1: REGRESSÃO ----------
df_reg = pd.DataFrame({
    "idade": np.random.randint(18, 70, n),
    "anos_experiencia": np.random.randint(0, 40, n),
    "area": np.random.choice(["TI", "Saude", "Vendas"], n),
})
df_reg["salario"] = (
    2000 + df_reg["idade"] * 30 + df_reg["anos_experiencia"] * 150
    + (df_reg["area"] == "TI") * 2000 + np.random.normal(0, 500, n)
)

tarefa = app.detectar_tipo_tarefa(df_reg["salario"])
assert tarefa == "regressao", f"esperado regressao, veio {tarefa}"
print("OK deteccao regressao:", tarefa)

X = pd.get_dummies(df_reg[["idade", "anos_experiencia", "area"]])
scaler = StandardScaler()
Xn = scaler.fit_transform(X.astype(float))
y = df_reg["salario"].values.astype(float)
X_train, X_test, y_train, y_test = train_test_split(Xn, y, test_size=0.2, random_state=42)

modelo = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="linear"),
])
modelo.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse", metrics=["mae"])
hist = modelo.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, verbose=0)
pred = modelo.predict(X_test, verbose=0).flatten()

from sklearn.metrics import r2_score
r2 = r2_score(y_test, pred)
print("OK treino regressao. R2 =", r2)
nivel, cor = app.calcular_satisfacao("regressao", {"r2": r2})
print("Nivel de satisfacao:", nivel, cor)

# ---------- Teste 2: CLASSIFICAÇÃO (multi-classe) ----------
df_cls = pd.DataFrame({
    "nota1": np.random.uniform(0, 10, n),
    "nota2": np.random.uniform(0, 10, n),
    "frequencia": np.random.uniform(0, 100, n),
})
media = (df_cls["nota1"] + df_cls["nota2"]) / 2
condicoes = [media < 5, (media >= 5) & (media < 7), media >= 7]
categorias = ["Reprovado", "Recuperacao", "Aprovado"]
df_cls["situacao"] = np.select(condicoes, categorias, default="Reprovado")

tarefa2 = app.detectar_tipo_tarefa(df_cls["situacao"])
assert tarefa2 == "classificacao", f"esperado classificacao, veio {tarefa2}"
print("OK deteccao classificacao:", tarefa2)

le = LabelEncoder()
y2 = le.fit_transform(df_cls["situacao"])
num_classes = len(le.classes_)
y2_cat = keras.utils.to_categorical(y2, num_classes=num_classes)

X2 = df_cls[["nota1", "nota2", "frequencia"]].values.astype(float)
scaler2 = StandardScaler()
X2n = scaler2.fit_transform(X2)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2n, y2_cat, test_size=0.2, random_state=42)

modelo2 = keras.Sequential([
    keras.layers.Input(shape=(X2_train.shape[1],)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(num_classes, activation="softmax"),
])
modelo2.compile(optimizer=keras.optimizers.Adam(0.01), loss="categorical_crossentropy", metrics=["accuracy"])
modelo2.fit(X2_train, y2_train, validation_data=(X2_test, y2_test), epochs=20, batch_size=32, verbose=0)
pred2 = modelo2.predict(X2_test, verbose=0)
from sklearn.metrics import f1_score
y_pred_idx = np.argmax(pred2, axis=1)
y_real_idx = np.argmax(y2_test, axis=1)
f1 = f1_score(y_real_idx, y_pred_idx, average="weighted")
print("OK treino classificacao multi-classe. F1 =", f1)
nivel2, cor2 = app.calcular_satisfacao("classificacao", {"f1": f1})
print("Nivel de satisfacao:", nivel2, cor2)

# ---------- Teste 3: carregar_arquivos com múltiplos CSVs ----------
df_reg.iloc[:300].to_csv("/tmp/parte1.csv", index=False)
df_reg.iloc[300:].to_csv("/tmp/parte2.csv", index=False)
df_junto = app.carregar_arquivos(["/tmp/parte1.csv", "/tmp/parte2.csv"])
assert len(df_junto) == len(df_reg), "concatenacao de multiplos arquivos falhou"
print("OK carregamento e concatenacao de multiplos arquivos:", len(df_junto), "linhas")

print("\nTODOS OS TESTES DE LOGICA PASSARAM COM SUCESSO.")
