# Dataset de exemplo

`dataset_exemplo.csv` — dados **100% sintéticos** (gerados aleatoriamente,
sem nenhuma informação real ou pessoal), criados apenas para você testar o
aplicativo rapidamente sem precisar de dados próprios.

Colunas:
- `idade`, `anos_experiencia`, `area`, `nivel_educacao` — recursos (features)
- `salario` — bom para testar **regressão** (prever um valor numérico)
- `promovido` (0 ou 1) — bom para testar **classificação** (prever uma categoria)

## Como usar

1. Abra o app (`python app.py`) e vá na aba "1. Dados".
2. Carregue este arquivo (`dataset_exemplo.csv`).
3. Marque `idade`, `anos_experiencia`, `area`, `nivel_educacao` como **Feature**.
4. Marque `salario` **ou** `promovido` como **Rótulo** (teste os dois, um de
   cada vez, para ver regressão e classificação funcionando).
5. Clique em "Preparar Dados" e siga o fluxo normalmente.
