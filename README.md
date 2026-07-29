# Treinador de IA — Regressão e Classificação (PyQt6 + TensorFlow)

Aplicativo com interface gráfica para treinar modelos de rede neural
(regressão ou classificação) a partir de arquivos CSV/Excel, sem precisar
escrever código a cada novo conjunto de dados.


## Tecnologias

- Python
- TensorFlow
- PyQt6
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Funcionalidades

- Importação de datasets (.csv)
- Pré-processamento automático
- Treinamento de modelos de regressão
- Treinamento de modelos de classificação
- Visualização das métricas
- Interface gráfica amigável

## Objetivo

Facilitar o aprendizado de Machine Learning permitindo o treinamento de modelos sem necessidade de programação direta.

## Como executar

```bash
pip install -r requirements.txt
python main.py
## Instalação (PyCharm / qualquer ambiente local)

```bash
python -m venv venv
# Windows: venv\Scripts\activate | Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

## Execução

```bash
python app.py
```

Isso abre a janela do programa. **A partir daí, todo o passo a passo está
disponível dentro do próprio aplicativo, na aba "📘 Tutorial"** — não é
necessário consultar nada externo.

## Resumo do fluxo

1. **Aba "1. Dados"** — carregue um ou vários arquivos CSV/Excel (podem ser
   selecionados vários de uma vez), marque as colunas de entrada (Feature) e
   a coluna que deseja prever (Rótulo) usando checkboxes, e clique em
   "Preparar Dados".
2. **Aba "2. Treinamento"** — ajuste a arquitetura da rede (opcional) e
   clique em "Iniciar Treinamento". Acompanhe o progresso e a curva de perda
   em tempo real.
3. **Aba "3. Resultados"** — veja as métricas, os gráficos e o **Nível de
   Satisfação** do modelo (Excelente / Bom / Regular / Fraco /
   Insatisfatório). Salve o modelo treinado se estiver satisfeito.
4. **Aba "4. Usar IA"** — carregue novos dados (sem o rótulo) e gere
   previsões com o modelo treinado, exportando os resultados em CSV.

## Sobre o Google Colab

O Colab não possui tela gráfica, então esta janela PyQt6 não abre
diretamente nele. O código foi escrito com **foco no PyCharm**. As notas de
como adaptar a lógica para o Colab (com `ipywidgets`, por exemplo) estão
comentadas no final do arquivo `app.py` e também na aba "Tutorial" do
programa.

## Estrutura do repositório

```
├── app.py                      # aplicativo principal (PyQt6 + TensorFlow)
├── requirements.txt
├── README.md
├── LICENSE
├── exemplos/
│   ├── dataset_exemplo.csv     # dados sintéticos para testar o app
│   └── README.md
└── tests/
    ├── test_logica.py          # testes da lógica de dados/treino (sem GUI)
    └── test_save_predict.py    # teste de salvar/carregar modelo + prever
```

## Testando rapidamente

Não tem dataset em mãos? Use `exemplos/dataset_exemplo.csv` (dados
sintéticos, veja `exemplos/README.md`).

Para rodar os testes de lógica (não abrem a interface gráfica):

```bash
python tests/test_logica.py
python tests/test_save_predict.py
```

## Arquivos gerados ao salvar um modelo

- `modelo_treinado.keras` — o modelo em si.
- `modelo_treinado_preprocessamento.pkl` — normalizador, codificador de
  rótulo e lista de colunas usadas no treino. **Guarde os dois arquivos
  juntos**: ambos são necessários para reutilizar a IA depois.
