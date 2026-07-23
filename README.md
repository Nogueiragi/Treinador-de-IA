# Treinador de IA — Regressão e Classificação (PyQt6 + TensorFlow)
 
Aplicativo com interface gráfica para treinar modelos de rede neural
(regressão ou classificação) a partir de arquivos CSV/Excel, sem precisar
escrever código a cada novo conjunto de dados.
 
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

