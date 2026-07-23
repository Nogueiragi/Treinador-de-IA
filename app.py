# -*- coding: utf-8 -*-
"""
==================================================================================
TREINADOR DE IA - REGRESSÃO E CLASSIFICAÇÃO (PyQt6 + TensorFlow)
==================================================================================

DESCRIÇÃO GERAL:
Este aplicativo permite que qualquer usuário, mesmo sem saber programar, treine
um modelo de Inteligência Artificial (rede neural) usando o TensorFlow/Keras,
a partir de arquivos de dados (CSV/Excel), com uma interface gráfica completa.

FLUXO DE USO (resumido - veja também a aba "Tutorial" dentro do programa):
  1) Aba "1. Dados"        -> Carregar um ou vários arquivos, escolher quais
                               colunas são "recursos" (features) e qual coluna
                               é o "rótulo" (target/label) via checkbox.
  2) Aba "2. Treinamento"  -> Configurar a rede neural (camadas, épocas, etc.)
                               e iniciar o treinamento.
  3) Aba "3. Resultados"   -> Ver métricas, gráficos e o "nível de satisfação"
                               do modelo (quão bom ele ficou).
  4) Aba "4. Usar IA"      -> Usar o modelo já treinado para prever novos dados
                               e exportar os resultados.

FOCO: PyCharm (execução local com interface gráfica nativa).
IMPORTANTE SOBRE O GOOGLE COLAB: o Colab roda em um servidor remoto sem tela
gráfica, portanto janelas do PyQt6 NÃO abrem diretamente no navegador do Colab.
Deixei instruções de como adaptar/rodar no Colab no final deste arquivo e
também dentro da aba "Tutorial" do programa.

DEPENDÊNCIAS (ver requirements.txt):
    pip install PyQt6 tensorflow pandas numpy scikit-learn matplotlib openpyxl
==================================================================================
"""

import os
import sys
import pickle
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

# --- Bibliotecas de Machine Learning -------------------------------------------
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# --- Gráficos embutidos na interface --------------------------------------------
import matplotlib
matplotlib.use("QtAgg")  # backend compatível com PyQt6
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# --- Interface gráfica PyQt6 -----------------------------------------------------
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QCheckBox, QTabWidget, QMessageBox, QProgressBar, QComboBox, QSpinBox,
    QDoubleSpinBox, QLineEdit, QTextEdit, QTextBrowser, QGroupBox, QRadioButton,
    QButtonGroup, QSplitter, QHeaderView, QFrame
)


# ==================================================================================
# FUNÇÕES AUXILIARES
# ==================================================================================

def carregar_arquivos(caminhos):
    """
    Lê um ou VÁRIOS arquivos (CSV ou Excel) e os concatena em um único DataFrame.
    Isso atende ao requisito de "receber grande quantidade de arquivos externos":
    o usuário pode selecionar dezenas de arquivos de uma vez, desde que tenham
    as mesmas colunas (ex.: exportações mensais de um sistema).

    Parâmetros:
        caminhos (list[str]): lista de caminhos de arquivos escolhidos pelo usuário.

    Retorna:
        pd.DataFrame: dados de todos os arquivos, empilhados (concatenados).
    """
    dataframes = []
    for caminho in caminhos:
        extensao = os.path.splitext(caminho)[1].lower()
        if extensao == ".csv":
            # sep=None + engine='python' faz o pandas detectar o separador (, ; ou tab)
            df = pd.read_csv(caminho, sep=None, engine="python")
        elif extensao in (".xlsx", ".xls"):
            df = pd.read_excel(caminho)
        else:
            raise ValueError(f"Formato de arquivo não suportado: {extensao}")
        dataframes.append(df)

    # Concatena todos os arquivos um embaixo do outro (mesmas colunas esperadas)
    df_final = pd.concat(dataframes, ignore_index=True, sort=False)
    return df_final


def detectar_tipo_tarefa(coluna_rotulo: pd.Series) -> str:
    """
    Tenta adivinhar automaticamente se o problema é de REGRESSÃO ou CLASSIFICAÇÃO,
    olhando para o tipo de dado da coluna escolhida como rótulo.

    Regra utilizada (heurística simples e comum em AutoML):
      - Se a coluna for texto (object/category)         -> classificação
      - Se for número, mas com poucos valores distintos  -> classificação
        (ex.: 0/1, ou notas de 1 a 5)
      - Se for número com muitos valores distintos       -> regressão
        (ex.: preço, salário, temperatura)
    """
    # is_numeric_dtype cobre int/float; qualquer coisa fora disso (object, a nova
    # StringDtype do pandas 2.x/3.x, category, bool, etc.) é tratada como texto/categoria.
    if not pd.api.types.is_numeric_dtype(coluna_rotulo):
        return "classificacao"

    valores_unicos = coluna_rotulo.nunique()
    if pd.api.types.is_integer_dtype(coluna_rotulo) and valores_unicos <= 20:
        return "classificacao"

    return "regressao"


def calcular_satisfacao(tarefa: str, metricas: dict) -> tuple:
    """
    Traduz o valor numérico da métrica principal em um "nível de satisfação"
    (linguagem simples), para que o usuário entenda a qualidade do modelo
    sem precisar interpretar números de estatística.

    Retorna:
        (texto_nivel, cor_css) -> ex.: ("Excelente", "#2ecc71")
    """
    if tarefa == "regressao":
        valor = metricas["r2"]  # R² varia de -infinito a 1 (quanto mais perto de 1, melhor)
        if valor >= 0.90:
            return "Excelente", "#2ecc71"
        elif valor >= 0.75:
            return "Bom", "#27ae60"
        elif valor >= 0.50:
            return "Regular", "#f39c12"
        elif valor >= 0.0:
            return "Fraco", "#e67e22"
        else:
            return "Insatisfatório", "#e74c3c"
    else:
        valor = metricas["f1"]  # F1-score varia de 0 a 1
        if valor >= 0.90:
            return "Excelente", "#2ecc71"
        elif valor >= 0.80:
            return "Bom", "#27ae60"
        elif valor >= 0.65:
            return "Regular", "#f39c12"
        elif valor >= 0.50:
            return "Fraco", "#e67e22"
        else:
            return "Insatisfatório", "#e74c3c"


# ==================================================================================
# CANVAS DE GRÁFICOS (Matplotlib embutido no PyQt6)
# ==================================================================================

class CanvasGrafico(FigureCanvasQTAgg):
    """
    Widget de gráfico que pode ser inserido diretamente na interface do PyQt6.
    Usado para: curva de perda (loss) durante o treino, gráfico de dispersão
    (previsto x real) na regressão, e matriz de confusão na classificação.
    """

    def __init__(self, largura=5, altura=4, dpi=100):
        self.fig = Figure(figsize=(largura, altura), dpi=dpi)
        self.eixo = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def limpar(self):
        self.eixo.clear()
        self.draw()


# ==================================================================================
# CALLBACK PERSONALIZADO DO KERAS -> ENVIA PROGRESSO PARA A INTERFACE
# ==================================================================================

class CallbackDeProgresso(keras.callbacks.Callback):
    """
    O Keras chama automaticamente os métodos on_epoch_end etc. durante o
    model.fit(). Aqui nós "escutamos" esses eventos e repassamos para a
    thread de treinamento (via função emissora), que por sua vez emite um
    sinal do PyQt6 para atualizar a barra de progresso e o gráfico ao vivo.
    """

    def __init__(self, funcao_emissora, total_epocas):
        super().__init__()
        self.funcao_emissora = funcao_emissora
        self.total_epocas = total_epocas

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.funcao_emissora(epoch + 1, self.total_epocas, logs)


# ==================================================================================
# THREAD DE TREINAMENTO (para não travar a interface gráfica durante o fit)
# ==================================================================================

class ThreadTreinamento(QThread):
    """
    Roda o treinamento do TensorFlow em uma thread separada da interface.
    Isso é ESSENCIAL: se o treinamento rodasse na thread principal, a janela
    do programa "congelaria" (ficaria sem responder) até o treino terminar.

    Sinais emitidos (comunicação segura entre threads no PyQt6):
        progresso(int epoca, int total, dict logs) -> a cada época concluída
        finalizado(dict resultado)                  -> quando o treino termina bem
        erro(str mensagem)                           -> se algo der errado
    """
    progresso = pyqtSignal(int, int, dict)
    finalizado = pyqtSignal(dict)
    erro = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config  # dicionário com todos os dados/parâmetros necessários

    def run(self):
        try:
            cfg = self.config
            X_train, X_test = cfg["X_train"], cfg["X_test"]
            y_train, y_test = cfg["y_train"], cfg["y_test"]
            tarefa = cfg["tarefa"]
            num_classes = cfg.get("num_classes", 1)

            # ---------------------------------------------------------------
            # MONTAGEM DINÂMICA DA REDE NEURAL (arquitetura Sequential/Keras)
            # ---------------------------------------------------------------
            modelo = keras.Sequential(name="modelo_treinado")
            modelo.add(keras.layers.Input(shape=(X_train.shape[1],)))

            for neuronios in cfg["camadas_ocultas"]:
                modelo.add(keras.layers.Dense(neuronios, activation="relu"))
                if cfg.get("dropout", 0) > 0:
                    modelo.add(keras.layers.Dropout(cfg["dropout"]))

            # Camada de saída depende do tipo de tarefa
            if tarefa == "regressao":
                modelo.add(keras.layers.Dense(1, activation="linear"))
                loss = "mse"
                metrics = ["mae"]
            elif num_classes == 2:
                modelo.add(keras.layers.Dense(1, activation="sigmoid"))
                loss = "binary_crossentropy"
                metrics = ["accuracy"]
            else:
                modelo.add(keras.layers.Dense(num_classes, activation="softmax"))
                loss = "categorical_crossentropy"
                metrics = ["accuracy"]

            otimizador = keras.optimizers.Adam(learning_rate=cfg["taxa_aprendizado"])
            modelo.compile(optimizer=otimizador, loss=loss, metrics=metrics)

            # Repassa o progresso de cada época para a interface gráfica
            callback_progresso = CallbackDeProgresso(
                funcao_emissora=lambda ep, tot, logs: self.progresso.emit(ep, tot, logs),
                total_epocas=cfg["epocas"]
            )

            # Interrompe o treino cedo se o modelo parar de melhorar (evita overfitting)
            parada_antecipada = keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True
            )

            historico = modelo.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=cfg["epocas"],
                batch_size=cfg["batch_size"],
                callbacks=[callback_progresso, parada_antecipada],
                verbose=0
            )

            # ---------------------------------------------------------------
            # AVALIAÇÃO DO MODELO NO CONJUNTO DE TESTE (dados nunca vistos)
            # ---------------------------------------------------------------
            previsoes = modelo.predict(X_test, verbose=0)

            metricas = {}
            if tarefa == "regressao":
                y_pred = previsoes.flatten()
                metricas["mae"] = float(mean_absolute_error(y_test, y_pred))
                metricas["mse"] = float(mean_squared_error(y_test, y_pred))
                metricas["rmse"] = float(np.sqrt(metricas["mse"]))
                metricas["r2"] = float(r2_score(y_test, y_pred))
                y_pred_final = y_pred
            else:
                if num_classes == 2:
                    y_pred = (previsoes.flatten() > 0.5).astype(int)
                    y_real = y_test
                else:
                    y_pred = np.argmax(previsoes, axis=1)
                    y_real = np.argmax(y_test, axis=1)

                metricas["accuracy"] = float(accuracy_score(y_real, y_pred))
                metricas["precision"] = float(precision_score(y_real, y_pred, average="weighted", zero_division=0))
                metricas["recall"] = float(recall_score(y_real, y_pred, average="weighted", zero_division=0))
                metricas["f1"] = float(f1_score(y_real, y_pred, average="weighted", zero_division=0))
                metricas["matriz_confusao"] = confusion_matrix(y_real, y_pred)
                y_pred_final = y_pred
                y_test = y_real  # normaliza para o formato "classe única" no resultado

            resultado = {
                "modelo": modelo,
                "historico": historico.history,
                "metricas": metricas,
                "y_test": y_test,
                "y_pred": y_pred_final,
            }
            self.finalizado.emit(resultado)

        except Exception as e:
            self.erro.emit(f"{e}\n\n{traceback.format_exc()}")


# ==================================================================================
# JANELA PRINCIPAL
# ==================================================================================

class JanelaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Treinador de IA - Regressão e Classificação (TensorFlow)")
        self.resize(1200, 800)

        # --- Estado interno da aplicação (guardado entre as abas) -------------
        self.df_original = None          # DataFrame bruto carregado dos arquivos
        self.colunas_feature = []        # colunas escolhidas como recursos
        self.coluna_rotulo = None        # coluna escolhida como rótulo
        self.scaler = None               # normalizador (StandardScaler) salvo
        self.label_encoder = None        # LabelEncoder salvo (classificação)
        self.colunas_dummies = None      # nomes das colunas após one-hot encoding
        self.tarefa = None               # "regressao" ou "classificacao"
        self.num_classes = 1
        self.modelo_treinado = None
        self.thread_treino = None
        self.historico_epocas = {"loss": [], "val_loss": []}

        self._montar_interface()

    # ------------------------------------------------------------------------
    # MONTAGEM GERAL DA INTERFACE (abas)
    # ------------------------------------------------------------------------
    def _montar_interface(self):
        abas = QTabWidget()
        self.setCentralWidget(abas)

        abas.addTab(self._criar_aba_tutorial(), "📘 Tutorial")
        abas.addTab(self._criar_aba_dados(), "1. Dados")
        abas.addTab(self._criar_aba_treinamento(), "2. Treinamento")
        abas.addTab(self._criar_aba_resultados(), "3. Resultados")
        abas.addTab(self._criar_aba_usar_ia(), "4. Usar IA")

        self.abas = abas

    # ==========================================================================
    # ABA 0 - TUTORIAL
    # ==========================================================================
    def _criar_aba_tutorial(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        navegador = QTextBrowser()
        navegador.setOpenExternalLinks(True)
        navegador.setHtml(self._texto_tutorial_html())
        layout.addWidget(navegador)
        return widget

    def _texto_tutorial_html(self):
        return """
        <h1>📘 Tutorial - Como usar este programa</h1>

        <h2>Visão geral</h2>
        <p>Este programa treina um modelo de <b>Inteligência Artificial</b> (rede neural,
        usando TensorFlow/Keras) para resolver dois tipos de problema:</p>
        <ul>
            <li><b>Regressão</b>: prever um número (ex.: preço de um imóvel, nota de um aluno).</li>
            <li><b>Classificação</b>: prever uma categoria (ex.: aprovado/reprovado, tipo de cliente).</li>
        </ul>

        <h2>Passo 1 - Aba "1. Dados"</h2>
        <ol>
            <li>Clique em <b>"Selecionar Arquivo(s)"</b> e escolha um ou vários arquivos
                CSV/Excel. Você pode selecionar <b>vários arquivos de uma vez</b>
                (segurando Ctrl ou Shift) — eles serão empilhados automaticamente,
                desde que tenham as mesmas colunas.</li>
            <li>A tabela de colunas vai aparecer. Marque a caixa <b>"Feature"</b>
                em cada coluna que deve ser usada como informação de entrada
                (recurso) para o modelo.</li>
            <li>Marque a caixa <b>"Rótulo"</b> na coluna que o modelo deve
                <b>aprender a prever</b>. Só é permitido escolher UMA coluna como rótulo.</li>
            <li>Escolha o tipo de tarefa: deixe em <b>"Automático"</b> para o programa
                decidir sozinho (baseado no tipo de dado da coluna), ou force manualmente
                "Regressão" ou "Classificação".</li>
            <li>Clique em <b>"Preparar Dados"</b>. O programa vai normalizar os números,
                transformar texto em números (one-hot encoding) e separar uma parte
                dos dados para teste.</li>
        </ol>

        <h2>Passo 2 - Aba "2. Treinamento"</h2>
        <ol>
            <li>Configure a <b>arquitetura da rede neural</b>: número de neurônios em
                cada camada oculta (separados por vírgula, ex.: <code>64,32</code>),
                número de épocas (quantas vezes o modelo revisa os dados) e o tamanho do lote (batch size).</li>
            <li>Clique em <b>"Iniciar Treinamento"</b>. Acompanhe a barra de progresso
                e o gráfico de perda (loss) em tempo real.</li>
            <li>O treinamento roda em segundo plano, então você pode continuar usando
                a janela normalmente.</li>
        </ol>

        <h2>Passo 3 - Aba "3. Resultados"</h2>
        <p>Aqui você vê as <b>métricas de avaliação</b> do modelo (ex.: R², MAE, Acurácia,
        F1-score), os <b>gráficos</b> (curva de aprendizado, dispersão previsto x real,
        ou matriz de confusão) e o <b>Nível de Satisfação</b>, que traduz os números
        em uma classificação simples: <i>Excelente, Bom, Regular, Fraco ou Insatisfatório</i>.</p>
        <p>Se estiver satisfeito com o resultado, clique em <b>"Salvar Modelo"</b> para
        guardar a IA treinada em disco (arquivo <code>.keras</code> + arquivo de
        pré-processamento <code>.pkl</code>).</p>

        <h2>Passo 4 - Aba "4. Usar IA"</h2>
        <p>Depois de treinado (ou carregando um modelo salvo anteriormente), você pode
        importar um novo arquivo de dados (sem o rótulo, ou com ele para comparar) e
        clicar em <b>"Prever"</b>. Os resultados aparecem em uma tabela e podem ser
        exportados para CSV.</p>

        <h2>⚠️ Sobre o Google Colab</h2>
        <p>O Google Colab executa o código em um servidor remoto, <b>sem tela gráfica</b>.
        Por isso, janelas do PyQt6 (como esta) <b>não abrem diretamente</b> dentro do
        Colab. Este programa foi desenhado com <b>foco no PyCharm</b> (ou em qualquer
        ambiente Python local com interface gráfica).</p>
        <p>Se quiser rodar no Colab mesmo assim, duas opções comuns:</p>
        <ul>
            <li>Usar um servidor de área de trabalho virtual (VNC) dentro do Colab
                (mais complexo, exige pacotes extras como <code>pyvirtualdisplay</code>).</li>
            <li><b>Recomendado</b>: usar a lógica deste mesmo código (as funções de
                carregar dados, pré-processar, treinar e avaliar) só que com
                <code>ipywidgets</code> no lugar do PyQt6, já que o Colab tem
                suporte nativo a widgets interativos no navegador.</li>
        </ul>
        <p>No PyCharm, basta rodar: <code>python app.py</code></p>

        <h2>Dicas</h2>
        <ul>
            <li>Comece com poucas camadas/neurônios (ex.: 32,16) e poucas épocas
                (ex.: 50) para testar rápido, depois aumente se precisar de mais precisão.</li>
            <li>Se o "Nível de Satisfação" vier baixo, tente: mais dados, mais épocas,
                mudar a arquitetura da rede, ou revisar quais colunas fazem sentido
                como recursos.</li>
        </ul>
        """

    # ==========================================================================
    # ABA 1 - DADOS (carregar arquivo + selecionar features/rótulo)
    # ==========================================================================
    def _criar_aba_dados(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # --- Bloco de seleção de arquivo(s) ------------------------------------
        grupo_arquivo = QGroupBox("Origem dos Dados")
        layout_arquivo = QHBoxLayout(grupo_arquivo)

        self.botao_selecionar_arquivo = QPushButton("📂 Selecionar Arquivo(s) CSV/Excel")
        self.botao_selecionar_arquivo.clicked.connect(self._acao_selecionar_arquivos)
        self.rotulo_arquivos = QLabel("Nenhum arquivo selecionado.")
        self.rotulo_arquivos.setWordWrap(True)

        layout_arquivo.addWidget(self.botao_selecionar_arquivo)
        layout_arquivo.addWidget(self.rotulo_arquivos, stretch=1)
        layout.addWidget(grupo_arquivo)

        # --- Prévia dos dados ----------------------------------------------------
        grupo_previa = QGroupBox("Prévia dos Dados (primeiras linhas)")
        layout_previa = QVBoxLayout(grupo_previa)
        self.tabela_previa = QTableWidget()
        self.tabela_previa.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout_previa.addWidget(self.tabela_previa)
        layout.addWidget(grupo_previa, stretch=1)

        # --- Seleção de colunas (Feature / Rótulo) via checkbox ------------------
        grupo_colunas = QGroupBox("Seleção de Recursos (Features) e Rótulo (Label)")
        layout_colunas = QVBoxLayout(grupo_colunas)
        self.tabela_colunas = QTableWidget(0, 4)
        self.tabela_colunas.setHorizontalHeaderLabels(
            ["Coluna", "Tipo de Dado", "Usar como Feature", "Usar como Rótulo (Alvo)"]
        )
        self.tabela_colunas.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        layout_colunas.addWidget(self.tabela_colunas)
        layout.addWidget(grupo_colunas, stretch=1)

        # --- Configurações de preparação de dados --------------------------------
        grupo_config = QGroupBox("Configurações de Preparação")
        layout_config = QGridLayout(grupo_config)

        layout_config.addWidget(QLabel("Tipo de Tarefa:"), 0, 0)
        self.combo_tarefa = QComboBox()
        self.combo_tarefa.addItems(["Automático (recomendado)", "Regressão", "Classificação"])
        layout_config.addWidget(self.combo_tarefa, 0, 1)

        layout_config.addWidget(QLabel("% dos dados para Teste:"), 0, 2)
        self.spin_teste = QSpinBox()
        self.spin_teste.setRange(10, 50)
        self.spin_teste.setValue(20)
        self.spin_teste.setSuffix(" %")
        layout_config.addWidget(self.spin_teste, 0, 3)

        layout.addWidget(grupo_config)

        # --- Botão preparar dados --------------------------------------------------
        self.botao_preparar = QPushButton("✅ Preparar Dados")
        self.botao_preparar.clicked.connect(self._acao_preparar_dados)
        self.botao_preparar.setEnabled(False)
        layout.addWidget(self.botao_preparar)

        self.rotulo_status_dados = QLabel("")
        layout.addWidget(self.rotulo_status_dados)

        return widget

    def _acao_selecionar_arquivos(self):
        """Abre o diálogo de seleção de arquivos (permite múltiplos arquivos)."""
        caminhos, _ = QFileDialog.getOpenFileNames(
            self, "Selecione um ou mais arquivos de dados", "",
            "Arquivos de Dados (*.csv *.xlsx *.xls);;Todos os Arquivos (*)"
        )
        if not caminhos:
            return

        try:
            self.df_original = carregar_arquivos(caminhos)
        except Exception as e:
            QMessageBox.critical(self, "Erro ao ler arquivo(s)", str(e))
            return

        nomes = [os.path.basename(c) for c in caminhos]
        self.rotulo_arquivos.setText(
            f"{len(caminhos)} arquivo(s) carregado(s): {', '.join(nomes[:5])}"
            + (" ..." if len(nomes) > 5 else "")
            + f"\nTotal de linhas: {len(self.df_original)} | Total de colunas: {len(self.df_original.columns)}"
        )

        self._preencher_previa()
        self._preencher_tabela_colunas()
        self.botao_preparar.setEnabled(True)

    def _preencher_previa(self):
        """Mostra as primeiras 20 linhas do DataFrame na tabela de prévia."""
        df_amostra = self.df_original.head(20)
        self.tabela_previa.setRowCount(len(df_amostra))
        self.tabela_previa.setColumnCount(len(df_amostra.columns))
        self.tabela_previa.setHorizontalHeaderLabels(list(df_amostra.columns))

        for i in range(len(df_amostra)):
            for j, col in enumerate(df_amostra.columns):
                valor = str(df_amostra.iloc[i, j])
                self.tabela_previa.setItem(i, j, QTableWidgetItem(valor))

    def _preencher_tabela_colunas(self):
        """Cria uma linha por coluna do DataFrame, com checkboxes de Feature/Rótulo."""
        colunas = list(self.df_original.columns)
        self.tabela_colunas.setRowCount(len(colunas))
        self.checkboxes_feature = {}
        self.checkboxes_rotulo = {}

        for i, col in enumerate(colunas):
            self.tabela_colunas.setItem(i, 0, QTableWidgetItem(col))
            tipo = str(self.df_original[col].dtype)
            self.tabela_colunas.setItem(i, 1, QTableWidgetItem(tipo))

            chk_feature = QCheckBox()
            chk_feature.setChecked(True)  # por padrão, marca tudo como feature
            self._centralizar_checkbox(self.tabela_colunas, i, 2, chk_feature)
            self.checkboxes_feature[col] = chk_feature

            chk_rotulo = QCheckBox()
            chk_rotulo.stateChanged.connect(
                lambda estado, coluna=col: self._ao_marcar_rotulo(coluna, estado)
            )
            self._centralizar_checkbox(self.tabela_colunas, i, 3, chk_rotulo)
            self.checkboxes_rotulo[col] = chk_rotulo

    @staticmethod
    def _centralizar_checkbox(tabela, linha, coluna, checkbox):
        """Centraliza visualmente um QCheckBox dentro de uma célula da tabela."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addWidget(checkbox)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        tabela.setCellWidget(linha, coluna, container)

    def _ao_marcar_rotulo(self, coluna, estado):
        """
        Garante que apenas UMA coluna pode ser marcada como Rótulo por vez
        (desmarca as demais automaticamente) e também remove essa coluna
        da lista de Features, pois o rótulo não deve ser usado como entrada.
        """
        if estado == Qt.CheckState.Checked.value:
            for outra_col, chk in self.checkboxes_rotulo.items():
                if outra_col != coluna:
                    chk.blockSignals(True)
                    chk.setChecked(False)
                    chk.blockSignals(False)
            # Se a coluna virou rótulo, ela não pode continuar marcada como feature
            self.checkboxes_feature[coluna].setChecked(False)
            self.checkboxes_feature[coluna].setEnabled(False)
        else:
            self.checkboxes_feature[coluna].setEnabled(True)

    def _acao_preparar_dados(self):
        """
        Lê as escolhas do usuário (features/rótulo), faz o pré-processamento
        completo (one-hot encoding, normalização, split treino/teste) e deixa
        tudo pronto para a aba de Treinamento.
        """
        try:
            self.colunas_feature = [c for c, chk in self.checkboxes_feature.items() if chk.isChecked()]
            colunas_rotulo_marcadas = [c for c, chk in self.checkboxes_rotulo.items() if chk.isChecked()]

            if not self.colunas_feature:
                raise ValueError("Selecione ao menos uma coluna como Feature (recurso).")
            if len(colunas_rotulo_marcadas) != 1:
                raise ValueError("Selecione exatamente UMA coluna como Rótulo (Label).")

            self.coluna_rotulo = colunas_rotulo_marcadas[0]

            # Remove linhas com valores ausentes nas colunas escolhidas
            colunas_uso = self.colunas_feature + [self.coluna_rotulo]
            df = self.df_original[colunas_uso].dropna().reset_index(drop=True)

            X_bruto = df[self.colunas_feature]
            y_bruto = df[self.coluna_rotulo]

            # --- Define o tipo de tarefa (automático ou manual) -----------------
            escolha = self.combo_tarefa.currentText()
            if escolha.startswith("Automático"):
                self.tarefa = detectar_tipo_tarefa(y_bruto)
            elif escolha == "Regressão":
                self.tarefa = "regressao"
            else:
                self.tarefa = "classificacao"

            # --- Pré-processamento das FEATURES (X) -----------------------------
            # One-hot encoding para colunas de texto/categóricas
            X_dummies = pd.get_dummies(X_bruto, drop_first=False)
            self.colunas_dummies = list(X_dummies.columns)  # necessário para prever depois

            self.scaler = StandardScaler()
            X_normalizado = self.scaler.fit_transform(X_dummies.astype(float))

            # --- Pré-processamento do RÓTULO (y) ---------------------------------
            self.label_encoder = None
            self.num_classes = 1
            if self.tarefa == "classificacao":
                self.label_encoder = LabelEncoder()
                y_codificado = self.label_encoder.fit_transform(y_bruto)
                self.num_classes = len(self.label_encoder.classes_)
                if self.num_classes > 2:
                    y_final = keras.utils.to_categorical(y_codificado, num_classes=self.num_classes)
                else:
                    y_final = y_codificado
            else:
                y_final = y_bruto.values.astype(float)

            # --- Divisão treino/teste ---------------------------------------------
            tamanho_teste = self.spin_teste.value() / 100.0
            X_train, X_test, y_train, y_test = train_test_split(
                X_normalizado, y_final, test_size=tamanho_teste, random_state=42
            )

            # Guarda tudo para a próxima etapa (treinamento)
            self.dados_preparados = {
                "X_train": X_train, "X_test": X_test,
                "y_train": y_train, "y_test": y_test,
                "tarefa": self.tarefa, "num_classes": self.num_classes
            }

            self.rotulo_status_dados.setText(
                f"✅ Dados preparados! Tarefa detectada: <b>{self.tarefa.upper()}</b> | "
                f"Features finais (após one-hot): {len(self.colunas_dummies)} | "
                f"Treino: {len(X_train)} linhas | Teste: {len(X_test)} linhas"
            )
            self.rotulo_status_dados.setTextFormat(Qt.TextFormat.RichText)

            QMessageBox.information(self, "Dados prontos",
                                     "Dados preparados com sucesso! Vá para a aba '2. Treinamento'.")
            self.abas.setCurrentIndex(2)

        except Exception as e:
            QMessageBox.critical(self, "Erro ao preparar dados", str(e))

    # ==========================================================================
    # ABA 2 - TREINAMENTO
    # ==========================================================================
    def _criar_aba_treinamento(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        grupo_arquitetura = QGroupBox("Configuração da Rede Neural")
        grade = QGridLayout(grupo_arquitetura)

        grade.addWidget(QLabel("Camadas ocultas (neurônios, separados por vírgula):"), 0, 0)
        self.campo_camadas = QLineEdit("64,32")
        grade.addWidget(self.campo_camadas, 0, 1)

        grade.addWidget(QLabel("Épocas:"), 1, 0)
        self.spin_epocas = QSpinBox()
        self.spin_epocas.setRange(1, 2000)
        self.spin_epocas.setValue(100)
        grade.addWidget(self.spin_epocas, 1, 1)

        grade.addWidget(QLabel("Tamanho do lote (batch size):"), 2, 0)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 4096)
        self.spin_batch.setValue(32)
        grade.addWidget(self.spin_batch, 2, 1)

        grade.addWidget(QLabel("Taxa de aprendizado:"), 3, 0)
        self.spin_taxa = QDoubleSpinBox()
        self.spin_taxa.setDecimals(5)
        self.spin_taxa.setRange(0.00001, 1.0)
        self.spin_taxa.setSingleStep(0.0005)
        self.spin_taxa.setValue(0.001)
        grade.addWidget(self.spin_taxa, 3, 1)

        grade.addWidget(QLabel("Dropout (0 = desativado):"), 4, 0)
        self.spin_dropout = QDoubleSpinBox()
        self.spin_dropout.setDecimals(2)
        self.spin_dropout.setRange(0.0, 0.9)
        self.spin_dropout.setSingleStep(0.05)
        self.spin_dropout.setValue(0.0)
        grade.addWidget(self.spin_dropout, 4, 1)

        layout.addWidget(grupo_arquitetura)

        self.botao_treinar = QPushButton("🚀 Iniciar Treinamento")
        self.botao_treinar.clicked.connect(self._acao_iniciar_treinamento)
        layout.addWidget(self.botao_treinar)

        self.barra_progresso = QProgressBar()
        layout.addWidget(self.barra_progresso)

        self.rotulo_epoca_atual = QLabel("Aguardando início do treinamento...")
        layout.addWidget(self.rotulo_epoca_atual)

        self.canvas_treino = CanvasGrafico(largura=6, altura=4)
        layout.addWidget(self.canvas_treino)

        return widget

    def _acao_iniciar_treinamento(self):
        if not hasattr(self, "dados_preparados"):
            QMessageBox.warning(self, "Dados não preparados",
                                 "Volte na aba '1. Dados' e clique em 'Preparar Dados' primeiro.")
            return

        try:
            camadas_texto = self.campo_camadas.text().strip()
            camadas_ocultas = [int(n.strip()) for n in camadas_texto.split(",") if n.strip()]
            if not camadas_ocultas:
                raise ValueError("Informe ao menos uma camada oculta (ex.: 64,32).")
        except ValueError:
            QMessageBox.critical(self, "Erro de configuração",
                                  "O campo de camadas deve conter números separados por vírgula. Ex.: 64,32")
            return

        config = dict(self.dados_preparados)
        config.update({
            "camadas_ocultas": camadas_ocultas,
            "epocas": self.spin_epocas.value(),
            "batch_size": self.spin_batch.value(),
            "taxa_aprendizado": self.spin_taxa.value(),
            "dropout": self.spin_dropout.value(),
        })

        self.historico_epocas = {"loss": [], "val_loss": []}
        self.barra_progresso.setValue(0)
        self.botao_treinar.setEnabled(False)
        self.rotulo_epoca_atual.setText("Iniciando treinamento...")

        self.thread_treino = ThreadTreinamento(config)
        self.thread_treino.progresso.connect(self._ao_progredir_epoca)
        self.thread_treino.finalizado.connect(self._ao_finalizar_treinamento)
        self.thread_treino.erro.connect(self._ao_ocorrer_erro_treino)
        self.thread_treino.start()

    def _ao_progredir_epoca(self, epoca, total, logs):
        """Atualizado a cada época concluída (executado na thread principal)."""
        percentual = int((epoca / total) * 100)
        self.barra_progresso.setValue(percentual)

        texto_loss = " | ".join(f"{k}: {v:.4f}" for k, v in logs.items())
        self.rotulo_epoca_atual.setText(f"Época {epoca}/{total} -> {texto_loss}")

        self.historico_epocas["loss"].append(logs.get("loss"))
        self.historico_epocas["val_loss"].append(logs.get("val_loss"))

        # Redesenha o gráfico de perda ao vivo
        self.canvas_treino.eixo.clear()
        self.canvas_treino.eixo.plot(self.historico_epocas["loss"], label="Perda (treino)")
        self.canvas_treino.eixo.plot(self.historico_epocas["val_loss"], label="Perda (validação)")
        self.canvas_treino.eixo.set_xlabel("Época")
        self.canvas_treino.eixo.set_ylabel("Loss")
        self.canvas_treino.eixo.set_title("Curva de Aprendizado (ao vivo)")
        self.canvas_treino.eixo.legend()
        self.canvas_treino.draw()

    def _ao_finalizar_treinamento(self, resultado):
        self.botao_treinar.setEnabled(True)
        self.modelo_treinado = resultado["modelo"]
        self.ultimo_resultado = resultado
        self.rotulo_epoca_atual.setText("✅ Treinamento concluído! Veja a aba '3. Resultados'.")

        self._exibir_resultados(resultado)
        QMessageBox.information(self, "Treinamento concluído",
                                 "O modelo foi treinado com sucesso! Confira a aba '3. Resultados'.")
        self.abas.setCurrentIndex(3)

    def _ao_ocorrer_erro_treino(self, mensagem):
        self.botao_treinar.setEnabled(True)
        self.rotulo_epoca_atual.setText("❌ Erro durante o treinamento.")
        QMessageBox.critical(self, "Erro no treinamento", mensagem)

    # ==========================================================================
    # ABA 3 - RESULTADOS (métricas, gráficos, nível de satisfação)
    # ==========================================================================
    def _criar_aba_resultados(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.rotulo_satisfacao = QLabel("Ainda não há um modelo treinado.")
        self.rotulo_satisfacao.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fonte = QFont()
        fonte.setPointSize(16)
        fonte.setBold(True)
        self.rotulo_satisfacao.setFont(fonte)
        self.rotulo_satisfacao.setFrameShape(QFrame.Shape.Box)
        self.rotulo_satisfacao.setMinimumHeight(60)
        layout.addWidget(self.rotulo_satisfacao)

        self.texto_metricas = QTextEdit()
        self.texto_metricas.setReadOnly(True)
        self.texto_metricas.setMaximumHeight(160)
        layout.addWidget(self.texto_metricas)

        self.canvas_resultado = CanvasGrafico(largura=6, altura=4)
        layout.addWidget(self.canvas_resultado, stretch=1)

        self.botao_salvar_modelo = QPushButton("💾 Salvar Modelo Treinado")
        self.botao_salvar_modelo.clicked.connect(self._acao_salvar_modelo)
        self.botao_salvar_modelo.setEnabled(False)
        layout.addWidget(self.botao_salvar_modelo)

        return widget

    def _exibir_resultados(self, resultado):
        metricas = resultado["metricas"]
        nivel, cor = calcular_satisfacao(self.tarefa, metricas)

        self.rotulo_satisfacao.setText(f"Nível de Satisfação do Modelo: {nivel}")
        self.rotulo_satisfacao.setStyleSheet(
            f"background-color: {cor}; color: white; border-radius: 6px; padding: 8px;"
        )

        # --- Texto de métricas ---------------------------------------------------
        if self.tarefa == "regressao":
            texto = (
                f"<b>Métricas de Regressão (conjunto de teste):</b><br>"
                f"MAE (Erro Absoluto Médio): {metricas['mae']:.4f}<br>"
                f"MSE (Erro Quadrático Médio): {metricas['mse']:.4f}<br>"
                f"RMSE (Raiz do Erro Quadrático Médio): {metricas['rmse']:.4f}<br>"
                f"R² (Coeficiente de Determinação): {metricas['r2']:.4f}<br>"
                f"<i>R² mede o quanto o modelo explica a variação dos dados "
                f"(1.0 = perfeito, 0.0 = tão bom quanto chutar a média).</i>"
            )
        else:
            texto = (
                f"<b>Métricas de Classificação (conjunto de teste):</b><br>"
                f"Acurácia: {metricas['accuracy']:.4f}<br>"
                f"Precisão (Precision): {metricas['precision']:.4f}<br>"
                f"Revocação (Recall): {metricas['recall']:.4f}<br>"
                f"F1-Score: {metricas['f1']:.4f}<br>"
                f"<i>F1-Score equilibra Precisão e Recall (bom para dados desbalanceados).</i>"
            )
        self.texto_metricas.setHtml(texto)

        # --- Gráfico -----------------------------------------------------------
        self.canvas_resultado.eixo.clear()
        if self.tarefa == "regressao":
            y_test = resultado["y_test"]
            y_pred = resultado["y_pred"]
            self.canvas_resultado.eixo.scatter(y_test, y_pred, alpha=0.5)
            limite_min = min(np.min(y_test), np.min(y_pred))
            limite_max = max(np.max(y_test), np.max(y_pred))
            self.canvas_resultado.eixo.plot([limite_min, limite_max], [limite_min, limite_max],
                                             color="red", linestyle="--", label="Previsão perfeita")
            self.canvas_resultado.eixo.set_xlabel("Valor Real")
            self.canvas_resultado.eixo.set_ylabel("Valor Previsto")
            self.canvas_resultado.eixo.set_title("Previsto x Real")
            self.canvas_resultado.eixo.legend()
        else:
            matriz = metricas["matriz_confusao"]
            self.canvas_resultado.eixo.imshow(matriz, cmap="Blues")
            self.canvas_resultado.eixo.set_xlabel("Classe Prevista")
            self.canvas_resultado.eixo.set_ylabel("Classe Real")
            self.canvas_resultado.eixo.set_title("Matriz de Confusão")
            for i in range(matriz.shape[0]):
                for j in range(matriz.shape[1]):
                    self.canvas_resultado.eixo.text(j, i, str(matriz[i, j]),
                                                      ha="center", va="center", color="black")
        self.canvas_resultado.draw()

        self.botao_salvar_modelo.setEnabled(True)

    def _acao_salvar_modelo(self):
        """
        Salva o modelo treinado (.keras) e um arquivo auxiliar (.pkl) contendo
        tudo o que é necessário para pré-processar novos dados exatamente da
        mesma forma que os dados de treino (scaler, colunas dummies, encoder, etc.)
        """
        caminho, _ = QFileDialog.getSaveFileName(
            self, "Salvar Modelo", "modelo_treinado.keras", "Modelo Keras (*.keras)"
        )
        if not caminho:
            return

        try:
            self.modelo_treinado.save(caminho)

            caminho_pre = caminho.replace(".keras", "_preprocessamento.pkl")
            pacote_pre = {
                "scaler": self.scaler,
                "colunas_dummies": self.colunas_dummies,
                "colunas_feature": self.colunas_feature,
                "coluna_rotulo": self.coluna_rotulo,
                "tarefa": self.tarefa,
                "label_encoder": self.label_encoder,
                "num_classes": self.num_classes,
                "data_treinamento": str(datetime.now()),
            }
            with open(caminho_pre, "wb") as f:
                pickle.dump(pacote_pre, f)

            QMessageBox.information(
                self, "Modelo salvo",
                f"Modelo salvo em:\n{caminho}\n\nArquivo de pré-processamento salvo em:\n{caminho_pre}\n\n"
                f"Guarde os dois arquivos juntos — ambos são necessários para usar a IA depois."
            )
        except Exception as e:
            QMessageBox.critical(self, "Erro ao salvar", str(e))

    # ==========================================================================
    # ABA 4 - USAR IA (carregar modelo salvo ou usar o recém-treinado + prever)
    # ==========================================================================
    def _criar_aba_usar_ia(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        grupo_modelo = QGroupBox("Modelo")
        layout_modelo = QHBoxLayout(grupo_modelo)
        self.botao_carregar_modelo = QPushButton("📂 Carregar Modelo Salvo (.keras)")
        self.botao_carregar_modelo.clicked.connect(self._acao_carregar_modelo)
        self.rotulo_modelo_ativo = QLabel("Nenhum modelo carregado (use o modelo recém-treinado ou carregue um).")
        self.rotulo_modelo_ativo.setWordWrap(True)
        layout_modelo.addWidget(self.botao_carregar_modelo)
        layout_modelo.addWidget(self.rotulo_modelo_ativo, stretch=1)
        layout.addWidget(grupo_modelo)

        grupo_dados_novos = QGroupBox("Novos Dados para Previsão")
        layout_dados_novos = QHBoxLayout(grupo_dados_novos)
        self.botao_carregar_dados_previsao = QPushButton("📂 Selecionar Arquivo(s) para Prever")
        self.botao_carregar_dados_previsao.clicked.connect(self._acao_carregar_dados_previsao)
        self.rotulo_dados_previsao = QLabel("Nenhum arquivo selecionado.")
        self.rotulo_dados_previsao.setWordWrap(True)
        layout_dados_novos.addWidget(self.botao_carregar_dados_previsao)
        layout_dados_novos.addWidget(self.rotulo_dados_previsao, stretch=1)
        layout.addWidget(grupo_dados_novos)

        self.botao_prever = QPushButton("🔮 Prever")
        self.botao_prever.clicked.connect(self._acao_prever)
        layout.addWidget(self.botao_prever)

        self.tabela_previsoes = QTableWidget()
        self.tabela_previsoes.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.tabela_previsoes, stretch=1)

        self.botao_exportar_previsoes = QPushButton("⬇️ Exportar Previsões (CSV)")
        self.botao_exportar_previsoes.clicked.connect(self._acao_exportar_previsoes)
        self.botao_exportar_previsoes.setEnabled(False)
        layout.addWidget(self.botao_exportar_previsoes)

        self.modelo_para_uso = None
        self.pacote_preprocessamento_uso = None
        self.df_para_previsao = None
        self.df_resultado_previsao = None

        return widget

    def _acao_carregar_modelo(self):
        """Permite carregar um modelo .keras salvo anteriormente (junto do .pkl)."""
        caminho, _ = QFileDialog.getOpenFileName(
            self, "Selecione o arquivo do modelo (.keras)", "", "Modelo Keras (*.keras)"
        )
        if not caminho:
            return

        caminho_pre = caminho.replace(".keras", "_preprocessamento.pkl")
        if not os.path.exists(caminho_pre):
            QMessageBox.critical(
                self, "Arquivo de pré-processamento não encontrado",
                f"Não encontrei o arquivo:\n{caminho_pre}\n\n"
                f"Esse arquivo é criado junto com o modelo ao salvar na aba '3. Resultados' "
                f"e é necessário para preparar os novos dados corretamente."
            )
            return

        try:
            self.modelo_para_uso = keras.models.load_model(caminho)
            with open(caminho_pre, "rb") as f:
                self.pacote_preprocessamento_uso = pickle.load(f)
            self.rotulo_modelo_ativo.setText(f"✅ Modelo carregado: {os.path.basename(caminho)}")
        except Exception as e:
            QMessageBox.critical(self, "Erro ao carregar modelo", str(e))

    def _acao_carregar_dados_previsao(self):
        caminhos, _ = QFileDialog.getOpenFileNames(
            self, "Selecione arquivo(s) para prever", "",
            "Arquivos de Dados (*.csv *.xlsx *.xls);;Todos os Arquivos (*)"
        )
        if not caminhos:
            return
        try:
            self.df_para_previsao = carregar_arquivos(caminhos)
            self.rotulo_dados_previsao.setText(
                f"{len(caminhos)} arquivo(s) carregado(s) | {len(self.df_para_previsao)} linhas"
            )
        except Exception as e:
            QMessageBox.critical(self, "Erro ao ler arquivo(s)", str(e))

    def _acao_prever(self):
        """
        Usa o modelo (recém-treinado nesta sessão OU carregado do disco) para
        gerar previsões sobre um novo conjunto de dados, aplicando EXATAMENTE
        o mesmo pré-processamento usado no treino (mesmas colunas dummies e
        mesmo scaler), garantindo consistência.
        """
        # Decide qual modelo/pré-processamento usar: o carregado manualmente
        # tem prioridade; senão, usa o modelo treinado na sessão atual.
        if self.modelo_para_uso is not None:
            modelo = self.modelo_para_uso
            scaler = self.pacote_preprocessamento_uso["scaler"]
            colunas_dummies = self.pacote_preprocessamento_uso["colunas_dummies"]
            colunas_feature = self.pacote_preprocessamento_uso["colunas_feature"]
            tarefa = self.pacote_preprocessamento_uso["tarefa"]
            label_encoder = self.pacote_preprocessamento_uso["label_encoder"]
        elif self.modelo_treinado is not None:
            modelo = self.modelo_treinado
            scaler = self.scaler
            colunas_dummies = self.colunas_dummies
            colunas_feature = self.colunas_feature
            tarefa = self.tarefa
            label_encoder = self.label_encoder
        else:
            QMessageBox.warning(self, "Nenhum modelo disponível",
                                 "Treine um modelo na aba '2. Treinamento' ou carregue um modelo salvo.")
            return

        if self.df_para_previsao is None:
            QMessageBox.warning(self, "Nenhum dado carregado",
                                 "Selecione um arquivo com novos dados para prever.")
            return

        try:
            colunas_faltando = [c for c in colunas_feature if c not in self.df_para_previsao.columns]
            if colunas_faltando:
                raise ValueError(
                    f"O arquivo selecionado não possui as colunas necessárias: {colunas_faltando}"
                )

            X_novo = self.df_para_previsao[colunas_feature]
            X_novo_dummies = pd.get_dummies(X_novo, drop_first=False)

            # Alinha as colunas do novo arquivo com as colunas vistas no treino:
            # colunas que não apareceram agora são preenchidas com 0,
            # colunas extras (categorias novas) são descartadas.
            X_novo_alinhado = X_novo_dummies.reindex(columns=colunas_dummies, fill_value=0)

            X_novo_normalizado = scaler.transform(X_novo_alinhado.astype(float))
            previsoes_bruta = modelo.predict(X_novo_normalizado, verbose=0)

            df_resultado = self.df_para_previsao.copy()
            if tarefa == "regressao":
                df_resultado["Previsão"] = previsoes_bruta.flatten()
            else:
                if previsoes_bruta.shape[1] == 1:  # classificação binária (sigmoid)
                    classes_idx = (previsoes_bruta.flatten() > 0.5).astype(int)
                    confianca = np.where(classes_idx == 1, previsoes_bruta.flatten(), 1 - previsoes_bruta.flatten())
                else:  # multi-classe (softmax)
                    classes_idx = np.argmax(previsoes_bruta, axis=1)
                    confianca = np.max(previsoes_bruta, axis=1)

                if label_encoder is not None:
                    df_resultado["Previsão"] = label_encoder.inverse_transform(classes_idx)
                else:
                    df_resultado["Previsão"] = classes_idx
                df_resultado["Confiança"] = confianca

            self.df_resultado_previsao = df_resultado
            self._preencher_tabela_previsoes(df_resultado)
            self.botao_exportar_previsoes.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Erro ao prever", str(e))

    def _preencher_tabela_previsoes(self, df):
        df_amostra = df.head(200)  # limita a exibição para não travar a interface
        self.tabela_previsoes.setRowCount(len(df_amostra))
        self.tabela_previsoes.setColumnCount(len(df_amostra.columns))
        self.tabela_previsoes.setHorizontalHeaderLabels(list(df_amostra.columns))
        for i in range(len(df_amostra)):
            for j, col in enumerate(df_amostra.columns):
                self.tabela_previsoes.setItem(i, j, QTableWidgetItem(str(df_amostra.iloc[i, j])))

    def _acao_exportar_previsoes(self):
        if self.df_resultado_previsao is None:
            return
        caminho, _ = QFileDialog.getSaveFileName(
            self, "Exportar previsões", "previsoes.csv", "CSV (*.csv)"
        )
        if not caminho:
            return
        self.df_resultado_previsao.to_csv(caminho, index=False, encoding="utf-8-sig")
        QMessageBox.information(self, "Exportado", f"Previsões exportadas para:\n{caminho}")


# ==================================================================================
# PONTO DE ENTRADA DO PROGRAMA
# ==================================================================================

def main():
    app = QApplication(sys.argv)
    janela = JanelaPrincipal()
    janela.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


# ==================================================================================
# NOTAS PARA EXECUÇÃO NO GOOGLE COLAB
# ==================================================================================
# O Colab não tem tela gráfica, então uma janela PyQt6 não abre nele diretamente.
# Caminhos possíveis para quem quiser adaptar este código para o Colab:
#
# 1) EXTRAIR A LÓGICA (recomendado):
#    As funções `carregar_arquivos`, `detectar_tipo_tarefa`, `calcular_satisfacao`
#    e a lógica de pré-processamento/treinamento dentro de `ThreadTreinamento.run`
#    não dependem do PyQt6. Elas podem ser copiadas para um notebook e usadas com
#    `ipywidgets` (FileUpload, Checkbox, Button) para recriar a mesma experiência
#    dentro do navegador do Colab.
#
# 2) TELA VIRTUAL (mais avançado):
#    !apt-get install -y xvfb
#    !pip install pyvirtualdisplay
#    from pyvirtualdisplay import Display
#    display = Display(visible=0, size=(1200, 800))
#    display.start()
#    # A janela roda, mas você precisará de um servidor VNC/noVNC para "ver" a tela.
#
# Para uso normal no PyCharm, basta instalar as dependências do requirements.txt
# e rodar:  python app.py
# ==================================================================================
