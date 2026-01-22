# -*- coding: utf-8 -*-
"""Projeto de Portfólio: Análise de Dados Integrada para E-Commerce - Streamlit App"""

# --- Bloco 1: Importação de Bibliotecas e Configuração da Página ---

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from datetime import datetime, date
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import unicodedata
import sqlite3
import os

# Configurações do pandas e matplotlib
pd.set_option('display.float_format', lambda x: '%.4f' % x)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Configuração Inicial da Aplicação Streamlit
st.set_page_config(
    page_title="E-Commerce Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --- Bloco 2: Funções do Projeto Colab (Adaptadas) ---

def dsa_init_db(conn):
    """
    Inicializa o banco de dados a partir do CSV limpo.
    """
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tb_ecommerce (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Data_Compra TEXT,
            Cliente_ID TEXT,
            Categoria TEXT,
            Produto TEXT,
            Quantidade INTEGER,
            Preco_Unitario REAL,
            Fatura_Mensal REAL,
            Fidelidade_Meses INTEGER,
            Tipo_Contrato TEXT,
            Servico_Entrega TEXT,
            Status_Entrega TEXT,
            Review_Texto TEXT,
            Sentimento TEXT,
            Churn TEXT
        )
    """)
    
    conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM tb_ecommerce")
    
    if cursor.fetchone()[0] == 0:
        try:
            df_limpo = pd.read_csv("dataset_ecommerce_realista_limpo.csv")
            
            if 'Data_Compra' in df_limpo.columns:
                df_limpo['Data_Compra'] = pd.to_datetime(df_limpo['Data_Compra'], errors='coerce')
            
            rows = []
            for _, row in df_limpo.iterrows():
                rows.append((
                    row['Data_Compra'].isoformat() if pd.notna(row.get('Data_Compra')) else None,
                    row.get('Cliente_ID'),
                    row.get('Categoria'),
                    row.get('Produto'),
                    int(row.get('Quantidade', 0)),
                    float(row.get('Preco_Unitario', 0)),
                    float(row.get('Fatura_Mensal', 0)),
                    int(row.get('Fidelidade_Meses', 0)),
                    row.get('Tipo_Contrato'),
                    row.get('Servico_Entrega'),
                    row.get('Status_Entrega'),
                    row.get('Review_Texto'),
                    row.get('Sentimento'),
                    row.get('Churn')
                ))
            
            cursor.executemany(
                """INSERT INTO tb_ecommerce 
                (Data_Compra, Cliente_ID, Categoria, Produto, Quantidade, Preco_Unitario, 
                 Fatura_Mensal, Fidelidade_Meses, Tipo_Contrato, Servico_Entrega, 
                 Status_Entrega, Review_Texto, Sentimento, Churn) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            
            conn.commit()
            st.sidebar.success("✅ Banco de dados populado com dados limpos!")
            
        except Exception as e:
            st.sidebar.warning(f"⚠️ Erro ao carregar CSV: {e}")


def dsa_cria_conexao(db_path="ecommerce_database.db"):
    """
    Cria e retorna uma conexão com o banco de dados SQLite.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return conn


@st.cache_data(ttl=600)
def dsa_carrega_dados():
    """
    Função principal para carregar os dados do banco SQLite.
    """
    # Verifica se o arquivo CSV limpo existe
    csv_path = "dataset_ecommerce_realista_limpo.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, parse_dates=['Data_Compra'])
            st.sidebar.success(f"✅ Dados carregados do CSV limpo! Dimensões: {df.shape}")
            return df
        except Exception as e:
            st.sidebar.error(f"❌ Erro ao carregar CSV: {e}")
    
    # Fallback para banco de dados
    try:
        conn = dsa_cria_conexao()
        dsa_init_db(conn)
        df = pd.read_sql_query("SELECT * FROM tb_ecommerce", conn, parse_dates=["Data_Compra"])
        conn.close()
        st.sidebar.success(f"✅ Dados carregados do banco! Dimensões: {df.shape}")
        return df
    except Exception as e:
        st.error(f"❌ Erro crítico: {e}")
        return pd.DataFrame()


@st.cache_data
def engenharia_atributos(df):
    """
    Cria novas features para análise.
    """
    if df.empty:
        return df
    
    df_features = df.copy()
    
    # Cria coluna Total_Venda
    if all(col in df_features.columns for col in ['Quantidade', 'Preco_Unitario']):
        df_features['Total_Venda'] = df_features['Quantidade'] * df_features['Preco_Unitario']
    
    # Extrai mês e ano da compra
    if 'Data_Compra' in df_features.columns:
        df_features['Mes_Compra'] = df_features['Data_Compra'].dt.month
        df_features['Ano_Compra'] = df_features['Data_Compra'].dt.year
    
    return df_features


def analise_exploratoria(df):
    """
    Realiza análises exploratórias e gera visualizações.
    """
    if df.empty:
        st.warning("⚠️ Não há dados para análise exploratória.")
        return
    
    # Cálculos de métricas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'Total_Venda' in df.columns:
            receita_total = df['Total_Venda'].sum()
            st.metric("Receita Total", f"R$ {receita_total:,.2f}")
    
    with col2:
        if 'Cliente_ID' in df.columns:
            num_clientes = df['Cliente_ID'].nunique()
            st.metric("Clientes Únicos", num_clientes)
    
    with col3:
        if 'Produto' in df.columns:
            num_produtos = df['Produto'].nunique()
            st.metric("Produtos Únicos", num_produtos)
    
    with col4:
        if 'Categoria' in df.columns:
            num_categorias = df['Categoria'].nunique()
            st.metric("Categorias", num_categorias)
    
    st.markdown("---")
    
    # Gráfico 1: Receita por Categoria (Plotly - mesmo estilo do molde)
    if 'Categoria' in df.columns and 'Total_Venda' in df.columns:
        receita_categoria = df.groupby('Categoria')['Total_Venda'].sum().reset_index()
        
        fig1 = px.bar(
            receita_categoria, 
            x='Categoria', 
            y='Total_Venda',
            title='Receita Total por Categoria',
            color='Categoria',
            template='plotly_dark',
            height=400
        )
        fig1.update_layout(
            xaxis_title='Categoria',
            yaxis_title='Receita (R$)',
            showlegend=False
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    # Gráfico 2: Tendência de Vendas ao Longo do Tempo
    if 'Data_Compra' in df.columns and 'Total_Venda' in df.columns:
        vendas_por_dia = df.set_index('Data_Compra').resample('D')['Total_Venda'].sum().reset_index()
        
        fig2 = px.line(
            vendas_por_dia,
            x='Data_Compra',
            y='Total_Venda',
            title='Tendência de Vendas Diárias',
            template='plotly_dark',
            height=400
        )
        fig2.update_traces(line=dict(color='#00CC96', width=3))
        fig2.update_layout(
            xaxis_title='Data',
            yaxis_title='Receita (R$)'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Gráfico 3: Quantidade Vendida por Produto (Top 10)
    if 'Produto' in df.columns and 'Quantidade' in df.columns:
        produto_mais_vendido = df.groupby('Produto')['Quantidade'].sum().sort_values(ascending=False).head(10)
        
        # Usando matplotlib com estilo escuro para combinar com o tema
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        fig3.patch.set_facecolor('#0E1117')
        ax3.set_facecolor('#0E1117')
        
        colors = plt.cm.Set3(np.linspace(0, 1, 10))
        produto_mais_vendido.plot(kind='barh', color=colors, ax=ax3)
        
        ax3.set_title('Top 10 Produtos Mais Vendidos (Quantidade)', color='white')
        ax3.set_xlabel('Quantidade Vendida', color='white')
        ax3.set_ylabel('Produto', color='white')
        ax3.tick_params(axis='x', colors='white')
        ax3.tick_params(axis='y', colors='white')
        ax3.invert_yaxis()
        
        # Configurar cores das bordas
        for spine in ax3.spines.values():
            spine.set_color('white')
        
        st.pyplot(fig3)
    
    # Gráfico 4: Distribuição do Status de Entrega
    if 'Status_Entrega' in df.columns:
        status_counts = df['Status_Entrega'].value_counts()
        
        fig4 = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title='Distribuição do Status de Entrega',
            height=400,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    # Gráfico 5: Taxa de Churn Geral - CORRIGIDO tamanho
    if 'Churn' in df.columns:
        churn_counts = df['Churn'].value_counts()
        
        # Usando Plotly para manter consistência de tamanho
        fig5 = px.pie(
            values=churn_counts.values,
            names=churn_counts.index,
            title='Taxa de Churn Geral',
            height=400,
            color_discrete_sequence=['#4CAF50', '#FF5252']  # Verde para Não, Vermelho para Sim
        )
        st.plotly_chart(fig5, use_container_width=True)


@st.cache_data
def prepara_dados_churn(df):
    """
    Prepara os dados para a modelagem de churn.
    """
    if df.empty or 'Cliente_ID' not in df.columns:
        return None, None, None
    
    # Agregação no nível do cliente
    cliente_data = df.groupby('Cliente_ID').agg({
        'Fidelidade_Meses': 'first',
        'Tipo_Contrato': 'first',
        'Servico_Entrega': 'first',
        'Fatura_Mensal': 'first',
        'Churn': 'first',
        'Total_Venda': 'sum'
    }).reset_index()
    
    # Codifica churn como numérico
    cliente_data['Churn_Num'] = cliente_data['Churn'].map({'Sim': 1, 'Não': 0})
    
    # Cria variáveis dummy
    modelo_data = pd.get_dummies(cliente_data,
                                 columns=['Tipo_Contrato', 'Servico_Entrega'],
                                 drop_first=True)
    
    # Define variáveis independentes e dependente
    features_to_drop = ['Cliente_ID', 'Churn', 'Churn_Num']
    X_churn = modelo_data.drop(columns=features_to_drop, errors='ignore')
    
    # Filtra apenas colunas numéricas
    X_churn = X_churn.select_dtypes(include=np.number)
    
    # Remove NaNs e infinitos
    X_churn = X_churn.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Alinha y_churn com X_churn
    y_churn = modelo_data.loc[X_churn.index, 'Churn_Num']
    
    # Adiciona constante
    X_churn = sm.add_constant(X_churn, has_constant='add')
    
    return X_churn, y_churn, modelo_data


def modelo_churn(df):
    """
    Executa a regressão logística para análise de churn.
    """
    if df.empty:
        st.warning("⚠️ Não há dados para modelagem de churn.")
        return
    
    with st.spinner("Preparando dados e treinando modelo..."):
        X_churn, y_churn, modelo_data = prepara_dados_churn(df)
        
        if X_churn is None or y_churn is None:
            st.error("❌ Não foi possível preparar dados para o modelo de churn.")
            return
        
        # Cria e treina o modelo
        modelo_churn = sm.Logit(y_churn, X_churn)
        resultado_churn = modelo_churn.fit(disp=False)
    
    # Exibe resultados
    st.markdown("### Resultados do Modelo de Regressão Logística")
    
    # Cria uma tabela com os coeficientes
    summary_df = pd.DataFrame({
        'Coeficiente': resultado_churn.params,
        'Erro Padrão': resultado_churn.bse,
        'z': resultado_churn.tvalues,
        'P>|z|': resultado_churn.pvalues,
        '[0.025': resultado_churn.conf_int()[0],
        '0.975]': resultado_churn.conf_int()[1]
    })
    
    # Formata a tabela
    st.dataframe(summary_df.style.format({
        'Coeficiente': '{:.4f}',
        'Erro Padrão': '{:.4f}',
        'z': '{:.4f}',
        'P>|z|': '{:.4f}',
        '[0.025': '{:.4f}',
        '0.975]': '{:.4f}'
    }))
    
    # Calcula Odds Ratio
    st.markdown("### Tabela de Razão de Chances (Odds Ratio)")
    params = resultado_churn.params
    conf = resultado_churn.conf_int()
    conf['Odds Ratio'] = params
    conf.columns = ['2.5%', '97.5%', 'Odds Ratio']
    conf = np.exp(conf)
    
    st.dataframe(conf.style.format('{:.4f}'))
    
    # Gráfico: Churn por Tipo de Contrato
    if all(col in df.columns for col in ['Tipo_Contrato', 'Churn']):
        fig = px.histogram(df, x='Tipo_Contrato', color='Churn',
                          barmode='group', title='Churn por Tipo de Contrato',
                          color_discrete_map={'Não': '#00CC96', 'Sim': '#FF5252'},
                          height=400)
        fig.update_layout(xaxis_title='Tipo de Contrato', yaxis_title='Número de Clientes')
        st.plotly_chart(fig, use_container_width=True)


def limpa_texto(texto):
    """
    Limpa e normaliza texto para análise de sentimentos.
    """
    if not isinstance(texto, str):
        return ""
    # Remove acentos
    texto = ''.join(c for c in unicodedata.normalize('NFKD', texto)
                    if unicodedata.category(c) != 'Mn')
    # Converte para minúsculas e remove caracteres especiais
    texto = re.sub(r'[^a-zA-Z\s]', '', texto.lower())
    # Remove espaços extras
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto


@st.cache_data
def prepara_dados_sentimentos(df):
    """
    Prepara os dados para a modelagem de sentimentos.
    """
    if df.empty or 'Review_Texto' not in df.columns or 'Sentimento' not in df.columns:
        return None, None, None, None
    
    # Aplica limpeza aos reviews - SOMENTE COM DADOS DO CSV
    df_sent = df.copy()
    df_sent = df_sent.dropna(subset=['Review_Texto', 'Sentimento'])
    df_sent['Review_Limpo'] = df_sent['Review_Texto'].apply(limpa_texto)
    
    # Codifica sentimentos como numéricos - SOMENTE COM DADOS DO CSV
    df_sent['Sentimento_Num'] = df_sent['Sentimento'].map({'positivo': 1, 'negativo': 0})
    
    # Remove linhas com sentimentos não mapeados
    df_sent = df_sent.dropna(subset=['Sentimento_Num'])
    
    # Prepara dados para modelo de sentimentos
    X_sent = df_sent['Review_Limpo']
    y_sent = df_sent['Sentimento_Num']
    
    # Divisão treino-teste - SOMENTE SE HOUVER DADOS SUFICIENTES
    if len(X_sent) > 10 and len(y_sent.unique()) == 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X_sent, y_sent, test_size=0.25, random_state=42, stratify=y_sent)
        return X_train, X_test, y_train, y_test
    else:
        return None, None, None, None


@st.cache_resource
def treina_modelo_sentimentos(X_train, y_train):
    """
    Treina o modelo de classificação de sentimentos.
    """
    if X_train is None or y_train is None or len(X_train) == 0:
        return None
    
    # Cria pipeline simplificado
    pipeline_sent = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=500, stop_words=['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um'])),
        ('logreg', LogisticRegression(solver='liblinear', random_state=42, max_iter=1000))
    ])
    
    # Treina o modelo - SOMENTE COM DADOS DO CSV
    try:
        pipeline_sent.fit(X_train, y_train)
        return pipeline_sent
    except:
        return None


def modelo_sentimentos(df):
    """
    Executa a análise de sentimentos e avalia o modelo.
    """
    if df.empty:
        st.warning("⚠️ Não há dados para modelagem de sentimentos.")
        return
    
    with st.spinner("Preparando dados e treinando modelo..."):
        # Prepara dados - SOMENTE COM DADOS DO CSV
        X_train, X_test, y_train, y_test = prepara_dados_sentimentos(df)
        
        if X_train is None:
            st.error("❌ Não há dados suficientes para treinar o modelo de sentimentos.")
            return
        
        # Treina modelo
        modelo = treina_modelo_sentimentos(X_train, y_train)
        
        if modelo is None:
            st.error("❌ Falha ao treinar o modelo de sentimentos.")
            return
    
    # Previsões no conjunto de teste
    y_pred = modelo.predict(X_test)
    
    # Calcula métricas de avaliação
    acuracia = accuracy_score(y_test, y_pred)
    
    # Exibe métricas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Acurácia do Modelo", f"{acuracia:.2%}")
    
    with col2:
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        if '1' in report_dict:
            st.metric("Precisão (Positivo)", f"{report_dict['1']['precision']:.2%}")
    
    with col3:
        if '1' in report_dict:
            st.metric("Recall (Positivo)", f"{report_dict['1']['recall']:.2%}")
    
    # Matriz de confusão - CORRIGIDO tamanho
    st.markdown("### Matriz de Confusão")
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Usando Plotly para matriz de confusão com tamanho consistente
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale='Blues',
        labels=dict(x="Previsão", y="Verdadeiro", color="Quantidade"),
        x=['Negativo', 'Positivo'],
        y=['Negativo', 'Positivo'],
        height=400  # Altura fixa igual aos outros gráficos
    )
    
    fig_cm.update_layout(
        title='Matriz de Confusão - Modelo de Sentimentos',
        xaxis_title='Previsão',
        yaxis_title='Verdadeiro'
    )
    
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Teste Interativo - RESTAURADO design anterior
    st.markdown("### Teste Interativo com Novos Reviews")
    
    # Recupera exemplos REAIS do CSV para demonstração
    exemplos_reais = []
    if 'Review_Texto' in df.columns and 'Sentimento' in df.columns:
        # Pega alguns exemplos reais do dataset
        df_exemplos = df.dropna(subset=['Review_Texto', 'Sentimento']).head(4)
        for _, row in df_exemplos.iterrows():
            exemplos_reais.append({
                'texto': row['Review_Texto'],
                'sentimento_real': row['Sentimento']
            })
    
    # Layout de duas colunas RESTAURADO
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Classifique um Review")
        novo_review = st.text_area(
            "Digite um review para classificação:",
            "Produto excelente, qualidade impecável!",
            height=100
        )
        
        if st.button("🔍 Classificar Sentimento", use_container_width=True):
            if novo_review:
                try:
                    sentimento_pred = modelo.predict([novo_review])[0]
                    sentimento_texto = "Positivo" if sentimento_pred == 1 else "Negativo"
                    
                    if sentimento_pred == 1:
                        st.success(f"✅ **Sentimento:** {sentimento_texto}")
                    else:
                        st.error(f"❌ **Sentimento:** {sentimento_texto}")
                except:
                    st.warning("⚠️ Não foi possível classificar este review.")
    
    with col_b:
        st.subheader("Exemplos do Dataset")
        
        if exemplos_reais:
            for i, exemplo in enumerate(exemplos_reais):
                with st.expander(f"Exemplo {i+1}: {exemplo['texto'][:50]}...", expanded=False):
                    st.write(f"**Review original:** {exemplo['texto']}")
                    st.write(f"**Sentimento real no dataset:** {exemplo['sentimento_real']}")
                    
                    # Faz predição para o exemplo real
                    try:
                        pred = modelo.predict([exemplo['texto']])[0]
                        pred_texto = "Positivo" if pred == 1 else "Negativo"
                        st.write(f"**Classificação do modelo:** {pred_texto}")
                        
                        # Verifica se acertou
                        real_num = 1 if exemplo['sentimento_real'] == 'positivo' else 0
                        if pred == real_num:
                            st.success("✅ Classificação correta!")
                        else:
                            st.error("❌ Classificação incorreta!")
                    except:
                        st.write("⚠️ Não foi possível classificar este exemplo")
        else:
            st.info("Nenhum exemplo disponível no dataset.")


# --- Bloco 3: Funções do Molde Streamlit ---

def dsa_filtros_sidebar(df):
    """
    Cria widgets de filtro na barra lateral.
    """
    # Banner da Sidebar
    st.sidebar.markdown(
        """
        <div style="background-color:#00CC96; padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 15px;">
            <h3 style="color:white; margin:0; font-weight:bold;">E-Commerce Analytics</h3>
            <p style="color:white; margin:5px 0 0 0; font-size:12px;">Projeto de Estudo - Cauan Cicone</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.sidebar.header("🔍 Filtros")
    
    # Filtro de Data
    if 'Data_Compra' in df.columns and not df.empty:
        min_date = df["Data_Compra"].min().date()
        max_date = df["Data_Compra"].max().date()
        
        date_range = st.sidebar.date_input(
            "Período de Análise", 
            (min_date, max_date), 
            min_value=min_date, 
            max_value=max_date
        )
    
    # Filtros de Categoria
    if 'Categoria' in df.columns and not df.empty:
        all_categorias = sorted(df["Categoria"].unique())
        selected_categorias = st.sidebar.multiselect(
            "Categorias", 
            all_categorias, 
            default=all_categorias
        )
    else:
        selected_categorias = []
    
    # Filtros de Produto
    if 'Produto' in df.columns and not df.empty:
        all_produtos = sorted(df["Produto"].unique())
        selected_produtos = st.sidebar.multiselect(
            "Produtos", 
            all_produtos, 
            default=all_produtos
        )
    else:
        selected_produtos = []
    
    # Filtro de Status de Entrega
    if 'Status_Entrega' in df.columns and not df.empty:
        all_status = sorted(df["Status_Entrega"].unique())
        selected_status = st.sidebar.multiselect(
            "Status de Entrega", 
            all_status, 
            default=all_status
        )
    else:
        selected_status = []
    
    # Filtro de Churn
    if 'Churn' in df.columns and not df.empty:
        all_churn = sorted(df["Churn"].unique())
        selected_churn = st.sidebar.multiselect(
            "Status do Cliente", 
            all_churn, 
            default=all_churn
        )
    else:
        selected_churn = []
    
    # Aplica filtros
    df_filtrado = df.copy()
    
    # Filtro de Data
    if 'Data_Compra' in df.columns and 'date_range' in locals() and len(date_range) == 2:
        start_date, end_date = date_range
        df_filtrado = df_filtrado[
            (df_filtrado["Data_Compra"].dt.date >= start_date) &
            (df_filtrado["Data_Compra"].dt.date <= end_date)
        ]
    
    # Filtros de seleção múltipla
    if selected_categorias:
        df_filtrado = df_filtrado[df_filtrado["Categoria"].isin(selected_categorias)]
    
    if selected_produtos:
        df_filtrado = df_filtrado[df_filtrado["Produto"].isin(selected_produtos)]
    
    if selected_status:
        df_filtrado = df_filtrado[df_filtrado["Status_Entrega"].isin(selected_status)]
    
    if selected_churn:
        df_filtrado = df_filtrado[df_filtrado["Churn"].isin(selected_churn)]
    
    # Rodapé da Sidebar
    st.sidebar.markdown("---")
    
    with st.sidebar.expander("📚 Sobre Este Projeto", expanded=False):
        st.markdown("""
        ### 🎓 Projeto de Estudo - 
        
        **Autor:** Cauan Cicone  
        **Finalidade:** Exclusivamente educacional  
        **Dados:** Criados e totalmente ficticios  
        
        
        **Créditos:**  
        Baseado no molde aprendido no curso de Python da **Data Science Academy**.
        """)
    
    st.sidebar.caption("Dashboard desenvolvido para fins educacionais - Dados fictícios")
    
    return df_filtrado


def dsa_renderiza_cards_kpis(df):
    """
    Calcula e exibe KPIs principais.
    """
    if df.empty:
        st.warning("⚠️ Não há dados para calcular KPIs.")
        return 0, 0, 0, 0
    
    # Cálculos dos KPIs - SOMENTE COM DADOS DO CSV
    if 'Total_Venda' in df.columns:
        total_faturamento = df["Total_Venda"].sum()
    else:
        total_faturamento = 0
    
    if 'Quantidade' in df.columns:
        total_qty = df["Quantidade"].sum()
    else:
        total_qty = 0
    
    if 'Cliente_ID' in df.columns:
        clientes_unicos = df["Cliente_ID"].nunique()
    else:
        clientes_unicos = 0
    
    if 'Total_Venda' in df.columns and 'Quantidade' in df.columns and total_qty > 0:
        avg_ticket = total_faturamento / total_qty
    else:
        avg_ticket = 0
    
    # Criação do Layout
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Receita Total</h3>
            <h2>R$ {total_faturamento:,.0f}</h2>
            <div class="delta">Faturamento consolidado</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Unidades Vendidas</h3>
            <h2>{total_qty:,.0f}</h2>
            <div class="delta">Volume total</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Clientes Únicos</h3>
            <h2>{clientes_unicos:,.0f}</h2>
            <div class="delta">Base de clientes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Ticket Médio</h3>
            <h2>R$ {avg_ticket:,.2f}</h2>
            <div class="delta">Por transação</div>
        </div>
        """, unsafe_allow_html=True)
    
    return total_faturamento, total_qty, clientes_unicos, avg_ticket


def dsa_gera_pdf_report(df_filtrado, total_faturamento, total_quantidade, clientes_unicos):
    """
    Gera um relatório PDF com os resultados da análise.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Título
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Relatório Executivo de Análise de E-Commerce", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)
    
    # Data de geração
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    # Informação do projeto
    pdf.cell(0, 8, "Projeto de Estudo - Dados Fictícios - Autor: Cauan Cicone", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)
    
    # Bloco de KPIs
    pdf.set_fill_color(240, 240, 240)
    pdf.rect(10, 45, 190, 25, 'F')
    pdf.set_y(50)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(63, 8, f"Receita Total", align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(63, 8, f"Quantidade", align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(63, 8, f"Clientes Únicos", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(63, 8, f"R$ {total_faturamento:,.2f}", align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(63, 8, f"{total_quantidade:,}", align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(63, 8, f"{clientes_unicos:,}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.ln(15)
    
    # Top 10 Vendas
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Top 10 Vendas (por valor):", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    col_widths = [25, 40, 30, 40, 25, 30]
    headers = ["Data", "Cliente", "Categoria", "Produto", "Qtd", "Valor"]
    
    # Cabeçalho da tabela
    pdf.set_font("Helvetica", "B", 9)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, 1, align='C', new_x=XPos.RIGHT, new_y=YPos.TOP)
    
    pdf.ln()
    
    # Dados da tabela - SOMENTE COM DADOS DO CSV
    if 'Total_Venda' in df_filtrado.columns and not df_filtrado.empty:
        df_top = df_filtrado.sort_values("Total_Venda", ascending=False).head(10)
        
        pdf.set_font("Helvetica", "", 9)
        for _, row in df_top.iterrows():
            data = [
                str(row['Data_Compra'].date()) if 'Data_Compra' in df_filtrado.columns else "N/A",
                str(row['Cliente_ID'])[:8] + "..." if 'Cliente_ID' in df_filtrado.columns else "N/A",
                row['Categoria'][:15] if 'Categoria' in df_filtrado.columns else "N/A",
                row['Produto'][:15] if 'Produto' in df_filtrado.columns else "N/A",
                str(row['Quantidade']) if 'Quantidade' in df_filtrado.columns else "N/A",
                f"R$ {row['Total_Venda']:,.2f}" if 'Total_Venda' in df_filtrado.columns else "N/A"
            ]
            
            for i, d in enumerate(data):
                safe_txt = str(d).encode("latin-1", "replace").decode("latin-1")
                pdf.cell(col_widths[i], 7, safe_txt, 1, align=('C' if i==4 else 'L'), 
                        new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.ln()
    
    # Gera PDF
    result = pdf.output()
    return result.encode("latin-1") if isinstance(result, str) else bytes(result)


def dsa_set_custom_theme():
    """
    Define e injeta CSS customizado no app.
    """
    card_bg_color = "#262730"
    text_color = "#FAFAFA"
    gold_color = "#E1C16E"
    dark_text = "#1E1E1E"
    
    css = f"""
    <style>
        /* Aumentar Altura Mínima dos Filtros Multiselect */
        [data-testid="stMultiSelect"] div[data-baseweb="select"] > div:first-child {{
            min-height: 100px !important;
            overflow-y: auto !important;
        }}
    
        /* Estilização dos Cards de KPI */
        .metric-card {{
            background-color: {card_bg_color};
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #444;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
            text-align: center;
            margin-bottom: 10px;
        }}

        .metric-card h3 {{
            margin: 0;
            font-size: 1.2rem;
            color: #AAA;
            font-weight: normal;
        }}

        .metric-card h2 {{
            margin: 10px 0 0 0;
            font-size: 2rem;
            color: {text_color};
            font-weight: bold;
        }}

        .metric-card .delta {{
            font-size: 0.9rem;
            color: #4CAF50;
            margin-top: 5px;
        }}
                
        /* Estilização dos Itens de Filtro Selecionados */
        [data-baseweb="tag"] {{
            background-color: {gold_color} !important;
            color: {dark_text} !important;
            border-radius: 4px !important;
        }}
        
        [data-baseweb="tag"] svg {{
            color: {dark_text} !important;
        }}
        
        [data-baseweb="tag"] svg:hover {{
            color: #FF0000 !important; 
        }}
        
        /* Estilo para containers de expansão */
        .streamlit-expanderHeader {{
            background-color: #262730;
            border-radius: 5px;
            border: 1px solid #444;
        }}
        
        /* Tamanho consistente dos gráficos */
        .js-plotly-plot, .plotly {{
            height: 400px !important;
        }}
        
        /* Aviso de projeto fictício */
        .warning-box {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
            color: #856404;
        }}
        
        /* Melhorar contraste para gráficos matplotlib */
        .stPlot {{
            background-color: #0E1117;
        }}
        
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)


# --- Bloco 4: Função Principal ---

def main():
    """
    Função principal que orquestra todo o aplicativo.
    """
    # Configura tema customizado
    dsa_set_custom_theme()
    
    # Aviso de projeto fictício
    st.markdown("""
    <div class="warning-box">
        <h4>Projeto de Análise de Dados</h4>
        <p>Este dashboard foi desenvolvido <strong>exclusivamente para fins educacionais</strong> por <strong>Cauan Cicone</strong>.</p>
        <p>Todos os dados, análises e conclusões apresentados servem apenas para aprendizado.</p>
        <p><em>Baseado no molde aprendido no curso da Data Science Academy.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Títulos da página
    st.title("📊 E-Commerce Analytics Dashboard")
    st.subheader("Projeto de Portfólio: Análise de Dados Integrada para E-Commerce")
    st.markdown("""
    **Autor:** Cauan Cicone | **Finalidade:** Exclusivamente educacional | **Dados:** Completamente fictícios
    """)
    st.markdown("---")
    
    # Carrega dados automaticamente do CSV
    with st.spinner("Carregando dados do CSV limpo..."):
        df_raw = dsa_carrega_dados()
        
        if df_raw.empty:
            st.error("""
            ❌ **Erro ao carregar dados!**
            
            Por favor, certifique-se de que o arquivo `dataset_ecommerce_realista_limpo.csv` está na mesma pasta do aplicativo.
            
            O arquivo CSV deve conter as seguintes colunas (após limpeza):
            - Data_Compra, Cliente_ID, Categoria, Produto
            - Quantidade, Preco_Unitario, Fatura_Mensal, Fidelidade_Meses
            - Tipo_Contrato, Servico_Entrega, Status_Entrega
            - Review_Texto, Sentimento, Churn
            """)
            return
        
        # Engenharia de atributos
        df_completo = engenharia_atributos(df_raw)
    
    # Sidebar com filtros
    df_filtrado = dsa_filtros_sidebar(df_completo)
    
    # Verifica se há dados após filtragem
    if df_filtrado.empty:
        st.warning("⚠️ Nenhum dado encontrado com os filtros selecionados.")
        return
    
    # Cards de KPIs
    st.markdown("### 📈 Principais Métricas")
    total_faturamento, total_qty, clientes_unicos, avg_ticket = dsa_renderiza_cards_kpis(df_filtrado)
    
    # Linha divisória
    st.markdown("---")
    
    # Abas principais
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Definição do Problema", 
        "📊 Análise Exploratória", 
        "📈 Análise de Churn", 
        "😀 Análise de Sentimentos", 
        "💾 Dados & Exportação"
    ])
    
    # Aba 1: Definição do Problema de Negócio
    with tab1:
        st.markdown("""
        # 📋 Definição do Problema de Negócio
        
        ## 1.1. Contexto
        
        A **E-Commerce Plus** é uma empresa de varejo online fictícia em rápido crescimento, que opera em múltiplos segmentos (eletrônicos, moda, casa, etc.). A empresa coleta diariamente dados transacionais, informações de clientes e feedbacks (reviews) dos produtos vendidos.
        
        ## 1.2. Problema de Negócio
        
        A empresa não consegue extrair insights acionáveis de seus dados de forma integrada, o que resulta em:
        
        - **Decisões estratégicas** baseadas em dados inconsistentes
        - **Perda de clientes** sem ações de retenção proativas
        - **Feedback de clientes** não utilizado para melhorar produtos e serviços
        
        ## 1.3. Objetivos do Projeto
        
        1. **Limpeza e estruturação** dos dados brutos para garantir qualidade e confiabilidade
        2. **Identificação dos fatores** que influenciam o churn dos clientes
        3. **Automação da classificação** de sentimentos nos reviews de produtos
        
        ## 1.4. Critérios de Sucesso
        
        - ✅ Entrega de um dataset limpo e validado
        - ✅ Modelo estatístico identificando variáveis significativas para o churn
        - ✅ Modelo de classificação de sentimentos com boa acurácia
        - ✅ Relatório com visualizações claras e interpretações orientadas a negócio
        
        ---
        
       
        
        **Autor:** Cauan Cicone  
        **Baseado em:** Projeto do curso Data Science Academy
        """)
    
    # Aba 2: Análise Exploratória
    with tab2:
        analise_exploratoria(df_filtrado)
    
    # Aba 3: Análise de Churn
    with tab3:
        modelo_churn(df_filtrado)
    
    # Aba 4: Análise de Sentimentos
    with tab4:
        modelo_sentimentos(df_filtrado)
    
    # Aba 5: Dados e Exportação
    with tab5:
        st.subheader("📊 Visualização dos Dados Filtrados")
        
        # Mostrar DataFrame
        st.dataframe(df_filtrado, use_container_width=True, height=400)
        
        # Estatísticas resumidas
        with st.expander("📈 Estatísticas Descritivas"):
            if not df_filtrado.empty:
                st.write(df_filtrado.describe())
        
        # Área de exportação
        st.markdown("### 📥 Área de Exportação")
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            # Exportar CSV
            csv = df_filtrado.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Baixar CSV",
                data=csv,
                file_name="dados_ecommerce_filtrados.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_exp2:
            # Exportar PDF
            if st.button("📄 Gerar Relatório PDF", use_container_width=True):
                with st.spinner("Gerando relatório PDF..."):
                    pdf_bytes = dsa_gera_pdf_report(
                        df_filtrado, 
                        total_faturamento, 
                        total_qty, 
                        clientes_unicos
                    )
                    
                    st.download_button(
                        label="⬇️ Clique para Baixar PDF",
                        data=pdf_bytes,
                        file_name=f"Relatorio_Ecommerce_{date.today()}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="pdf-download"
                    )
    
    # Rodapé com Conclusões
    st.markdown("---")
    
    with st.expander("📋 Conclusões e Recomendações do Projeto", expanded=False):
        st.markdown("""
        # 📋 Conclusões e Recomendações
        
        ## Conclusões Técnicas
        
        1. **Qualidade dos Dados**: O dataset original continha problemas típicos (valores ausentes, outliers) que foram corrigidos na limpeza.
        
        2. **Modelo de Churn**: A regressão logística identificou variáveis significativas que influenciam o cancelamento.
        
        3. **Modelo de Sentimentos**: O pipeline TF-IDF + Regressão Logística alcançou acurácia satisfatória.
        
        4. **Visualizações**: Gráficos produzidos permitem compreensão clara dos padrões nos dados.
        
        ## Recomendações de Negócio (Fictícias)
        
        1. **Redução de Churn**:
           - Oferecer incentivos para migração de contratos mensais para anuais
           - Implementar programa de fidelização nos primeiros 6 meses
           - Revisar preços para clientes com faturas elevadas
        
        2. **Melhoria de Produtos**:
           - Analisar reviews negativos para identificar problemas recorrentes
           - Priorizar melhorias em categorias com maior volume de feedback negativo
        
        3. **Otimização Operacional**:
           - Monitorar continuamente a qualidade dos dados
           - Automatizar a classificação de novos reviews usando o modelo treinado
        
        ## Próximos Passos
        
        1. **Validação em Produção**: Testar os modelos com dados em tempo real
        2. **Monitoramento Contínuo**: Implementar dashboard para acompanhar métricas chave
        3. **Refinamento dos Modelos**: Coletar mais dados para melhorar a performance
        
        ---
        
        ### 🎓 Considerações Finais
        
        Este projeto demonstra habilidades completas em:
        
        - **Limpeza e preparação** de dados em um ambiente realista
        - **Análise exploratória** com visualizações profissionais
        - **Modelagem estatística** para identificação de fatores de negócio
        - **Machine Learning** para classificação de texto
        
        **Aviso Final:** Todo o conteúdo deste projeto é **FICTÍCIO** e foi desenvolvido **exclusivamente para fins educacionais** por **Cauan Cicone**.
        
        **Créditos:** Baseado no molde aprendido no curso gratuito de Python da **Data Science Academy**.
        """)
    
    # Créditos finais
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #262730; border-radius: 10px;">
        <h4 style="color: #FAFAFA; margin-bottom: 10px;">🎓 Projeto Educacional - Dados Fictícios</h4>
        <p style="color: #AAA; margin: 5px 0;">Desenvolvido por <strong>Cauan Cicone</strong> para fins de estudo</p>
        <p style="color: #AAA; margin: 5px 0;">Baseado no molde do curso Data Science Academy</p>
        <p style="color: #666; font-size: 12px; margin-top: 10px;"> Todos os dados e análises são fictícios e não representam informações reais</p>
    </div>
    """, unsafe_allow_html=True)


# --- Bloco 5: Ponto de Entrada ---

if __name__ == "__main__":
    main()