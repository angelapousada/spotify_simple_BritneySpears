import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from wordcloud import WordCloud
from textblob import TextBlob
from collections import Counter
import matplotlib.pyplot as plt
import nltk

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Britney Spears — Spotify & Análisis de letras",
    layout="wide",
    initial_sidebar_state="expanded",
)

AUDIO_FEATURES = [
    'acousticness', 'danceability', 'duration_ms', 'energy',
    'instrumentalness', 'liveness', 'loudness', 'speechiness',
    'tempo', 'valence'
]

RADAR_FEATURES = ['acousticness', 'danceability', 'energy',
                  'instrumentalness', 'liveness', 'valence']

STOPWORDS_EN = set(stopwords.words('english'))
STOPWORDS_EN.update([
    'oh', 'ya', 'yeah', 'ah', 'uh', 'la', 'na', 'da', 'ooh', 'hey',
    'gonna', 'got', 'get', 'let', 'like', 'know', 'go', 'come',
    'could', 'would', 'make', 'take', 'want', 'think', 'say', 'said',
    'one', 'two', 'also', 'back', 'even', 'still', 'well', 'way',
    'im', 'ive', 'dont', 'cant', 'wont', 'thats', 'youre', 'youve',
    'ill', 'theyre', 'weve', 'hes', 'shes', 'its', 'cause',
    'ee', 'mmm', 'mm', 'whoa', 'oooh', 'baby',
])

# ============================================================
# THEME — paleta editorial sobria
# ============================================================
PALETTE = [
    "#1F1B24",  # casi negro
    "#7A2E4A",  # vino
    "#B5838D",  # rosa empolvado
    "#C9A66B",  # oro envejecido
    "#6B4E71",  # ciruela
    "#8C8C8C",  # gris cálido
    "#3D2C2E",  # marrón oscuro
    "#D5BDAF",  # nude
    "#A4778B",  # malva
]

ACCENT = "#7A2E4A"
ACCENT_SOFT = "#B5838D"
BG = "#FAF7F2"
INK = "#1F1B24"

# Tema Plotly minimal-editorial
pio.templates["editorial"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, Helvetica, Arial, sans-serif", color=INK, size=13),
        title=dict(font=dict(family="Inter, Helvetica, Arial, sans-serif", size=20, color=INK)),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        colorway=PALETTE,
        xaxis=dict(showgrid=False, linecolor="#D9D2C7", ticks="outside", tickcolor="#D9D2C7"),
        yaxis=dict(showgrid=True, gridcolor="#ECE6DC", zeroline=False, linecolor="#D9D2C7"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
)
pio.templates.default = "editorial"

# CSS personalizado
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #1F1B24;
    }
    .stApp {
        background-color: #FAF7F2;
    }
    h1, h2, h3, h4 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        font-weight: 600 !important;
        color: #1F1B24 !important;
        letter-spacing: -0.01em;
    }
    h1 { font-size: 2.4rem !important; line-height: 1.15; font-weight: 700 !important; }
    h2 { font-size: 1.6rem !important; margin-top: 1.5rem; }
    h3 { font-size: 1.2rem !important; }

    .editorial-eyebrow {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        letter-spacing: 0.28em;
        text-transform: uppercase;
        color: #7A2E4A;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .editorial-rule {
        border: none;
        border-top: 1px solid #D9D2C7;
        margin: 1.6rem 0;
    }
    .editorial-lede {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 1rem;
        color: #4A4147;
        line-height: 1.6;
        max-width: 60rem;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #ECE4D6;
        border-radius: 4px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 1px 2px rgba(31, 27, 36, 0.04);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.7rem !important;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #8C7B6F !important;
        font-weight: 500 !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        font-size: 1.9rem !important;
        color: #1F1B24 !important;
        font-weight: 600 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1F1B24;
    }
    [data-testid="stSidebar"] * {
        color: #EDE6DA !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] a {
        color: #C9A66B !important;
        text-decoration: none;
        border-bottom: 1px dotted #C9A66B;
    }
    [data-testid="stSidebar"] hr {
        border-top: 1px solid #3D353F;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #D9D2C7;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        font-weight: 500;
        color: #8C7B6F;
        background: transparent;
        padding: 0.9rem 1.6rem;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #1F1B24 !important;
        border-bottom: 2px solid #7A2E4A !important;
        background: transparent !important;
    }

    /* Captions */
    .stCaption, [data-testid="stCaptionContainer"] {
        font-style: italic;
        color: #6B5E55 !important;
    }

    /* Footer */
    .editorial-footer {
        font-size: 0.78rem;
        color: #8C7B6F;
        text-align: center;
        letter-spacing: 0.06em;
        padding: 1.5rem 0;
        border-top: 1px solid #D9D2C7;
        margin-top: 3rem;
    }

    /* Selectbox / multiselect */
    [data-baseweb="select"] {
        border-radius: 2px !important;
    }

    /* Ocultar los anchor links junto a los títulos */
    h1 > div > a, h2 > div > a, h3 > div > a, h4 > div > a,
    h1 a.anchor-link, h2 a.anchor-link, h3 a.anchor-link, h4 a.anchor-link,
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    """
    Carga los datos de Spotify y las letras desde los CSV.
    """
    artist_df = pd.read_csv('artist_data.csv')
    lyrics_df = pd.read_csv('britney_lyrics.csv')

    if 'album' in lyrics_df.columns and 'short_album_name' not in lyrics_df.columns:
        lyrics_df = lyrics_df.rename(columns={'album': 'short_album_name'})

    for d in [artist_df, lyrics_df]:
        d['name'] = d['name'].str.strip()
        d['short_album_name'] = d['short_album_name'].str.strip()

    merged = artist_df.merge(
        lyrics_df[['name', 'short_album_name', 'lyrics_clean']],
        on=['name', 'short_album_name'],
        how='left',
    )

    def get_sentiment(text):
        if pd.isna(text) or not text:
            return np.nan
        return TextBlob(str(text)).sentiment.polarity

    merged['sentiment'] = merged['lyrics_clean'].apply(get_sentiment)

    def lexical_richness(text):
        if pd.isna(text) or not text:
            return np.nan
        words = [w for w in str(text).split() if w not in STOPWORDS_EN]
        if len(words) == 0:
            return np.nan
        return len(set(words)) / len(words)

    merged['lex_richness'] = merged['lyrics_clean'].apply(lexical_richness)

    album_order = (
        merged.drop_duplicates('short_album_name')
        .sort_values('year')['short_album_name']
        .tolist()
    )

    return merged, album_order


data, ALBUM_ORDER = load_data()


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("<div class='editorial-eyebrow' style='color: #C9A66B;'>Filtro</div>", unsafe_allow_html=True)

selected_albums = st.sidebar.multiselect(
    "Álbumes incluidos en el análisis",
    options=ALBUM_ORDER,
    default=ALBUM_ORDER,
)

st.sidebar.markdown("---")
st.sidebar.markdown("<div class='editorial-eyebrow' style='color: #C9A66B;'>Fuentes</div>", unsafe_allow_html=True)
st.sidebar.markdown(
    "Spotify 1.2M+ Songs — [Kaggle](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs)  \n"
    "Discografía — [Wikipedia](https://en.wikipedia.org/wiki/Britney_Spears_discography)  \n"
    "Letras — lyrics.ovh"
)

# Filtrar datos
df = data[data['short_album_name'].isin(selected_albums)].copy()
if not df.empty:
    df['short_album_name'] = pd.Categorical(
        df['short_album_name'], categories=ALBUM_ORDER, ordered=True
    )


# ============================================================
# HEADER
# ============================================================
st.markdown("<div class='editorial-eyebrow'>Discografía de Britney Spears · 1998 – 2016</div>", unsafe_allow_html=True)

st.markdown("<hr class='editorial-rule'>", unsafe_allow_html=True)

# Métricas
col1, col2, col3, col4 = st.columns(4)
col1.metric("Álbumes", len(selected_albums))
col2.metric("Canciones", len(df))
col3.metric("Letras analizadas", df['lyrics_clean'].notna().sum())
col4.metric(
    "Sentimiento medio",
    f"{df['sentiment'].mean():+.3f}" if df['sentiment'].notna().any() else "—",
)

if df.empty:
    st.markdown("<hr class='editorial-rule'>", unsafe_allow_html=True)
    st.info("Selecciona al menos un álbum en la barra lateral para continuar.")
    st.stop()

st.markdown("<hr class='editorial-rule'>", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["Audio Features", "Análisis de Letras", "Exploración"])


# ------------------------------------------------------------
# TAB 1 — AUDIO FEATURES
# ------------------------------------------------------------
with tab1:
    # --- Tracks por álbum ---
    st.markdown("## Distribución de pistas por álbum")
    counts = df.groupby('short_album_name')['name'].count().reindex(
        [a for a in ALBUM_ORDER if a in selected_albums]
    )
    fig = px.bar(
        x=counts.values, y=counts.index, orientation='h',
        labels={'x': 'Número de pistas', 'y': ''},
        color=counts.values, color_continuous_scale=[[0, ACCENT_SOFT], [1, ACCENT]],
    )
    fig.update_layout(
        height=420, showlegend=False, coloraxis_showscale=False,
        yaxis={'categoryorder': 'array', 'categoryarray': counts.index[::-1]},
    )
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='editorial-rule'>", unsafe_allow_html=True)

    # --- Scatter: Acousticness vs Valence ---
    st.markdown("## Acústica frente a positividad")
    st.caption("Valence mide la positividad musical de 0 a 1, siendo 0 más triste y 1 más alegre.")
    fig = px.scatter(
        df, x='valence', y='acousticness',
        color='short_album_name', size='duration_ms',
        hover_data=['name'],
        color_discrete_sequence=PALETTE,
        labels={'valence': 'Valence', 'acousticness': 'Acousticness', 'short_album_name': 'Álbum'},
    )
    fig.update_layout(height=580, legend_title_text='')
    fig.update_traces(marker=dict(line=dict(width=0.5, color='white'), opacity=0.85))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='editorial-rule'>", unsafe_allow_html=True)

    # --- Radar Chart ---
    st.markdown("## Perfil sonoro por álbum")
    st.caption("Promedio normalizado de las seis características de audio representado por una gráfica de radar.")

    album_means = df.groupby('short_album_name')[RADAR_FEATURES].mean().reindex(
        [a for a in ALBUM_ORDER if a in selected_albums]
    )
    album_norm = (album_means - album_means.min()) / (album_means.max() - album_means.min() + 1e-9)

    fig = go.Figure()
    for i, (album, row) in enumerate(album_norm.iterrows()):
        fig.add_trace(go.Scatterpolar(
            r=row.tolist() + [row.tolist()[0]],
            theta=RADAR_FEATURES + [RADAR_FEATURES[0]],
            fill='toself',
            fillcolor=f'rgba(0,0,0,0)',
            name=album,
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
        ))
    fig.update_layout(
        polar=dict(
            bgcolor=BG,
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#E5DED1", linecolor="#D9D2C7"),
            angularaxis=dict(gridcolor="#E5DED1", linecolor="#D9D2C7"),
        ),
        height=560, legend_title_text='',
    )
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# TAB 2 — Análisis de letras
# ------------------------------------------------------------
with tab2:

    lyrics_data = df[df['lyrics_clean'].notna()]

    if lyrics_data.empty:
        st.info("No hay letras disponibles para los álbumes seleccionados.")
    else:
        # --- Word Cloud ---
        st.markdown("## World Cloud")
        wc_album = st.selectbox("Álbum", ["Discografía completa"] + selected_albums, label_visibility="collapsed")

        if wc_album == "Discografía completa":
            text = ' '.join(lyrics_data['lyrics_clean'].tolist())
        else:
            text = ' '.join(lyrics_data[lyrics_data['short_album_name'] == wc_album]['lyrics_clean'].tolist())

        if text.strip():
            wc = WordCloud(
                width=1000, height=440,
                background_color=BG,
                stopwords=STOPWORDS_EN,
                colormap='RdGy',
                max_words=110,
                prefer_horizontal=0.95,
                relative_scaling=0.45,
                font_path=None,
            ).generate(text)

            fig_wc, ax_wc = plt.subplots(figsize=(12, 5.2))
            fig_wc.patch.set_facecolor(BG)
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)

        st.markdown("<hr class='editorial-rule'>", unsafe_allow_html=True)

        # --- Sentimiento por álbum ---
        st.markdown("## Polaridad emocional de cada álbum")
        st.caption("Polaridad TextBlob en el rango -1 (negativo) a +1 (positivo). Distribución y promedio.")

        sentiment_df = lyrics_data[lyrics_data['sentiment'].notna()]

        col_a, col_b = st.columns(2)

        with col_a:
            fig = px.box(
                sentiment_df.sort_values('short_album_name'),
                x='short_album_name', y='sentiment',
                color='short_album_name',
                color_discrete_sequence=PALETTE,
                labels={'sentiment': 'Polaridad', 'short_album_name': ''},
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=-30, height=460)
            fig.add_hline(y=0, line_dash="dot", line_color="#8C7B6F", opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            mean_sent = (
                sentiment_df.groupby('short_album_name')['sentiment']
                .mean()
                .reindex([a for a in ALBUM_ORDER if a in selected_albums])
                .dropna()
                .reset_index()
            )
            fig = px.line(
                mean_sent, x='short_album_name', y='sentiment',
                markers=True,
                labels={'sentiment': 'Sentimiento medio', 'short_album_name': ''},
                color_discrete_sequence=[ACCENT],
            )
            fig.update_traces(line=dict(width=2.5), marker=dict(size=10, color=ACCENT, line=dict(color='white', width=2)))
            fig.update_layout(xaxis_tickangle=-30, height=460)
            fig.add_hline(y=0, line_dash="dot", line_color="#8C7B6F", opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<hr class='editorial-rule'>", unsafe_allow_html=True)

        # --- Riqueza Léxica ---
        st.markdown("## Riqueza léxica")
        st.caption("Ratio de palabras únicas sobre el total sin stopwords ni otras palabras que tampoco aportan significado.")

        richness = (
            lyrics_data.groupby('short_album_name')['lex_richness']
            .mean()
            .reindex([a for a in ALBUM_ORDER if a in selected_albums])
            .dropna()
            .reset_index()
        )
        fig = px.bar(
            richness, x='short_album_name', y='lex_richness',
            color='lex_richness', color_continuous_scale=[[0, ACCENT_SOFT], [1, ACCENT]],
            labels={'lex_richness': 'Riqueza léxica', 'short_album_name': ''},
        )
        fig.update_layout(xaxis_tickangle=-30, height=420, coloraxis_showscale=False)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<hr class='editorial-rule'>", unsafe_allow_html=True)

        # --- Top palabras ---
        st.markdown("## Las veinte palabras más frecuentes")
        all_words = [w for w in text.split() if w not in STOPWORDS_EN and len(w) > 2]
        freq = Counter(all_words).most_common(20)

        if freq:
            freq_df = pd.DataFrame(freq, columns=['word', 'count'])
            fig = px.bar(
                freq_df, x='count', y='word', orientation='h',
                color='count', color_continuous_scale=[[0, ACCENT_SOFT], [1, ACCENT]],
                labels={'count': 'Apariciones', 'word': ''},
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=520, coloraxis_showscale=False)
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# TAB 3 — EXPLORATION
# ------------------------------------------------------------
with tab3:
    # --- Valence vs Sentiment ---
    st.markdown("## Música contra texto")
    st.caption("Valence de Spotify frente al sentimiento extraído de la letra.")

    scatter_data = df.dropna(subset=['sentiment', 'valence'])
    if not scatter_data.empty:
        fig = px.scatter(
            scatter_data, x='valence', y='sentiment',
            color='short_album_name',
            hover_data=['name'],
            labels={
                'valence': 'Spotify Valence',
                'sentiment': 'Sentimiento de la letra',
                'short_album_name': 'Álbum',
            },
            color_discrete_sequence=PALETTE,
        )
        fig.update_layout(height=560, legend_title_text='')
        fig.update_traces(marker=dict(size=11, line=dict(width=0.5, color='white'), opacity=0.85))
        fig.add_hline(y=0, line_dash="dot", line_color="#8C7B6F", opacity=0.4)
        fig.add_vline(x=0.5, line_dash="dot", line_color="#8C7B6F", opacity=0.4)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='editorial-rule'>", unsafe_allow_html=True)

    # --- Feature explorer ---
    st.markdown("## Explorador de variables")
    col_x, col_y = st.columns(2)
    with col_x:
        feat_x = st.selectbox("Eje horizontal", AUDIO_FEATURES, index=AUDIO_FEATURES.index('valence'))
    with col_y:
        feat_y = st.selectbox("Eje vertical", AUDIO_FEATURES, index=AUDIO_FEATURES.index('energy'))

    fig = px.scatter(
        df, x=feat_x, y=feat_y,
        color='short_album_name',
        hover_data=['name'],
        size='duration_ms', size_max=22,
        color_discrete_sequence=PALETTE,
        labels={'short_album_name': 'Álbum'},
    )
    fig.update_layout(height=560, legend_title_text='')
    fig.update_traces(marker=dict(line=dict(width=0.5, color='white'), opacity=0.85))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='editorial-rule'>", unsafe_allow_html=True)

    # --- Tabla de datos ---
    st.markdown("## Tabla detallada")
    display_cols = ['name', 'short_album_name', 'year'] + AUDIO_FEATURES[:6] + ['sentiment', 'lex_richness']
    available_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[available_cols].sort_values(['year', 'name']),
        use_container_width=True,
        height=420,
    )
