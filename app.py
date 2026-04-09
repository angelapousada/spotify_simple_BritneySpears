"""
Spotify + Lyrics Analysis — Britney Spears
Web App (Streamlit) — Parte 3 del trabajo

Ejecutar localmente:  streamlit run app.py
Desplegar:           Streamlit Community Cloud (https://share.streamlit.io)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from textblob import TextBlob
from collections import Counter
import matplotlib.pyplot as plt
import nltk
import re

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Britney Spears — Spotify & Lyrics Analysis",
    page_icon="🎤",
    layout="wide",
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
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    """
    Carga los datos de Spotify y las letras desde los CSV.
    artist_data.csv = DataFrame de artist_df (audio features filtrado)
    britney_lyrics.csv = DataFrame de lyrics_df (letras descargadas)
    """
    artist_df = pd.read_csv('artist_data.csv')
    lyrics_df = pd.read_csv('britney_lyrics.csv')

    # Merge
    merged = artist_df.merge(lyrics_df[['name', 'album', 'lyrics_clean']],
                             on=['name', 'album'], how='left')

    # Sentimiento
    def get_sentiment(text):
        if pd.isna(text) or not text:
            return np.nan
        return TextBlob(str(text)).sentiment.polarity

    merged['sentiment'] = merged['lyrics_clean'].apply(get_sentiment)

    # Riqueza léxica
    def lexical_richness(text):
        if pd.isna(text) or not text:
            return np.nan
        words = [w for w in str(text).split() if w not in STOPWORDS_EN]
        if len(words) == 0:
            return np.nan
        return len(set(words)) / len(words)

    merged['lex_richness'] = merged['lyrics_clean'].apply(lexical_richness)

    # Orden cronológico
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
st.sidebar.image("https://i.imgur.com/8QHzVfd.png", width=200)  # Placeholder logo
st.sidebar.title("🎤 Britney Spears")
st.sidebar.markdown("**Spotify + Lyrics Analysis**")
st.sidebar.markdown("---")

selected_albums = st.sidebar.multiselect(
    "Selecciona álbumes:",
    options=ALBUM_ORDER,
    default=ALBUM_ORDER,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "📊 Dataset: [Spotify 1.2M+ Songs](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs)  \n"
    "📖 Discografía: [Wikipedia](https://en.wikipedia.org/wiki/Britney_Spears_discography)"
)

# Filtrar datos
df = data[data['short_album_name'].isin(selected_albums)].copy()
df['short_album_name'] = pd.Categorical(
    df['short_album_name'], categories=ALBUM_ORDER, ordered=True
)


# ============================================================
# HEADER
# ============================================================
st.title("🎵 Britney Spears — Spotify & Lyrics Analysis")
st.markdown(
    "Análisis exploratorio de audio features de Spotify y letras de canciones "
    "de los **álbumes de estudio** de Britney Spears."
)

# Métricas rápidas
col1, col2, col3, col4 = st.columns(4)
col1.metric("Álbumes", len(selected_albums))
col2.metric("Canciones", len(df))
col3.metric("Letras encontradas", df['lyrics_clean'].notna().sum())
col4.metric("Sentimiento medio", f"{df['sentiment'].mean():.3f}" if df['sentiment'].notna().any() else "N/A")


# ============================================================
# TAB 1 — AUDIO FEATURES
# ============================================================
tab1, tab2, tab3 = st.tabs(["🎧 Audio Features", "📝 Análisis de Letras", "🔬 Exploración"])

with tab1:
    st.header("Audio Features de Spotify")

    # --- Tracks por álbum ---
    st.subheader("Tracks por álbum")
    counts = df.groupby('short_album_name')['name'].count().reindex(
        [a for a in ALBUM_ORDER if a in selected_albums]
    )
    fig = px.bar(
        x=counts.values, y=counts.index, orientation='h',
        labels={'x': 'Tracks', 'y': ''},
        color=counts.values, color_continuous_scale='Magma',
    )
    fig.update_layout(height=400, showlegend=False, yaxis={'categoryorder': 'array', 'categoryarray': counts.index[::-1]})
    st.plotly_chart(fig, use_container_width=True)

    # --- Scatter: Acousticness vs Valence ---
    st.subheader("Acousticness vs. Valence")
    st.caption("Valence = positividad musical (0=triste, 1=alegre). Tamaño = duración.")
    fig = px.scatter(
        df, x='valence', y='acousticness',
        color='short_album_name', size='duration_ms',
        hover_data=['name'],
        color_discrete_sequence=px.colors.qualitative.Vivid,
        labels={'valence': 'Valence', 'acousticness': 'Acousticness', 'short_album_name': 'Album'},
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # --- Radar Chart ---
    st.subheader("Radar Chart — Audio Features promedio por álbum")
    album_means = df.groupby('short_album_name')[RADAR_FEATURES].mean().reindex(
        [a for a in ALBUM_ORDER if a in selected_albums]
    )
    album_norm = (album_means - album_means.min()) / (album_means.max() - album_means.min() + 1e-9)

    fig = go.Figure()
    colors = px.colors.qualitative.Vivid
    for i, (album, row) in enumerate(album_norm.iterrows()):
        fig.add_trace(go.Scatterpolar(
            r=row.tolist() + [row.tolist()[0]],
            theta=RADAR_FEATURES + [RADAR_FEATURES[0]],
            fill='toself', fillcolor=f'rgba(0,0,0,0)',
            name=album,
            line=dict(color=colors[i % len(colors)], width=2),
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=550)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 2 — LYRICS ANALYSIS
# ============================================================
with tab2:
    st.header("Análisis de Letras")

    lyrics_data = df[df['lyrics_clean'].notna()]

    if lyrics_data.empty:
        st.warning("No hay letras disponibles para los álbumes seleccionados.")
    else:
        # --- Word Cloud ---
        st.subheader("Nube de Palabras")
        wc_album = st.selectbox("Selecciona álbum para la nube:", ["Todos"] + selected_albums)

        if wc_album == "Todos":
            text = ' '.join(lyrics_data['lyrics_clean'].tolist())
        else:
            text = ' '.join(lyrics_data[lyrics_data['short_album_name'] == wc_album]['lyrics_clean'].tolist())

        if text.strip():
            wc = WordCloud(
                width=800, height=400,
                background_color='white',
                stopwords=STOPWORDS_EN,
                colormap='magma',
                max_words=100,
            ).generate(text)

            fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)

        # --- Sentimiento por álbum ---
        st.subheader("Sentimiento por Álbum")
        st.caption("Polaridad TextBlob: -1 (negativo) a +1 (positivo)")

        sentiment_df = lyrics_data[lyrics_data['sentiment'].notna()]

        col_a, col_b = st.columns(2)

        with col_a:
            fig = px.box(
                sentiment_df.sort_values('short_album_name'),
                x='short_album_name', y='sentiment',
                color='short_album_name',
                color_discrete_sequence=px.colors.qualitative.Vivid,
                labels={'sentiment': 'Polaridad', 'short_album_name': 'Album'},
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=-30, height=450)
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
                color_discrete_sequence=['#e07be0'],
            )
            fig.update_layout(xaxis_tickangle=-30, height=450)
            st.plotly_chart(fig, use_container_width=True)

        # --- Riqueza Léxica ---
        st.subheader("Riqueza Léxica por Álbum")
        st.caption("Ratio palabras únicas / total (sin stopwords). Mayor = vocabulario más diverso.")

        richness = (
            lyrics_data.groupby('short_album_name')['lex_richness']
            .mean()
            .reindex([a for a in ALBUM_ORDER if a in selected_albums])
            .dropna()
            .reset_index()
        )
        fig = px.bar(
            richness, x='short_album_name', y='lex_richness',
            color='lex_richness', color_continuous_scale='Magma',
            labels={'lex_richness': 'Riqueza Léxica', 'short_album_name': ''},
        )
        fig.update_layout(xaxis_tickangle=-30, height=400)
        st.plotly_chart(fig, use_container_width=True)

        # --- Top palabras ---
        st.subheader("Top 20 Palabras Más Frecuentes")
        all_words = [w for w in text.split() if w not in STOPWORDS_EN and len(w) > 2]
        freq = Counter(all_words).most_common(20)

        if freq:
            freq_df = pd.DataFrame(freq, columns=['word', 'count'])
            fig = px.bar(
                freq_df, x='count', y='word', orientation='h',
                color='count', color_continuous_scale='Magma',
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 3 — EXPLORATION
# ============================================================
with tab3:
    st.header("Exploración Interactiva")

    # --- Valence vs Sentiment ---
    st.subheader("Spotify Valence vs. Sentimiento de la Letra")
    st.caption("¿Coincide cómo 'suena' una canción con lo que 'dice'?")

    scatter_data = df.dropna(subset=['sentiment', 'valence'])
    if not scatter_data.empty:
        fig = px.scatter(
            scatter_data, x='valence', y='sentiment',
            color='short_album_name',
            hover_data=['name'],
            labels={
                'valence': 'Spotify Valence (positividad musical)',
                'sentiment': 'Sentimiento letra (TextBlob)',
                'short_album_name': 'Album',
            },
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)

    # --- Feature explorer ---
    st.subheader("Explorador de Features")
    col_x, col_y = st.columns(2)
    with col_x:
        feat_x = st.selectbox("Eje X:", AUDIO_FEATURES, index=AUDIO_FEATURES.index('valence'))
    with col_y:
        feat_y = st.selectbox("Eje Y:", AUDIO_FEATURES, index=AUDIO_FEATURES.index('energy'))

    fig = px.scatter(
        df, x=feat_x, y=feat_y,
        color='short_album_name',
        hover_data=['name'],
        size='duration_ms', size_max=20,
        color_discrete_sequence=px.colors.qualitative.Vivid,
        labels={'short_album_name': 'Album'},
    )
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)

    # --- Tabla de datos ---
    st.subheader("Tabla de Datos")
    display_cols = ['name', 'short_album_name', 'year'] + AUDIO_FEATURES[:6] + ['sentiment', 'lex_richness']
    available_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[available_cols].sort_values(['year', 'name']),
        use_container_width=True,
        height=400,
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "**Trabajo Individual — Extracción de Información** | "
    "Dataset: Spotify 1.2M+ Songs (Kaggle) | "
    "Letras: AZLyrics / lyrics.ovh"
)