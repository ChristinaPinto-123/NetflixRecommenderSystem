import streamlit as st
import json
import pandas as pd
from openai import OpenAI
from enum import Enum
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
load_dotenv()
github_api = os.getenv("GITHUB_TOKEN")
MODEL = "gpt-4o-mini"
client = OpenAI(base_url="https://models.inference.ai.azure.com", api_key=github_api)

MASCOT_PATH = "mascot.png"
USER_ICON_PATH = "user.png"

@st.cache_data
def load_data():
    return pd.read_csv('netflix_movies_detailed_up_to_2025.csv')

df = load_data()

# --- 2. SCHEMAS ---
class MovieGenre(str, Enum):
    action, adventure, animation, comedy = "Action", "Adventure", "Animation", "Comedy"
    crime, documentary, drama, family = "Crime", "Documentary", "Drama", "Family"
    fantasy, horror, romance, sci_fi, thriller = "Fantasy", "Horror", "Romance", "Science Fiction", "Thriller"

class MovieSearchParameters(BaseModel):
    genre: Optional[MovieGenre] = None
    min_rating: Optional[float] = 0.0
    start_year: Optional[int] = 1900
    end_year: Optional[int] = 2025
    keywords: Optional[str] = None

# --- 3. SEARCH LOGIC ---
def search_movies(params: MovieSearchParameters):
    results = df.copy()
    if params.genre:
        results = results[results['genres'].str.contains(params.genre.value, na=False, case=False)]

    results = results[(results['vote_average'] >= params.min_rating) &
                      (results['release_year'] >= params.start_year) &
                      (results['release_year'] <= params.end_year)]

    if params.keywords:
        k_list = params.keywords.lower().split()
        def get_match_score(text):
            text = str(text).lower()
            return sum(1 for word in k_list if word in text)

        results['match_score'] = results['description'].apply(get_match_score) + \
                                 results['title'].apply(get_match_score)
        results = results[results['match_score'] > 0]
        results = results.sort_values(by=['match_score', 'popularity'], ascending=False)
    else:
        results = results.sort_values(by='popularity', ascending=False)

    return results.head(3).to_dict(orient='records')

# --- 4. FRONTEND ---
st.set_page_config(
    page_title="Netflix Recommender",
    page_icon= MASCOT_PATH,
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background-color: #335c67;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(134,171,161,0.15) 0%, transparent 70%),
        radial-gradient(ellipse 40% 30% at 90% 80%, rgba(134,171,161,0.08) 0%, transparent 60%);
    font-family: 'DM Sans', sans-serif;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; max-width: 760px !important; }

/* ── Hero Header ── */
.scout-hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
}
.scout-wordmark {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 4rem !important; 
    letter-spacing: 0.15em !important;
    color: #ffffff !important;
    line-height: 0.8 !important; /* Tighter for a cinematic look */
    margin: 0 auto !important;
    display: block !important;
    text-align: center !important;
}
.scout-wordmark span {
    color: #86aba1;
}
.scout-tagline {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    font-size: 0.85rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #617891;
    margin-top: 0.5rem;
}
.scout-divider {
    width: 40px;
    height: 2px;
    background: #86aba1;
    margin: 1.5rem auto;
    border-radius: 2px;
}

/* ── Chat Messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.25rem 0 !important;
}

/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
}

.stChatMessage [data-testid="stMarkdownContainer"] p {
    color: #617891 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}

/* ── Custom message wrappers ── */
.msg-user {
    background: linear-gradient(135deg, #86aba1 0%, #86aba1 100%);
    color: #fff !important;
    border-radius: 20px 20px 4px 20px;
    padding: 0.75rem 1.1rem;
    max-width: 75%;
    margin-left: auto;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    line-height: 1.5;
    box-shadow: 0 4px 20px rgba(134,171,161,0.08);
}
.msg-assistant {
    background: #86aba1;
    border: 1px solid #86aba1;
    color: #003049 !important;
    border-radius: 20px 20px 20px 4px;
    padding: 0.75rem 1.1rem;
    max-width: 85%;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    line-height: 1.5;
}

/* ── Movie Cards ── */
.movie-card {
    background: #1d2b46;
    border: 1px solid #1d2b46;
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    margin: 0.75rem 0;
    transition: border-color 0.2s, transform 0.2s;
    position: relative;
    overflow: hidden;
}
.movie-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #86aba1, #37685b);
    border-radius: 3px 0 0 3px;
}
.movie-card:hover {
    border-color: #86aba1;
    transform: translateY(-2px);
}
.movie-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.5rem;
    letter-spacing: 0.05em;
    color: #ffffff;
    margin: 0 0 0.3rem 0;
}
.movie-meta {
    display: flex;
    gap: 1rem;
    align-items: center;
    margin-bottom: 0.75rem;
    flex-wrap: wrap;
}
.meta-pill {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 100px;
    background: #1e1e2e;
    color: #9898b0;
    border: 1px solid #2a2a40;
}
.meta-rating {
    background: rgba(134,171,161,0.15;
    color: #86aba1;
    border-color: rgba(134,171,161,0.15);
}
.movie-desc {
    font-size: 0.875rem;
    color: #7878a0;
    line-height: 1.65;
    font-weight: 300;
    margin: 0;
}

/* ── Result header ── */
.result-header {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #86aba1;
    margin-bottom: 0.5rem;
    font-weight: 500;
}

/* ── Chat Input ── */
[data-testid="stChatInput"] {
    background: #13131c !important;
    border: 1px solid #2a2a38 !important;
    border-radius: 16px !important;
    color: #e2e2ea !important;
    font-family: 'DM Sans', sans-serif !important;
    box-shadow: none !important;
}

/* REMOVE RED BORDER ON FOCUS */
[data-testid="stChatInput"]:focus-within {
    border: 1px solid #86aba1 !important;
    box-shadow: 0 0 0 2px rgba(134,171,161,0.15) !important;
}

/* Sometimes Streamlit adds error state */
[data-testid="stChatInput"][aria-invalid="true"] {
    border: 1px solid #86aba1 !important;
    box-shadow: none !important;
}

/* ALSO override textarea focus (important) */
[data-testid="stChatInput"] textarea:focus {
    outline: none !important;
    box-shadow: none !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    color: #6b6b7a !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2a2a38; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #86aba1; }

/* ── No results / error ── */
.no-result {
    text-align: center;
    padding: 1.5rem;
    color: #4a4a60;
    font-style: italic;
    font-size: 0.9rem;
}
.stAlert {
    background: rgba(134,171,161,0.08) !important;
    border: 1px solid rgba(134,171,161,0.08) !important;
    border-radius: 12px !important;
    color: #86aba1 !important;
}

/* ── Avatar icons ── */
/* ── Avatar icons ── */
/* 1. Force the user avatar container to show up */
[data-testid="chatAvatarIcon-user"] {
    display: flex !important; 
    visibility: visible !important;
    background-color: transparent !important;
}

/* 2. Style the actual image inside the user avatar */
[data-testid="chatAvatarIcon-user"] img {
    border-radius: 50% !important; 
    border: 2px solid #86aba1 !important;
    width: 45px !important;
    height: 45px !important;
    object-fit: cover !important;
    display: block !important;
}

/* 3. Keep the assistant (dragon) icon clean */
[data-testid="chatAvatarIcon-assistant"] img {
    border-radius: 50% !important;
    border: 2px solid #86aba1 !important;
    box-shadow: 0 0 10px rgba(134,171,161,0.08) !important;
}
            
/* ── Mascot hero image ── */
.mascot-hero {
    width: 140px; /* Bigger hero mascot */
    height: auto;
    border: none !important;      /* Removed circle border */
    border-radius: 0% !important;  /* Removed circle shape */
    box-shadow: none !important;  /* Removed glow */
    object-fit: contain;
    margin: 0 auto 1rem;
    display: block;
}

[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] div[data-baseweb="input"] {
    border: 1px solid #2a2a38 !important;
    box-shadow: none !important;
}

/* Focus state */
[data-testid="stChatInput"]:focus-within,
[data-testid="stChatInput"] div[data-baseweb="input"]:focus-within {
    border: 1px solid #86aba1 !important;
    box-shadow: 0 0 0 2px rgba(134,171,161,0.15) !important;
}

/* Remove Streamlit error red */
[data-testid="stChatInput"][aria-invalid="true"],
[data-testid="stChatInput"][aria-invalid="true"] * {
    border-color: #86aba1 !important;
    box-shadow: none !important;
}

/* Remove browser default red outline */
textarea:focus,
textarea:focus-visible {
    outline: none !important;
    box-shadow: none !important;
}
*:focus,
*:focus-visible,
*:focus-within {
    outline: none !important;
    box-shadow: none !important;
}

div[data-baseweb="base-input"],
div[data-baseweb="input"] {
    border: 1px solid #2a2a38 !important;
    box-shadow: none !important;
}
div[data-baseweb="base-input"]:focus-within,
div[data-baseweb="input"]:focus-within {
    border: 1px solid #86aba1 !important;
    box-shadow: 0 0 0 2px rgba(134,171,161,0.15) !important;
}

/* 🔥 Override ANY red inline styles */
[style*="red"],
[style*="#ff"],
[style*="rgb(255"] {
    border-color: #86aba1 !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──
import base64

def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

if os.path.exists(MASCOT_PATH):
    mascot_b64 = img_to_base64(MASCOT_PATH)
    mascot_html = f'<img class="mascot-hero" src="data:image/png;base64,{mascot_b64}" alt="Scout"/>'
else:
    mascot_html = '<div style="font-size:4rem;margin-bottom:1rem;">🎬</div>'

st.markdown(f"""
<div class="scout-hero">
    {mascot_html}
    <p class="scout-wordmark">NEXT<span>WATCH</span></p>
    <p class="scout-tagline">AI-Powered Movie Discovery</p>
    <div class="scout-divider"></div>
</div>
""", unsafe_allow_html=True)

# ── Chat History ──
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "user":
        avatar = USER_ICON_PATH if os.path.exists(USER_ICON_PATH) else "👤"
    else:
        avatar = MASCOT_PATH if message["role"] == "assistant" and os.path.exists(MASCOT_PATH) else None
    with st.chat_message(message["role"], avatar=avatar):
        if message["role"] == "user":
            st.markdown(f'<div class="msg-user">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="msg-assistant">{message["content"]}</div>', unsafe_allow_html=True)
        if "movies" in message:
            st.markdown('<p class="result-header">Top Picks</p>', unsafe_allow_html=True)
            for m in message["movies"]:
                rating = m.get('vote_average', 'N/A')
                year   = m.get('release_year', '')
                genres = m.get('genres', '')
                desc   = m.get('description', 'No description available.')
                title  = m.get('title', 'Untitled')
                st.markdown(f"""
                <div class="movie-card">
                    <p class="movie-title">{title}</p>
                    <div class="movie-meta">
                        <span class="meta-pill">{year}</span>
                        <span class="meta-pill meta-rating">★ {rating}/10</span>
                        <span class="meta-pill">{genres[:30] if genres else ''}</span>
                    </div>
                    <p class="movie-desc">{desc}</p>
                </div>
                """, unsafe_allow_html=True)

# ── Input ──
if prompt := st.chat_input("Describe what you're in the mood for…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f'<div class="msg-user">{prompt}</div>', unsafe_allow_html=True)

    asst_avatar = MASCOT_PATH if os.path.exists(MASCOT_PATH) else None
    with st.chat_message("assistant", avatar=asst_avatar):
        with st.spinner("Scanning the catalogue…"):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a movie expert. Extract search parameters from the user's request."},
                        {"role": "user", "content": prompt}
                    ],
                    tools=[{"type": "function", "function": {
                        "name": "search_movies",
                        "description": "Search the movie database",
                        "parameters": MovieSearchParameters.model_json_schema()
                    }}],
                    tool_choice={"type": "function", "function": {"name": "search_movies"}}
                )

                args   = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
                params = MovieSearchParameters(**args)
                movies = search_movies(params)

                if not movies:
                    msg = "Nothing matched your vibe exactly — try broadening the description a little."
                    st.markdown(f'<div class="msg-assistant"><div class="no-result">🎬 {msg}</div></div>', unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                else:
                    msg = "Here's what I found for you:"
                    st.markdown(f'<div class="msg-assistant">{msg}</div>', unsafe_allow_html=True)
                    st.markdown('<p class="result-header">Top Picks</p>', unsafe_allow_html=True)
                    for m in movies:
                        rating = m.get('vote_average', 'N/A')
                        year   = m.get('release_year', '')
                        genres = m.get('genres', '')
                        desc   = m.get('description', 'No description available.')
                        title  = m.get('title', 'Untitled')
                        st.markdown(f"""
                        <div class="movie-card">
                            <p class="movie-title">{title}</p>
                            <div class="movie-meta">
                                <span class="meta-pill">{year}</span>
                                <span class="meta-pill meta-rating">★ {rating}/10</span>
                                <span class="meta-pill">{genres[:30] if genres else ''}</span>
                            </div>
                            <p class="movie-desc">{desc}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": msg, "movies": movies})

            except Exception as e:
                st.markdown(
                f'<div class="msg-assistant">⚠️ Something went wrong: {e}</div>',
                unsafe_allow_html=True
                )