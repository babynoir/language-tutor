import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import kstest, expon, norm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import librosa
import soundfile as sf
import speech_recognition as sr
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from coqui_tts import TTS
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Função para adicionar estilos personalizados
def add_custom_styles():
    st.markdown(
        """
        <style>
            /* Estilo global */
            body {
                font-family: 'Arial', sans-serif;
                background-color: #f8f9fa;
                color: #343a40;
            }

            /* Estilo da barra lateral */
            .stSidebar {
                background-color: #ffffff;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            /* Estilo dos botões */
            .stButton button {
                background-color: #007bff !important;
                color: white !important;
                border-radius: 10px !important;
                border: none !important;
                padding: 10px 20px !important;
                font-size: 16px !important;
                transition: background-color 0.3s ease !important;
            }
            .stButton button:hover {
                background-color: #0056b3 !important;
            }

            /* Estilo das caixas de texto */
            .stTextInput input {
                border-radius: 10px !important;
                border: 1px solid #ced4da !important;
                padding: 10px !important;
                font-size: 16px !important;
            }

            /* Estilo das mensagens de sucesso */
            .stSuccess {
                border-radius: 10px !important;
                background-color: #d4edda !important;
                color: #155724 !important;
                padding: 15px !important;
                margin-bottom: 15px !important;
            }

            /* Estilo das mensagens de erro */
            .stError {
                border-radius: 10px !important;
                background-color: #f8d7da !important;
                color: #721c24 !important;
                padding: 15px !important;
                margin-bottom: 15px !important;
            }

            /* Estilo da página principal */
            .main {
                max-width: 1200px !important;
                margin: 0 auto !important;
                padding: 20px !important;
                background-color: #ffffff !important;
                border-radius: 15px !important;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
            }

            /* Estilo dos títulos */
            h1, h2, h3 {
                color: #0056b3 !important;
                text-align: center !important;
                margin-bottom: 20px !important;
            }

            /* Estilo das tabelas */
            table {
                border-radius: 10px !important;
                overflow: hidden !important;
                border: 1px solid #ced4da !important;
            }

            /* Estilo das badges */
            .badge {
                border-radius: 20px !important;
                padding: 5px 10px !important;
                font-size: 14px !important;
                margin: 5px !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Configurações iniciais
st.set_page_config(page_title="Tutor de Idiomas AI", page_icon="🌍", layout="centered")

# Adicionar estilos personalizados
add_custom_styles()

# Dicionário de idiomas
LANGUAGES = {"Inglês": "en", "Espanhol": "es", "Francês": "fr"}

# Inicialização do estado da sessão para gamificação
if "score" not in st.session_state:
    st.session_state.score = 0
    st.session_state.badges = []

# Carregar modelos apenas uma vez
@st.cache_resource
def load_models():
    try:
        translator = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-es-fr")
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-es-fr")
        grammar_corrector = pipeline("text2text-generation", model="pszemraj/flan-t5-base-grammar-synthesis")
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")
        return translator, tokenizer, grammar_corrector, tts
    except Exception as e:
        st.error(f"Falha ao carregar modelos: {e}")
        return None, None, None, None

# Função para analisar a pronúncia
def analyze_pronunciation(audio_path):
    try:
        y, sr = librosa.load(audio_path)
        pitches = librosa.yin(y, fmin=80, fmax=400, sr=sr)
        return f"Tom médio: {np.mean(pitches):.2f} Hz"
    except Exception as e:
        return f"Erro ao analisar pronúncia: {e}"

# Interface principal
st.title("🌟 Tutor de Idiomas AI 🌟")
language = st.selectbox("Escolha o idioma:", list(LANGUAGES.keys()), index=0)

audio_file = st.file_uploader("Grave um áudio (formato WAV):", type=["wav"])

# Processamento do áudio
if audio_file and st.button("🚀 Analisar 🚀"):
    with open("temp.wav", "wb") as f:
        f.write(audio_file.getbuffer())

    # Reconhecimento de Fala
    r = sr.Recognizer()
    with sr.AudioFile("temp.wav") as source:
        audio = r.record(source)
        try:
            text = r.recognize_google(audio, language=LANGUAGES[language])
        except sr.UnknownValueError:
            text = ""
        except sr.RequestError:
            st.error("Não foi possível conectar ao serviço de reconhecimento de fala.")
            text = ""

    if text:
        # Tradução e correção gramatical
        translator, tokenizer, corrector, tts = load_models()
        if translator and tokenizer and corrector and tts:
            inputs = tokenizer(text, return_tensors="pt")
            translated = translator.generate(**inputs)
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            corrected_text = corrector(text=[translated_text])[0]['generated_text']

            # Feedback
            st.subheader("Feedback:")
            st.write(f"**Você disse:** {text}")
            st.write(f"**Correção:** {corrected_text}")
            st.write(f"**Análise de Pronúncia:** {analyze_pronunciation('temp.wav')}")

            # Gamificação
            if translated_text.lower() == corrected_text.lower():
                st.session_state.score += 10
                if st.session_state.score >= 50:
                    st.session_state.badges.append("🏅 Iniciante")
                st.balloons()
            else:
                st.session_state.score = max(0, st.session_state.score - 5)

            st.success(f"Pontuação: {st.session_state.score}")
            st.write("Conquistas:", " ".join([f'<span class="badge" style="background-color:#28a745;">{badge}</span>' for badge in st.session_state.badges]), unsafe_allow_html=True)

            # Áudio da correção
            try:
                tts.tts_to_file(text=corrected_text, file_path="feedback.wav")
                st.audio("feedback.wav")
            except Exception as e:
                st.error(f"Falha ao gerar áudio: {e}")

# Integração com Google Classroom
if st.button("🔗 Conectar ao Google Classroom 🔗"):
    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            "client_secret.json",
            scopes=["https://www.googleapis.com/auth/classroom.courses.readonly"]
        )
        creds = flow.run_local_server(port=0)
        from googleapiclient.discovery import build
        service = build('classroom', 'v1', credentials=creds)
        courses = service.courses().list().execute().get('courses', [])
        st.subheader("Cursos Conectados:")
        for course in courses:
            st.write(f"- {course['name']}")
    except Exception as e:
        st.error(f"Falha ao conectar ao Google Classroom: {e}")

# Recursos Premium (opcional)
with st.expander("💎 Recursos Premium (US$ 7/mês) 💎"):
    st.write("""
    ```python
    # DeepL API (Tradução Premium)
    # import deepl
    # translator = deepl.Translator("SUA_CHAVE_AQUI")  # $6.99/mês
    ```
    """)

# Rodapé animado
st.markdown(
    """
    <style>
        .footer {
            text-align: center;
            font-size: 14px;
            margin-top: 20px;
            color: #6c757d;
            animation: fadeIn 2s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
    <div class="footer">
        Powered by Streamlit & ❤️ | Desenvolvido por Você!
    </div>
    """,
    unsafe_allow_html=True,
)
