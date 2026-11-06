import os
import cv2
import time
import random
import serial
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "phi-4-mini-instruct")

try:
    arduino = serial.Serial('COM3', 9600, timeout=1)
    time.sleep(2)
except:
    arduino = None

llm = ChatOpenAI(
    base_url=azure_endpoint,
    api_key=azure_key,
    model=deployment_name,
    temperature=0.8
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Polegar
    (0, 5), (5, 6), (6, 7), (7, 8),        # Indicador
    (0, 9), (9, 10), (10, 11), (11, 12),   # Médio
    (0, 13), (13, 14), (14, 15), (15, 16), # Anelar
    (0, 17), (17, 18), (18, 19), (19, 20), # Mindinho
    (5, 9), (9, 13), (13, 17)              # Palma
]

def to_pixel(x_norm: float, y_norm: float, w: int, h: int) -> tuple[int, int]:
    x = min(max(x_norm, 0.0), 1.0)
    y = min(max(y_norm, 0.0), 1.0)
    return int(x * w), int(y * h)

def desenhar_landmarks(image_bgr, hand_landmarks_list):
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]
    for hand_landmarks in hand_landmarks_list:
        pts = [to_pixel(lm.x, lm.y, w, h) for lm in hand_landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(annotated, pts[a], pts[b], (0, 255, 0), 2)
        for (x, y) in pts:
            cv2.circle(annotated, (x, y), 3, (0, 0, 255), -1)
    return annotated

def classificar_jogada_usuario(hand_landmarks):
    """Conta os 3 dedos (indicador [8], médio [12], anelar [16])"""
    dedos = []
    pontas = [8, 12, 16]
    for ponta in pontas:
        if hand_landmarks[ponta].y < hand_landmarks[ponta - 2].y:
            dedos.append(1)
        else:
            dedos.append(0)
            
    total_dedos = sum(dedos)
    if total_dedos == 0:
        return "Pedra"
    elif total_dedos >= 3:
        return "Papel"
    elif dedos[0] == 1 and dedos[1] == 1 and dedos[2] == 0:
        return "Tesoura"
    return None

def gerar_comentario_ai(jogada_usuario, jogada_ai, resultado):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é uma mão robótica de 3 dedos jogando Pedra, Papel e Tesoura. Você é sarcástico, competitivo e fala em português de Portugal. Responda em apenas uma frase curta."),
        ("human", f"O humano jogou {jogada_usuario}. Tu jogaste {jogada_ai}. Resultado: {resultado}. Diz algo sobre isso!")
    ])
    chain = prompt | llm
    resposta = chain.invoke({})
    return resposta.content

st.set_page_config(layout="wide", page_title="Robô Pedra, Papel, Tesoura - Tasks API")
st.title("Mão Robótica vs Humano")

if 'estado_jogo' not in st.session_state:
    st.session_state.estado_jogo = 'aguardando'
if 'comentario' not in st.session_state:
    st.session_state.comentario = "Posição inicial: PAPEL. Coloque a mão para começar!"
if 'tempo_resultado' not in st.session_state:
    st.session_state.tempo_resultado = 0

col_video, col_chat = st.columns([2, 1])
video_placeholder = col_video.empty()
info_placeholder = col_chat.empty()

if col_chat.button("🔄 Próxima Ronda / Reiniciar"):
    st.session_state.estado_jogo = 'aguardando'
    st.session_state.comentario = "Coloque a mão em frente à câmera para começar!"
    if arduino:
        arduino.write(b'2')  # Garante posição inicial em Papel

cap = cv2.VideoCapture(0)

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE
)

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        result = landmarker.detect(mp_image)
        mao_detetada = bool(result.hand_landmarks)

        if st.session_state.estado_jogo == 'aguardando':
            info_placeholder.info(st.session_state.comentario)
            if mao_detetada:
                st.session_state.estado_jogo = 'contagem'
                if arduino:
                    arduino.write(b'2')  
                    time.sleep(0.5)
                    arduino.write(b'c')  
                st.session_state.inicio_contagem = time.time()

        elif st.session_state.estado_jogo == 'contagem':
            tempo_decorrido = time.time() - st.session_state.inicio_contagem
            tempo_restante = 3 - int(tempo_decorrido)
            
            if tempo_restante > 0:
                info_placeholder.warning(f"Contagem regressiva: {tempo_restante}...")
                cv2.putText(frame, str(tempo_restante), (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
            else:
                if mao_detetada:
                    jogada_user = classificar_jogada_usuario(result.hand_landmarks[0])
                    if jogada_user:
                        opcoes = ["Pedra", "Papel", "Tesoura"]
                        jogada_ai = random.choice(opcoes)
                        
                        if jogada_user == jogada_ai:
                            vencedor = "Empate"
                        elif (jogada_user == "Pedra" and jogada_ai == "Tesoura") or \
                             (jogada_user == "Papel" and jogada_ai == "Pedra") or \
                             (jogada_user == "Tesoura" and jogada_ai == "Papel"):
                            vencedor = "Humano ganhou"
                        else:
                            vencedor = "Robô ganhou"

                        # Enviar jogada da IA para o Arduino
                        if arduino:
                            if jogada_ai == "Pedra": arduino.write(b'1')
                            elif jogada_ai == "Papel": arduino.write(b'2')
                            elif jogada_ai == "Tesoura": arduino.write(b'3')

                        comentario_ia = gerar_comentario_ai(jogada_user, jogada_ai, vencedor)
                        st.session_state.comentario = f"**Tu:** {jogada_user} | **Robô:** {jogada_ai}\n\n**Resultado:** {vencedor}\n\n🤖 *{comentario_ia}*"
                        st.session_state.estado_jogo = 'resultado'
                        st.session_state.tempo_resultado = time.time()
                    else:
                        st.session_state.estado_jogo = 'aguardando'
                else:
                    st.session_state.estado_jogo = 'aguardando'

        elif st.session_state.estado_jogo == 'resultado':
            info_placeholder.success(st.session_state.comentario)
            cv2.putText(frame, "FIM DE JOGO", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            
            if time.time() - st.session_state.tempo_resultado > 4:
                st.session_state.estado_jogo = 'aguardando'
                st.session_state.comentario = "Nova ronda! Coloque a mão em frente à câmera..."
                if arduino:
                    arduino.write(b'2')  # Reseta para Papel

        if mao_detetada:
            frame = desenhar_landmarks(frame, result.hand_landmarks)

        video_placeholder.image(frame, channels="BGR", use_container_width=True)
        time.sleep(0.03)

cap.release()