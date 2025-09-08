
import streamlit as st
import joblib          # Para cargar modelos
import pandas as pd
from pathlib import Path  # Para manejar rutas de archivos de forma robusta

# ============================
# Configuración de la página
# ============================
st.set_page_config(
    page_title="Test de Bienestar Mental",
    page_icon="💡",
    layout="centered"
)

# ============================
# Encabezado
# ============================
st.title("🧠 Test de Bienestar Mental")

st.markdown("""
Bienvenido a la aplicación de evaluación de bienestar mental.  
Este formulario recoge información sobre **respuestas en escalas clínicas** y **datos demográficos básicos**.  

👉 Al final se podrán usar estos datos como entrada para un modelo de Machine Learning que estime el nivel de bienestar.
""")


st.subheader("📝 Cuestionarios")

tab1, tab2, tab3 = st.tabs(["PHQ-9", "GAD-7", "CD-RISC-10"])

st.markdown(
    """
    <style>
    /* Cambiar fondo de los contenedores dentro de los tabs */
    div[data-baseweb="tab-panel"] {
        background-color: #37ADB0;  /* color de fondo */
        border: 1px solid #000000;  /* borde */
        border-radius: 10px;
        padding: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


with tab1:
    st.markdown("### Cuestionario de Salud del Paciente(PHQ-9)")
    st.markdown("Responde las siguientes 9 preguntas, a continuación se muestra la escala:\n- 0 = Nunca\n- 1 = Varios días\n- 2 = Más de la mitad de los días\n- 3 = Casi todos los días")
    st.markdown("En las últimas 2 semanas, ¿con qué frecuencia le han molestado los siguientes problemas?")
    
    questions = [
        "Poco interés o placer en hacer cosas.",
        "Sentirse desanimado, deprimido o sin esperanza.",
        "Dificultad para quedarse dormido, mantenerse dormido o dormir demasiado.",
        "Sentirse cansado o con poca energía.",
        "Poco apetito o comer en exceso.",
        "Sentirse mal consigo mismo o que es un fracaso o que ha defraudado a su familia o a usted mismo.",
        "Dificultad para concentrarse en cosas, como leer el periódico o ver la televisión.",
        "Moverse o hablar tan despacio que otras personas podrían haberlo notado o estar tan inquieto o agitado que se ha estado moviendo mucho más de lo habitual.",
        "Pensar que estaría mejor muerto o hacerse daño de alguna manera."
    ]
    
    phq_items = []
    for i in range(0, 9):
        val = st.radio(f"{questions[i]}", [0, 1, 2, 3], horizontal=True, key=f"phq{i}")
        phq_items.append(val)
    SUMPHQ = sum(phq_items)
    st.info(f"Puntaje total = {SUMPHQ}")

with tab2:
    st.markdown("###  Evaluación del Trastorno de Ansiedad Generalizada (GAD-7)")
    st.markdown("Responde las siguientes 7 preguntas, a continuación se muestra la escala:\n- 0 = Nunca\n- 1 = Varios días\n- 2 = Más de la mitad de los días\n- 3 = Casi todos los días")
    st.markdown("En las últimas 2 semanas, ¿con qué frecuencia le han molestado los siguientes problemas?")
    
    gad_items = []
    questions2 = [
        "Sentirse nervioso, ansioso o muy alterado.",
        "No ser capaz de dejar de preocuparse o de controlar la preocupación.",
        "Preocuparse demasiado por diferentes cosas.",
        "Dificultad para relajarse.",
        "Estar tan inquieto que es difícil permanecer quieto.",
        "Irritarse o enfadarse con facilidad.",
        "Sentir miedo como si algo terrible pudiera pasar."
    ]
    for i in range(0, 7):
        val = st.radio(f"{questions2[i]}", [0, 1, 2, 3], horizontal=True, key=f"gad{i}")
        gad_items.append(val)
    SUMAGAD = sum(gad_items)
    st.info(f"Puntaje total = {SUMAGAD}")

with tab3:
    st.markdown("### Escala Breve de Resiliencia de Connor-Davidson (CD-RISC-10)")
    st.markdown("Responde las siguientes 10 preguntas, a continuación se muestra la escala:\n- 0 = Nunca\n- 1 = Rara  vez\n- 2 = A veces \n- 3 = A menudo \n- 4 = Casi siempre")
    st.markdown("Indique con qué frecuencia, en el último mes, usted ha sentido que:")
    
    cdrisc_items = []
    questions3 = [
        "Soy capaz de adaptarme a los cambios.",
        "Puedo enfrentar lo que sea que venga.",
        "Trato de ver el lado positivo de las cosas cuando me suceden.",
        "Puedo lidiar con lo que me pasa gracias a mi experiencia pasada.",
        "Puedo alcanzar mis metas a pesar de los obstáculos.",
        "Mantengo la concentración bajo presión.",
        "No me desanimo por los fracasos.",
        "Creo que puedo manejar situaciones desagradables o dolorosas.",
        "Pienso en mí mismo como una persona fuerte.",
        "Soy capaz de mantener el control en situaciones difíciles."
    ]
    for i in range(0, 10):
        val = st.radio(f"{questions3[i]}", [0, 1, 2, 3, 4], horizontal=True, key=f"cdrisc{i}")
        cdrisc_items.append(val)
    SUMCDRISC = sum(cdrisc_items)
    st.info(f"Puntaje total = {SUMCDRISC}")


# ============================
# Entradas - Escalas
# ============================
st.subheader("📊 Escalas clínicas")

st.write("Puntaje total PHQ-9:", SUMPHQ)
st.write("Puntaje total GAD-7:", SUMAGAD)
st.write("Puntaje total CD-RISC-10:", SUMCDRISC)

# ============================
# Entradas - Datos demográficos
# ============================
st.subheader("✨ Datos demográficos")

edad = st.number_input("Edad", min_value=0, max_value=99, step=1)

Semestre = st.number_input("Ingresa tu semestre", min_value=1, max_value=13, step=1)

#-----------------------
unidadesA = {
  "Ciencias Médicas" : 1, 
  "Humanidades" : 2, 
  "Agronomía" : 3,
  "Arquitectura" : 4, 
  "Ciencias Económicas" : 5, 
  "Ciencias Jurídicas y Sociales" : 6, 
  "Ciencias Químicas y Farmacia" : 7, 
  "Ingeniería" : 8, 
  "Medicina Veterinaria y Zootecnia" :9, 
  "Ciencias Psicológicas" : 10, 
  "Historia" : 11, 
  "Trabajo Social" : 12, 
  "Ciencias de la Comunicación" : 13, 
  "Ciencia Política" : 14, 
  "Profesor de Enseñanza Media" : 15, 
  "Arte" : 16, 
  "Ciencias Físicas y Matemáticas" : 17, 
  "Estudios del Mar y Acuicultura" : 18, 
  "Otra" : 19, 
  "Odontología" : 20 
}
UnAca = st.selectbox("Unidad Académica", list(unidadesA.keys()))
unAcaValor = unidadesA[UnAca]
#------------------------
trabajoDatos = {
    "Trabajo de jornada completa" : 0, 
    "Trabajo de medio tiempo" : 1, 
    "Emprendimiento" : 2 
}
Trabajo = st.selectbox("¿Trabaja actualmente?", list(trabajoDatos.keys()))
valorTrabajo = trabajoDatos[Trabajo]
#------------------------
religiones = {
    "Católica": 1, 
    "Evangélica": 2, 
    "Adventista": 3, 
    "Testigo de Jehová": 4, 
    "Judía": 5, 
    "Mormona": 6, 
    "Ninguna": 7, 
    "Otra": 8 
}
Religion = st.selectbox("Religión", list(religiones.keys()))
valorRelig = religiones[Religion]
#------------------------
estadosC = {
    "Soltera (o)":1, 
    "Unida(o)":2, 
    "Casada (o)":3, 
    "Separada(o)":4, 
    "Divorciada(o)":5, 
    "Viuda (o)":6, 
    "Otro":7, 
    "Noviazgo / compromiso":8 
}
EstCivil = st.selectbox("Estado Civil", list(estadosC.keys()))
valorEstC = estadosC[EstCivil]
#------------------------
centrosUniv = {
    "Campus Central (Zona 12)": 0,
    "Centro Universitario Metropolitano, Ciudad Capital (Zona 11)": 1,
    "Facultad de Humanidades, Sedes Departamentales": 2,
    "Centro Universitario de Occidente -CUNOC-": 3,
    "Centro Universitario del Norte -CUNOR-": 4,
    "Centro Universitario de San Marcos -CUSAM-": 5,
    "Centro Universitario de Oriente -CUNORI-": 6,
    "Centro Universitario de Suroriente -CUNSURORI-": 7,
    "Centro Universitario de Peten -CUDEP-": 8,
    "Centro Universitario de Chimaltenango -CUNDECH-": 9,
    "Centro Universitario de Quiche -CUSACQ-": 10,
    "Centro Universitario de Noroccidente -CUNOROC-": 11,
    "Centro Universitario del Sur -CUNSUR-": 12,
    "Centro Universitario de Santa Rosa -CUSARO-": 13,
    "Centro Universitario de Jutiapa -JUSAC-": 14,
    "Centro Universitario de Izaba -CUNIZAB-": 15,
    "Centro Universitario de Suroccidente -CUNSUROC-": 16,
    "Centro Universitario de El Progreso -CUNPROGRESO-": 17,
    "Centro Universitario de Baja Verapaz -CUNBAV-": 18,
    "Centro Universitario de Totonicapán -CUNTOTO-": 19,
    "Centro Universitario de Zacapa -CUNZAC-": 20,
    "Centro Universitario de Sololá -CUNSOL-": 21,
    "Centro Universitario de Sacatepéquez - CUNSAC-": 22,
    "Centro Universitario Retalhuleu -CUNREU-": 23,
    "Instituto Tecnológico Maya de Estudios Superiores -ITMES-": 24,
    "Instituto Tecnológico Universitario Guatemala SUR -ITUGS-": 25,
    "Centro de Estudios del Mar y Acuicultura -CEMA-": 26,
    "Paraninfo Universitario, Zona 1 Guatemala": 27,
    "Otro": 28
}
CEntroU = st.selectbox("Centro Universitario", list(centrosUniv.keys()))
valorCentro = centrosUniv[CEntroU]
#-----------------------
jornadas = {
    "Matutina":0, 
    "Vespertina": 1, 
    "Nocturna":2, 
    "Fin de Semana" : 3, 
    "Cierre de pensum" : 4 
}
Jornada = st.selectbox("Jornada",list(jornadas.keys()))
valorJorn = jornadas[Jornada]

# ============================
# Mostrar datos capturados
# ============================
flag = False
if st.button("📝 Enviar datos"):
    flag = True

dataUser= {
    "SUMPHQ": SUMPHQ,
    "SumaGAD": SUMAGAD,
    "SUMCDrisc": SUMCDRISC,
    "edad": edad,
    "Semestre": Semestre,
    "UnAca": unAcaValor,
    "Trabajo": valorTrabajo,
    "Religion": valorRelig,
    "EstCivil": valorEstC,
    "CEntroU": valorCentro,
    "Jornada": valorJorn
}
#------------- carga y proceso de modelos
BASE_DIR = Path(__file__).parent  # Carpeta donde está app.py
model_GaussianNB = joblib.load(BASE_DIR / "m1_GaussianNB.joblib")
model_KNN = joblib.load(BASE_DIR / "m3_KNN_model.joblib")
model_MLP = joblib.load(BASE_DIR / "m4_MLP_classifier.joblib")

# Cargar scaler y modelo pkl
scaler_KM = joblib.load(BASE_DIR / "m2_scaler.pkl")
model_Kmeans = joblib.load(BASE_DIR / "m2_modelo_kmeans.pkl")

etiquetas = {
        0: "Desanimado (Languishing)",
        1: "Moderado",
        2: "Florecido (Flourishing)"
    }

if flag:
    # ---------- GAUSSIAN NB
    X_new_GNB = pd.DataFrame([dataUser])
    pred1 = model_GaussianNB.predict(X_new_GNB)
    pred_proba1 = model_GaussianNB.predict_proba(X_new_GNB)  # opcional: probabilidades de cada clase

    pred_text1 = etiquetas[pred1[0]]
    
    # Formatear probabilidades en %
    prob_percent = [f"{p*100:.1f}%" for p in pred_proba1[0]]
    
    # Probabilidades
    prob_line = ", ".join([f"{etiquetas[c]}: {prob_percent[i]}" 
                        for i, c in enumerate(model_GaussianNB.classes_)])

    st.markdown(
        f"""
        <div style="
            background-color:#d1fae5;  /* verde pastel */
            border: 2px solid #10b981;  /* verde más oscuro */
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 15px;
            color: #000000;
        ">
            <h3>🟢 GaussianNB - Predicción</h3>
            <p><b>Predicción:</b> {pred_text1}</p>
            <p><b>Probabilidades:</b> {prob_line}</p>
            <p style="color:#b91c1c;"><b>⚠️ Nota:</b> El modelo tiene una accuracy aproximada de 0.699 y no reemplaza evaluación profesional.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ------- KNN
    X_new_KNN = pd.DataFrame([dataUser])

    pred2 = model_KNN.predict(X_new_KNN)
    pred_proba2 = model_KNN.predict_proba(X_new_KNN)  # probabilidades por clase

    pred_text2 = etiquetas[pred2[0]]
    prob_percent2 = [f"{p*100:.1f}%" for p in pred_proba2[0]]
    prob_line2 = ", ".join([f"{etiquetas[c]}: {prob_percent2[i]}" 
                            for i, c in enumerate(model_GaussianNB.classes_)])

    st.markdown(
        f"""
        <div style="
            background-color:#D1F1FA;  
            border: 2px solid #05647A;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 15px;
            color: #000000;
        ">
            <h3>🟠 K-Nearest Neighbors - Predicción</h3>
            <p><b>Predicción:</b> {pred_text2}</p>
            <p><b>Probabilidades:</b> {prob_line2}</p>
            <p style="color:#b91c1c;"><b>⚠️ Nota: El modelo tiene una accuracy aproximada de 0.691 y no reemplaza una evaluación profesional. Esta predicción es solo indicativa.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ------- MLPClassifier
    X_new_MLP = pd.DataFrame([dataUser])
    pred3 = model_MLP.predict(X_new_MLP)
    pred_proba3 = model_MLP.predict_proba(X_new_MLP)  # probabilidades por clase

    pred_text3 = etiquetas[pred3[0]]
    prob_percent3 = [f"{p*100:.1f}%" for p in pred_proba3[0]]

    prob_line3 = ", ".join([f"{etiquetas[c]}: {prob_percent3[i]}" 
                            for i, c in enumerate(model_MLP.classes_)])
    
    st.markdown(
        f"""
        <div style="
            background-color:#D6D1FA;  
            border: 2px solid #6155CF;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 15px;
            color: #000000;
        ">
            <h3>🟡 MLPClassifier - Predicción</h3>
            <p><b>Predicción:</b> {pred_text3}</p>
            <p><b>Probabilidades:</b> {prob_line3}</p>
            <p style="color:#b91c1c;"><b>⚠️ Nota: Este modelo tiene una accuracy aproximada de 0.7404 y no reemplaza una evaluación profesional.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    #----- KMEANS
    X_new_KM = pd.DataFrame([dataUser])

    # Escalar los datos y predecir cluster
    X_new_scaled = scaler_KM.transform(X_new_KM)
    pred_cluster = model_Kmeans.predict(X_new_scaled)[0]

    # Ejemplo de proporciones por cluster (ajustar según tu df original)
    distribucion_clusters = {
        0: {0: 0.114, 1: 0.760, 2: 0.125},
        1: {0: 0.192, 1: 0.759, 2: 0.049},
        2: {0: 0.582, 1: 0.410, 2: 0.008}
    }

    probs_cluster = distribucion_clusters[pred_cluster]
    prob_percent_KM = [f"{v*100:.1f}%" for v in probs_cluster.values()]

    etiquetas = {0: "Desanimado (Languishing)", 1: "Moderado", 2: "Florecido (Flourishing)"}
    prob_line_KM = ", ".join([f"{etiquetas[c]}: {prob_percent_KM[i]}" 
                            for i, c in enumerate(probs_cluster.keys())])

    st.markdown(
        f"""
        <div style="
            background-color:#F9FAD1;  
            border: 2px solid #A0A32F;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 15px;
            color: #000000;
        ">
            <h3>🔵 Kmeans - Clasificación</h3>
            <p><b>Cluster asignado:</b> {pred_cluster}</p>
            <p><b>Distribución aproximada de bienestar mental en este cluster:</b> {prob_line_KM}</p>
            <p style="color:#b91c1c;"><b>⚠️ Nota: KMeans es un modelo no supervisado. Los clusters representan patrones generales, no un diagnóstico directo.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Aceptar ✔️"):
        flag=False