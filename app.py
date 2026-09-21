import streamlit as st
import joblib
import numpy as np

# ---------------------------------------------------------------
# 1) Cargamos el modelo K-NN y el escalador que exportamos desde el notebook
#    (son el resultado de todo el trabajo de las Fases II a IV del Taller 6)
# ---------------------------------------------------------------
modelo_knn = joblib.load('modelo_knn_segmentos.pkl')
scaler = joblib.load('scaler_segmentos.pkl')

# ---------------------------------------------------------------
# 2) Información de negocio de cada clúster (según el perfil que
#    encontramos en el punto 4.2 del notebook)
# ---------------------------------------------------------------
info_clusters = {
    0: {
        "emoji": "🏷️",
        "nombre": "Promoción agresiva",
        "color": "#FFF3CD",
        "borde": "#F5C451",
        "descripcion": (
            "Envíos livianos con descuentos comerciales muy altos (alrededor del 40%). "
            "Casi el 100% de estos envíos llega a tiempo — parecen recibir prioridad logística."
        ),
        "riesgo": "Riesgo de retraso: muy bajo ✅",
    },
    1: {
        "emoji": "📦",
        "nombre": "Envío estándar pesado",
        "color": "#F8D7DA",
        "borde": "#E8899A",
        "descripcion": (
            "Paquetes más pesados con descuentos bajos. Es el grupo más grande y también "
            "el que más se retrasa dentro de la operación."
        ),
        "riesgo": "Riesgo de retraso: alto ⚠️",
    },
    2: {
        "emoji": "🙋",
        "nombre": "Cliente frecuente",
        "color": "#D6EAF8",
        "borde": "#7FB3D5",
        "descripcion": (
            "Clientes con más llamadas a servicio al cliente y más compras previas. "
            "Su comportamiento de entrega es intermedio."
        ),
        "riesgo": "Riesgo de retraso: intermedio 🟡",
    },
}

# ---------------------------------------------------------------
# 3) Configuración visual de la página: fondo claro, título con emoji
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Clasificador de segmento logístico",
    page_icon="📦",
    layout="centered",
)

# Un poco de estilo propio: fondo clarito, tarjetas redondeadas, tipografía amigable
# (además del archivo .streamlit/config.toml, que fija el tema claro "de raíz",
# forzamos también aquí los colores de texto, por si el navegador de alguien
# intenta imponer su propio modo oscuro por encima)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFDF9;
        color: #3B3B3B;
    }
    h1, h2, h3, p, label, span, .stMarkdown, .stCaption {
        color: #3B3B3B !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #FBF6EE;
    }
    div[data-testid="stForm"] {
        background-color: #FFFFFF;
        padding: 1.5rem 1.5rem 0.5rem 1.5rem;
        border-radius: 18px;
        border: 1px solid #F0E6D6;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }
    div[data-testid="stNumberInput"] input {
        background-color: #FFFFFF !important;
        color: #3B3B3B !important;
        border: 1px solid #E0D6C2 !important;
    }
    .tarjeta-resultado {
        padding: 1.2rem 1.5rem;
        border-radius: 16px;
        margin-top: 1rem;
    }
    .tarjeta-resultado, .tarjeta-resultado * {
        color: #2B2B2B !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# 4) Encabezado y explicación en lenguaje sencillo
# ---------------------------------------------------------------
st.title("📦 ¿A qué grupo pertenece este envío?")
st.write(
    "Esta herramienta usa un modelo entrenado con **K-Means + K-NN** (Taller 6) para "
    "decirte, en segundos, a qué **segmento operativo** pertenece un envío nuevo, "
    "según sus características. 🚚✨"
)
st.write(
    "👉 Llena los campos con los datos del envío (puedes cambiarlos como quieras) "
    "y presiona **Clasificar envío**."
)

st.divider()

# ---------------------------------------------------------------
# 5) Formulario de entrada — un campo por variable, con explicación y emoji
# ---------------------------------------------------------------
with st.form("formulario_envio"):
    st.subheader("✏️ Datos del envío")

    col1, col2 = st.columns(2)

    with col1:
        costo = st.number_input(
            "💲 Costo del producto",
            min_value=0.0, value=200.0, step=1.0,
            help="Precio del producto que se está enviando.",
        )
        descuento = st.slider(
            "🏷️ Descuento ofrecido (%)",
            min_value=0, max_value=65, value=10,
            help="Porcentaje de descuento comercial aplicado a este envío.",
        )
        compras_previas = st.number_input(
            "🔁 Compras previas del cliente",
            min_value=0, value=3, step=1,
            help="Cuántas veces este cliente ha comprado antes.",
        )

    with col2:
        peso = st.number_input(
            "⚖️ Peso del envío (gramos)",
            min_value=0.0, value=3000.0, step=50.0,
            help="Peso total del paquete en gramos.",
        )
        llamadas = st.slider(
            "☎️ Llamadas a servicio al cliente",
            min_value=0, max_value=10, value=4,
            help="Número de veces que el cliente ha llamado por este pedido.",
        )
        calificacion = st.slider(
            "⭐ Calificación del cliente",
            min_value=1, max_value=5, value=3,
            help="Qué tan satisfecho está el cliente, de 1 (bajo) a 5 (alto).",
        )

    enviado = st.form_submit_button("🔎 Clasificar envío")

# ---------------------------------------------------------------
# 6) Cuando el usuario da clic: transformamos los datos y predecimos
# ---------------------------------------------------------------
if enviado:
    datos_nuevos = np.array([[costo, peso, descuento, llamadas, compras_previas, calificacion]])
    datos_escalados = scaler.transform(datos_nuevos)
    cluster_predicho = int(modelo_knn.predict(datos_escalados)[0])

    info = info_clusters.get(cluster_predicho)

    st.markdown(
        f"""
        <div class="tarjeta-resultado" style="background-color:{info['color']};
                    border: 2px solid {info['borde']};">
            <h3 style="margin-top:0;">{info['emoji']} Segmento: {info['nombre']}</h3>
            <p style="font-size:1.05rem;">{info['descripcion']}</p>
            <p style="font-weight:600;">{info['riesgo']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🤓 ¿Cómo llegó el modelo a esta respuesta?"):
        st.write(
            "El modelo compara este envío contra otros envíos ya agrupados por **K-Means** "
            "y mira a cuáles se parece más (sus vecinos más cercanos, de ahí el nombre "
            "**K-NN**). No vuelve a agrupar todo desde cero: solo mide distancias contra "
            "los grupos que ya existen, por eso responde tan rápido. 🐇"
        )

st.divider()
st.caption(
    "Hecho para el Taller 6 — ICYA3004, Herramientas de Inteligencia Artificial, "
    "Universidad de los Andes."
)
