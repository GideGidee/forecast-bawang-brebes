import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# st.set_page_config(page_title="Time Series Dashboard", layout="centered")

# CSS untuk card
card_css = """
<style>
.card {
    background: #ffffff;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    padding: 20px;
    width: 250px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
}

.card-title {
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 10px;
    color: #333;
}

.card-number {
    font-weight: bold;
    color: #009000;
}
</style>
"""

# HTML untuk card
card_html = """
<div class="card" style="margin-bottom: 20px;">
    <div class="card-title">{title}</div>
    <div class="card-number">{number}</div>
</div>
"""

# Load model TFLite
@st.cache_resource
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


# Fungsi scaling dan inverse scaling
def scaled_price(data):
    return data / 1000


def inverse_scale(data):
    return data * 1000


# Fungsi prediksi 2 minggu dengan model TFLite
def predict_2_weeks(actual, window_size, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    act_temp = actual.reshape(-1, 1).astype(
        np.float32
    )  # Pastikan data menggunakan FLOAT32
    seq_res = np.array([])

    for _ in range(2):  # Prediksi 2 minggu
        actual_scaled = scaled_price(act_temp)
        seq = actual_scaled[-window_size:]

        interpreter.set_tensor(
            input_details[0]["index"], seq.reshape(1, window_size, 1).astype(np.float32)
        )
        interpreter.invoke()

        hasil = interpreter.get_tensor(output_details[0]["index"])
        hasil = inverse_scale(hasil)

        seq_res = np.append(seq_res, hasil.item())
        act_temp = np.append(act_temp, np.expand_dims([hasil.item()], axis=1))

    return seq_res


# Fungsi untuk menampilkan hasil prediksi sebagai card
def display_prediction(predictions):
    # Pastikan predictions adalah iterable (list atau array)
    if isinstance(predictions, np.ndarray):
        predictions = predictions.tolist()  # Convert ndarray ke list jika perlu
    elif isinstance(predictions, float):  # Jika hanya ada satu prediksi, ubah ke list
        predictions = [predictions]

    # Menampilkan setiap prediksi dalam card
    for idx, pred in enumerate(predictions, 1):
        # Streamlit app
        st.markdown(card_css, unsafe_allow_html=True)
        # Menampilkan card
        st.markdown(
            card_html.format(
                title=f"Prediction week-{idx}",
                number=f"Rp{pred-1000:.2f} - Rp{pred+1000:.2f}",
            ),
            unsafe_allow_html=True,
        )


# Fungsi untuk menampilkan chart dengan Plotly
def display_chart(actual, predictions):
    df = pd.DataFrame(
        {
            "Actual": np.concatenate([actual[-4:], np.nan * np.ones(2)]),
            "Predictions": np.concatenate([np.nan * np.ones(3), actual[-1:], predictions]),
        }
    )

    # Membuat grafik menggunakan Plotly
    fig = go.Figure()

    # Garis untuk data aktual dengan warna biru
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Actual"],
            mode="lines",
            name="Actual",
            line=dict(color="blue"),
        )
    )

    # Memberikan layout untuk grafik
    fig.update_layout(
        title="Actual vs Predicted Data",
        xaxis_title="Weeks",
        yaxis_title="Value",
        legend_title="Legend",
        showlegend=True,
    )

    # Garis untuk data prediksi dengan warna merah
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Predictions"],
            mode="lines",
            name="Predictions",
            line=dict(color="red"),
        )
    )

    # Menampilkan chart dengan Plotly
    st.plotly_chart(fig)


# Streamlit Dashboard
st.title("Time Series Forecasting with TFLite")

# Ganti bagian ini dengan pembacaan CSV dari direktori
csv_file_path = "data/weekly_data.csv"
data = pd.read_csv(csv_file_path)

# Tampilkan data yang di-upload
st.write("Data:", data.head())
st.line_chart(data)

# Pilih kolom data aktual
column_name = "Harga/week"
series = np.array(data[column_name])

# Pilih ukuran window
window_size = 24

# Load model TFLite
interpreter = load_tflite_model("model.tflite")

# Inisialisasi data aktual di session_state jika belum ada
if "actual" not in st.session_state:
    st.session_state["actual"] = series  # Data awal dari file

# Tampilkan data aktual saat ini
st.write("Last 20 weeks Actual Data:")
st.line_chart(st.session_state["actual"][-20:])

# Prediksi 2 minggu pertama
st.write("Predicted Data for Next 2 Weeks:")
predictions = predict_2_weeks(st.session_state["actual"], window_size, interpreter)
display_prediction(predictions)

# Input 2 data aktual dari pengguna
st.write("Please input 2 new actual data for further prediction:")
actual_data_1 = st.number_input("Input 1st week:", min_value=0.0, key="input_1")
actual_data_2 = st.number_input("Input 2nd week:", min_value=0.0, key="input_2")

# Tombol untuk memulai prediksi 2 minggu berikutnya
if st.button("Predict"):
    if actual_data_1 > 0 and actual_data_2 > 0:
        try:
            # Tambahkan data aktual baru ke session_state
            st.session_state["actual"] = np.append(
                st.session_state["actual"], [actual_data_1, actual_data_2]
            )

            # Prediksi 2 minggu berikutnya
            predictions = predict_2_weeks(
                st.session_state["actual"], window_size, interpreter
            )

            # Tampilkan hasil prediksi
            st.write("Predicted Data for Next 2 Weeks:")
            display_prediction(predictions)
            display_chart(st.session_state["actual"], predictions)

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please input valid data points for prediction!")
