from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model dan scaler
model = joblib.load('model_rf3.pkl')
scaler = joblib.load('scaler3.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil input dari form
        tahun = float(request.form['tahun'])
        produksi = float(request.form['produksi'])
        curah_hujan = float(request.form['curah_hujan'])
        kelembapan = float(request.form['kelembapan'])
        suhu = float(request.form['suhu'])

        print("Input dari form:", tahun, produksi, curah_hujan, kelembapan, suhu)  # <== tambahkan ini

        input_data = np.array([[tahun, produksi, curah_hujan, kelembapan, suhu]])
        input_scaled = scaler.transform(input_data) 

        print("Data setelah scaling:", input_scaled)  # <== dan ini

        prediksi = model.predict(input_scaled)

        return render_template('index.html', 
                               prediction_text=f'Prediksi Luas Panen: {prediksi[0]:,.2f} ha',
                               tahun=tahun,
                               produksi=produksi,
                               curah_hujan=curah_hujan,
                               kelembapan=kelembapan,
                               suhu=suhu)
    except:
        return render_template('index.html', prediction_text="Terjadi kesalahan pada input atau prediksi.")

if __name__ == '__main__':
    app.run(debug=True)