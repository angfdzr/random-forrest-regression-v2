from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model dan scaler
model = joblib.load('model_rf.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        tahun = float(request.form['tahun'])
        produksi = float(request.form['produksi'])
        curah_hujan = float(request.form['curah_hujan'])
        kelembapan = float(request.form['kelembapan'])
        suhu = float(request.form['suhu'])

        input_data = np.array([[tahun, produksi, curah_hujan, kelembapan, suhu]])
        input_scaled = scaler.transform(input_data)

        prediksi = model.predict(input_scaled)

        return render_template('index.html', prediction_text=f'Prediksi Luas Panen: {prediksi[0]:,.2f} ha')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)