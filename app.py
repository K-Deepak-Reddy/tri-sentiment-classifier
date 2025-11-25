from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load tokenizer and label encoder
tokenizer = pickle.load(open("models/tokenizer.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))

# Load trained models
model_rnn = load_model("models/rnn_model.h5")
model_lstm = load_model("models/lstm_model.h5")
model_bilstm = load_model("models/bilstm_model.h5")

# Load test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Manual mapping if needed
label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form.get('user_input')
    sequence = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(sequence, maxlen=100)

    # Predict with each model
    preds_rnn = model_rnn.predict(padded)[0]
    preds_lstm = model_lstm.predict(padded)[0]
    preds_bilstm = model_bilstm.predict(padded)[0]

    # Convert predictions to labels
    result = {
        'rnn': label_map[int(np.argmax(preds_rnn))],
        'lstm': label_map[int(np.argmax(preds_lstm))],
        'bilstm': label_map[int(np.argmax(preds_bilstm))]
    }

    return jsonify(result)

@app.route('/evaluate', methods=['GET'])
def evaluate():
    os.makedirs("static/plots", exist_ok=True)

    def plot_cm(model, name, cmap):
        y_pred = model.predict(X_test).argmax(axis=1)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Neutral", "Positive"])
        disp.plot(cmap=cmap)
        plt.title(f"{name} Confusion Matrix")
        plt.savefig(f"static/plots/cm_{name.lower()}.png")
        plt.close()
        acc = (y_test == y_pred).mean()
        return round(float(acc), 4)

    accs = {
        'RNN': plot_cm(model_rnn, 'RNN', 'Blues'),
        'LSTM': plot_cm(model_lstm, 'LSTM', 'Greens'),
        'BiLSTM': plot_cm(model_bilstm, 'BiLSTM', 'Oranges'),
    }

    return jsonify(accs)

if __name__ == '__main__':
    app.run(debug=True)
