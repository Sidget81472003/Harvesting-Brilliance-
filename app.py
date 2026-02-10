from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        float_features = [float(x) for x in request.form.values()]
        
        final_features = [np.array(float_features)]
        
        prediction = model.predict(final_features)
        
        if prediction[0] == 0:
            result = "Çerçevelik"
        else:
            result = "Ürgüp Sivrisi"
            
        return render_template('predict.html', prediction_text='The Pumpkin Seed Variety is {}'.format(result))
    
    except Exception as e:
        return render_template('predict.html', prediction_text='Error: {}'.format(e))

if __name__ == "__main__":
    app.run(debug=True)