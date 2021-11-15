from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')
model = pickle.load(open('Naive Bayes model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('my_form.html')

@app.route('/', methods=['POST'])
def predict():
    user_input = int(request.form.get('ad')), int(request.form.get('phishing')), int(request.form.get('sender'))
    prediction = model.predict(np.array([user_input]))
    return 'Your email will be categorized as ' + str(prediction[0])

if __name__ == '__main__':
    app.run(port=5000, debug=True)