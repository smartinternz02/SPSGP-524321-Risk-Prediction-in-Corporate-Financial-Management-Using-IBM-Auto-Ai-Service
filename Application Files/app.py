import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


# List of numerical columns
numerical_columns = ['age', 'credit amount', 'duration']

# Create a scaler object
scaler = StandardScaler()



with open("./model.pkl",'rb') as file:
    model = pickle.load(file)

from flask import Flask,request, jsonify,render_template

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('forms.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    # final_features.append([np.array(features)])
    # scaled_features = scaler.fit_transform(final_features)
    # scaled_features = scaler.fit_transform(final_features)
    # final_features = np.array(final_features)
    # final_features.reshape(1, -1) 
    # final_features[0] = scaler.fit_transform(final_features[0])
    # final_features[6] = scaler.fit_transform(final_features[6]) 
    # final_features[7] = scaler.fit_transform(final_features[7])     
    prediction = model.predict(final_features)
    # age,sex,job,housing,saving account,checking acconut, credit amount,duration,purpose
    #     0-male,
    print(final_features)
    output = prediction[0]
    print("Output: -",output)
    s = ""
    for i in final_features:
        s = " "+str(i)+" "
        print(i)
    

    if output == 1:
        return render_template('forms.html', prediction_text = "Good")
    else:
        return render_template('forms.html',prediction_text ="bad")

if __name__ == '__main__':
    app.run(debug=True)