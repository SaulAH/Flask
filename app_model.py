from flask import Flask, jsonify, request
import os
import pandas as pd
import numpy as np
import pickle

os.chdir(os.path.dirname(__file__))

app= Flask(__name__)
app.config['DEBUG'] = True

@app.route('/',methods=['GET'])
def hello():
    return 'Mi primera API Flask'

@app.route('/api/v1/predict',methods=['GET'])
def predict():
    modelo = pickle.load(open('ad_model.pkl','rb'))
    tv = int(request.args.get('tv',None))
    radio = int(request.args.get('radio',None))
    newspaper = int(request.args.get('newspaper',None))

    if tv is None or radio is None or newspaper is None:
        return 'Error, no puede ser vacío'
    else:
        prediction = modelo.predict([[tv,radio,newspaper]])
    
    return jsonify({'prediction':prediction[0]})

@app.route('/api/v1/retrain',methods=['PUT'])
def retrain():
    data = pd.read_csv('data(Advertaising_.csv',index_col=0)
    X_train,X_test,y_train,y_test = train_test_split(data.drop(columns=['sales']),
                                                data['sales'],
                                                test_size = 0.20,
                                                random_state=42)
    
    model = Lasso(alpha=6000)
    model.fit(X_train, y_train)

    pickle.dump(model, open('ad_model.pkl','wb'))

    return 'Model retrained. New evaluation metric RMSE'+str(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))

if __name__=='__main__':
    app.run()