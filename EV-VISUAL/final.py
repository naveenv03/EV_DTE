import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score, precision_score, recall_score, f1_score


df = pd.read_csv('final_preprocessed_without_outliers.csv')

Y = df['trip_distance(km)']
X = df.drop(columns=['trip_distance(km)'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(X_train,Y_train)
imp=np.array(['quantity','City_Roads','Motor_way','Country_roads','consumption','A/C','park_heating','avg_speed','ecr_deviation','encoded_driving_style','encoded_tire_type'])
ip=imp.reshape(1,-1)
print("range estimation:")
print(int(model.predict(ip)))




from flask import Flask, render_template, request
app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def basic():
    if request.method == 'POST':
        quantity = (request.form['quantity'])
        City_Roads = float(request.form['City_Roads'])
        Motor_way = float(request.form['Motor_way'])
        Country_roads = float(request.form['Country_roads'])
        consumption = float(request.form['consumption'])
        ac = float(request.form['A/C'])
        park_heating= float(request.form['park_heating'])
        avg_speed = float(request.form['avg_speed'])
        ecr_deviation = float(request.form['ecr_deviation'])
        encoded_driving_style = float(request.form['encoded_drivin)g_style'])
        encoded_tire_type = float(request.form['encoded_tire_type'])
        y_pred1 = array[[quantity, City_Roads, Motor_way, Country_roads , consumption,ac,park_heating,avg_speed ,ecr_deviation ,encoded_driving_style , encoded_tire_type]]
        prediction_outcome = model.predict(y_pred1)
        return render_template('index.html', p=prediction_outcome)
        
    

if __name__ == '__main__':
    app.run(debug=True)