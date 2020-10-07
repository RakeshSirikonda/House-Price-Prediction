from flask import Flask,render_template,url_for, send_from_directory
from flask import request
import loadmodel
from flask import jsonify
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import get_dummies
import pickle
app=Flask(__name__, static_url_path='/static')
#model=loadmodel.load_model()
@app.route('/result/',methods=['GET','POST'])
def home():
	return render_template('coverpage.html')

@app.route('/result/firstpage.html',methods=['GET','POST'])
def fp():
	return render_template('firstpage.html')

@app.route('/result/hpp.html',methods=['GET','POST'])
def hpp():
	return render_template('hpp.html')

@app.route('/pred',methods=['GET','POST'])
def pred():
	houses = pd.read_csv("housingdata.csv")
	feature_cols = ['OverallQual','GrLivArea','GarageCars','GarageArea','YearBuilt','YearRemodAdd','OverallCond']
	x = houses[feature_cols] # predictor
	x=get_dummies(x)
	y = houses.SalePrice # response
	x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2) 
   	#linreg = LinearRegression()
	linreg = LinearRegression()
	linreg.fit(x_train, y_train)
	linreg.score(x_test,y_test)
	filename='finalised_model.sav'
	pickle.dump(linreg,open(filename,'wb'))
	

	if request.method =='POST':
		oq=request.form['oq']
		grdar=request.form['grdar']
		grgcr=request.form['grgcr']
		grgar=request.form['grgar']
		yrb=request.form['yrb']
		yrr=request.form['yrr']
		oc=request.form['oc']
		val1l=[[int(oq),int(grdar),int(grgcr),int(grgar),int(yrb),int(yrr),int(oc)]]
		arr=np.array(val1l)
		predic=linreg.predict(arr)
		p=round(predic[0],2)
	return render_template('hpp.html',prediction=p)



@app.route('/result/cpp.html',methods=['POST'])
def cpp():
	return render_template('cpp.html')

@app.route('/static/<path:path>')
def getStatic(path):
	return send_from_directory('static', path)

#def prediction():
#	inp =str(request.args.get('input'))
#	values=inp.split(',')
#	out=loadmodel.get_output(model,int(values[0]),int(values[1]),int(values[2]),int(values[3]),int(values[4]),int(values[5]),int(values[6]))
#	out=str(out)
#	return jsonify(out)

if 	__name__=='__main__':
	app.run(debug=True)
