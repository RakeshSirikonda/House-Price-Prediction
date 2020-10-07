import pickle

def load_model():
	filename='finalised_model.sav'
	load_model=pickle.load(open(filename,'rb'))
	return load_model

def get_output(model,value1,value2,value3,value4,value5,value6,value7,value8,value9,value10,value11,value12):
	out=model.predict([[value1,value2,value3,value4,value5,value6,value7,value8,value9,value10,value11,value12]])
	return out