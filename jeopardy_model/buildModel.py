import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn import preprocessing, feature_extraction, linear_model, metrics, model_selection
import math, os, boto3

####
# This file contains functions for building the classification model.
###

def optimizeLambdaLogistic(X_train, X_test, y_train, y_test,L='l1'):
	# use CV to optimize regularization hyperparameter! (using either L1 or L2) (lambda is inverse C here)

	if L=='l1':
		tuned_parameters = [ {'C':[1e-5,1e-3,5e-3,7.5e-3,1e-2,2.5e-2,5e-2,1e-1,5e-1,1e0,1e8]}]
	else:
		tuned_parameters = [ {'C':[1e-8,1e-6,1e-4,1e-2, 1e0,1e2,1e4,1e6,1e8]}]

	clf = model_selection.GridSearchCV(linear_model.LogisticRegression(penalty=L), tuned_parameters, cv=50,scoring='roc_auc')
	clf.fit(X_train, y_train)

	print "Hyperparameter Optimization, penalty=", L
	print(clf.best_params_)

	means = clf.cv_results_['mean_test_score']
	stds = clf.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, clf.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

	y_prob = clf.predict_proba(X_test)[:,1]
	y_class = clf.predict(X_test)
	#print y_test, y_prob, y_class
	#print "Hyperparameter Optimization"
	#print(metrics.classification_report(y_test, y_class))

	return clf.best_params_

def buildLogisticModel(X_scaled,Y,X_fix):
	# build a model! l1 for lasso, l2 for ridge

	# use CV and holdout.
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X_scaled, Y, test_size=0.3, random_state=0)

	# need to reshape for some reason...
	Y = Y.as_matrix()
	c, r = Y.shape
	Y = Y.reshape(c,)

	y_train = y_train.as_matrix()
	c, r = y_train.shape
	y_train = y_train.reshape(c,)

	y_test = y_test.as_matrix()
	c, r = y_test.shape
	y_test = y_test.reshape(c,)

	# optimize hyperparameter	
	la = optimizeLambdaLogistic(X_train, X_test, y_train, y_test,'l1')
	#lb = optimizeLambdaLogistic(X_train, X_test, y_train, y_test,'l2')	# consistently less good.

	# train model using hyperparameter
	model = linear_model.LogisticRegression(C=la['C'], penalty='l1')
	model.fit(X_train,y_train)

	y_prob = model.predict_proba(X_test)[:,1]
	y_class = model.predict(X_test)
	print "Final Model: Out of Sample Performance"
	print(metrics.classification_report(y_test, y_class))

	print "AUC:", metrics.roc_auc_score(y_test, y_prob)

	# retrain on whole data set.
	model = linear_model.LogisticRegression(C=la['C'], penalty='l1')
	model.fit(X_scaled,Y)

	print model.intercept_
	factors = list(X_fix.columns.values)
	coefs 	= list(model.coef_.ravel())
	for i,f in enumerate(factors):
		print f,"\t", coefs[i]

	return model


def predict_all(X_scaled,model):
    # for a given model and independent data, generate predictions
    y_prob = model.predict_proba(X_scaled)[:,1]
    return y_prob

def predict(X_scaled,model):
	# for a given model and independent data, generate predictions
	y_prob = model.predict_proba(X_scaled)[:,1]
	print "Avg of predictions= ", np.mean(y_prob)
	return y_prob[-1:]

def readRawFile():
	# read in csv from s3 and return pandas dataframe
	s3 = boto3.client('s3')
	#obj = s3.get_object(Bucket='jeopardydata', Key='raw.data')
	gameData = pd.read_csv('s3://jeopardydata/raw.data',delimiter=',',header=None, names=['g', 'gameNumber', 'date', 'winningDays', 'winningDollars', 'winner', 'gender', 'age', 'name', 'career', 'location'])
	return gameData

def readRawFile_local():
	# read in csv and return pandas dataframe
	gameData = pd.read_csv('{0}/jeopardy_model/data/raw.data'.format(os.path.expanduser("~")),delimiter=',',header=None, names=['g', 'gameNumber', 'date', 'winningDays', 'winningDollars', 'winner', 'gender', 'age', 'name', 'career', 'location'])
	return gameData


def constructFeatures(dff):
    #####################3
	# take pandas dataframe and manipulate to construct additional features.
    #######################

	df = dff.copy()
	df['avePrevDollars'] = df['winningDollars']/df['winningDays']
	df['prevWins_capped'] = np.minimum(df['winningDays'],10)
	
	# date variables
	#df['relative_year'] = 2017-df['date'].str[:4].apply(int)
	df['date'] = pd.to_datetime(df['date'])
	df['relative_year'] = 2017-pd.DatetimeIndex(df['date']).year
	df['afterdatecutoff'] = 0
	df.loc[(df['date']>'2004-10-04'),'afterdatecutoff'] =  1

    #monday
	df['monday'] = 0
	df.loc[(pd.DatetimeIndex(df['date']).dayofweek==0),'monday'] =  1

	# pandas bucketing!
	df['age_bucket'] = pd.cut(df['age'],bins=[0,35,55,500],labels=['a_lt35','b_35-55','c_gt55'])
	df['Avg_Dollars_buckets'] = pd.cut(df['avePrevDollars'],bins=[0,10000,30000,1e10],labels=['a_lt10k','b_10-30k','c_gt30k'])

	df['cityofchampions'] = 0
	df.loc[df['location'].isin(['Los Angeles California','Washington D.C.','Chicago Illinois','New York New York','Brooklyn New York','Arlington Virginia','Seattle Washington','New York City New York','Atlanta Georgia','San Diego California','Austin Texas','Philadelphia Pennsylvania','San Francisco California','Minneapolis Minnesota','Boston Massachusetts','Denver Colorado','Houston Texas','Baltimore Maryland','New Orleans Louisiana','Louisville Kentucky']),'cityofchampions'] = 1

	df['jobs'] = 'other'
	df.loc[df['career'].str.contains('attorney|lawyer|law student'),'jobs'] = 'law'
	df.loc[df['career'].str.contains('student|Ph.D. candidate'),'jobs'] = 'student'
	df.loc[df['career'].str.contains('writer'),'jobs'] = 'writer'
	df.loc[df['career'].str.contains('home'),'jobs'] = 'home'
	df.loc[df['career'].str.contains('teach|professor|college instructor|librarian'),'jobs'] = 'teacher'
	df.loc[df['career'].str.contains('engineer|scien|software|programmer'),'jobs'] = 'tech'
	df.loc[df['career'].str.contains('home'),'jobs'] = 'home'

	return df

def return_post_cutoff(df, dateFeature = 'relative_year'):
	################
	# take dataframe and return scaled versions after date cutoff for validation
	################
    
	# pick the features! If no date feature, cutoff after rule change (10/2004)
	if dateFeature is not None:
		X = df[['prevWins_capped','avePrevDollars','gender','age_bucket','cityofchampions','jobs',dateFeature]].loc[(df['afterdatecutoff']==1)]
	else:
		X = df[['prevWins_capped','avePrevDollars','gender','age_bucket','cityofchampions','jobs']].loc[(df['afterdatecutoff']==1)]

	X_fix = pd.get_dummies(X)
	scaler = preprocessing.StandardScaler().fit(X_fix)
	X_scaled = scaler.transform(X_fix)
	Y = df[['winner']].loc[(df['afterdatecutoff']==1)]

	return X,X_scaled, Y, scaler, X_fix


def processData(df, dateFeature = 'relative_year', scaler=None, columns=None):
	################3
	# take dataframe and reformat for sci-kit learn including normalization
	# include form of date feature as parameter so I can test its effect
	################
    
	# pick the features! If no date feature, cutoff after rule change (10/2004)
    if dateFeature is not None:
        X = df[['prevWins_capped','avePrevDollars','gender','age_bucket','cityofchampions','jobs',dateFeature]]
    else:
        X = df[['prevWins_capped','avePrevDollars','gender','age_bucket','cityofchampions','jobs']]

	# scikit learn can only handle numeric or dummy.
	# need to make multiple columns of 1/0 dummies
	# https://gist.github.com/ramhiser/982ce339d5f8c9a769a0
    X_fix = pd.get_dummies(X)

    if columns is not None:
	X_fix = X_fix.reindex(columns = columns, fill_value=0)


    # Could try this interaction term!
    #X_fix['Avg_Dollars_buckets_a_lt10k'] = X_fix['Avg_Dollars_buckets_a_lt10k']*X_fix['prevWins_capped']
    #X_fix['Avg_Dollars_buckets_b_10-30k'] = X_fix['Avg_Dollars_buckets_b_10-30k']*X_fix['prevWins_capped']
    #X_fix['Avg_Dollars_buckets_c_gt30k'] = X_fix['Avg_Dollars_buckets_c_gt30k']*X_fix['prevWins_capped']
    
    Y = df[['winner']]
    if scaler is None:
        print "Win Rate in data= ", Y['winner'].mean()
        scaler = preprocessing.StandardScaler().fit(X_fix)
        X_scaled = scaler.transform(X_fix)
    else:
        X_scaled = scaler.transform(X_fix)

    return X,X_scaled, Y, scaler, X_fix

	
def addRow(df, features):
	# add in row of new data which needs to be predicted.
	#gameData = pd.read_csv('data/raw.data',delimiter=',',header=None, names=['g', 'gameNumber', 'date', 'winningDays', 'winningDollars', 'winner', 'gender', 'age', 'name', 'career', 'location'])
	#features = {'date':date,'days':winningDays,'dollars':winningDollars,'gender':gender,'age':age,'name':name.replace(',',' '),'career':career.replace(',',' '),'location':location.replace(',',' ')}

	# given a dictionary of values for new row, turn it into a matching dataframe
	nr = {'g':pd.Series([0]),
		'gameNumber':pd.Series([0]),
		'date':pd.Series([features['date']]),
		'winningDays':pd.Series([int(features['days'])]),
		'winningDollars':pd.Series([int(features['dollars'])]),
		'winner':pd.Series([0]),
		'gender':pd.Series([features['gender']]),
		'age':pd.Series([features['age']]),
		'name':pd.Series([features['name']]),
		'career':pd.Series([features['career']]),
		'location':pd.Series([features['location']])}
	newrow = pd.DataFrame(nr)

	# column order looks like dict, rather than dataframe.
	newrow2 = newrow[['g', 'gameNumber', 'date', 'winningDays', 'winningDollars', 'winner', 'gender', 'age', 'name', 'career', 'location']]

	df2 = pd.concat([df, newrow2])
	return(df2)

def createNewInput(features):
        # given a dictionary of values for new row, turn it into a matching dataframe
        nr = {'g':pd.Series([0]),
                'gameNumber':pd.Series([0]),
                'date':pd.Series([features['date']]),
                'winningDays':pd.Series([int(features['days'])]),
                'winningDollars':pd.Series([int(features['dollars'])]),
                'winner':pd.Series([0]),
                'gender':pd.Series([features['gender']]),
                'age':pd.Series([features['age']]),
                'name':pd.Series([features['name']]),
                'career':pd.Series([features['career']]),
                'location':pd.Series([features['location']])}
        newrow = pd.DataFrame(nr)

        # column order looks like dict, rather than dataframe.
        newrow2 = newrow[['g', 'gameNumber', 'date', 'winningDays', 'winningDollars', 'winner', 'gender', 'age', 'name', 'career', 'location']]
	return(newrow2)



if __name__ == '__main__':
	df = readRawFile()
	df = constructFeatures(df)
	df = df.loc[(df['afterdatecutoff']==1)]
	X, X_scaled, Y, scaler, X_fix = processData(df)
	model = buildLogisticModel(X_scaled,Y,X_fix)



