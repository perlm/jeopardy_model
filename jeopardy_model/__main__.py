from .getData import *
from .buildModel import *
from .tweetIt import *
from .validation import *
import datetime, os, pickle, bz2, subprocess, boto3



##
# This is the master script which will call functions from the other scripts.
# Currently setup to check for new data, scrape, model, and publish.
# scheduled to run on cron.
## 


def main():
	s3 = boto3.client('s3')
	if not os.path.isdir('{}/jeopardy_model/data/'.format(os.path.expanduser("~"))):os.makedirs('{}/jeopardy_model/data'.format(os.path.expanduser("~")))

	# Local or using cloud run?
	if (False):
		getRawData()
		d = readRawFile()
		d2 = constructFeatures(d)
		d2 = d2.loc[(d2['afterdatecutoff']==1)]
		X, X_scaled, Y, scaler, X_fix = processData(d2,dateFeature = None)
		model = buildLogisticModel(X_scaled,Y,X_fix)

		with bz2.BZ2File("{}/jeopardy_model/model_pickles/model.pickle".format(os.path.expanduser("~")),"w") as f:
			pickle.dump(model, f)
		with bz2.BZ2File("{}/jeopardy_model/model_pickles/scaler.pickle".format(os.path.expanduser("~")),"w") as f:
			pickle.dump(scaler, f)
		with bz2.BZ2File("{}/jeopardy_model/model_pickles/indexDict.pickle".format(os.path.expanduser("~")),"w") as f:
			pickle.dump(columns, f)

		gamePage = getMostRecentSoup()
		for g in gamePage:
			features,lastWin = getCurrentStatus(g)
			if features is not None:
				break
	else:
		s3 = boto3.resource('s3')
		s3.Bucket('jeopardydata').download_file('raw.data', '{}/jeopardy_model/data/raw.data'.format(os.path.expanduser("~")))

                with bz2.BZ2File("{}/jeopardy_model/model_pickles/scaler.pickle".format(os.path.expanduser("~")),"r") as f:
                        scaler = pickle.load(f)
                with bz2.BZ2File("{}/jeopardy_model/model_pickles/model.pickle".format(os.path.expanduser("~")),"r") as f:
                        model = pickle.load(f)
                with bz2.BZ2File("{}/jeopardy_model/model_pickles/indexDict.pickle".format(os.path.expanduser("~")),"r") as f:
                        columns = pickle.load(f)

        	s3 = boto3.resource('s3')
        	s3.Bucket('jeopardydata').download_file('features.pickle', 'features.pickle')
        	with bz2.BZ2File("features.pickle","r") as f:
			features = pickle.load(f)



	# df is from file. df2 is with features. x, scaled, fixed are after processing.
	# predict - df3 is with additional row for predictions. then process in exact same way.

	d3 = createNewInput(features) 
	d4 = constructFeatures(d3)
	d4 = d4.loc[(d4['afterdatecutoff']==1)]
	X, X_scaled, Y, scaler, X_fix = processData(d4, None,scaler,columns)
	prob = predict(X_scaled,model)
	features['prob'] = prob

	# tweet it!
	daysOld = datetime.date.today() - datetime.datetime.strptime(features['date'], '%Y-%m-%d').date()

	print "Last game is ", daysOld, " days old. From ", features['date']

	if (datetime.datetime.today().weekday()==0 and daysOld.days <= 3) or (datetime.datetime.today().weekday()<=4 and daysOld.days <= 1):
		tweetProb(features)
		
		probs = predict_all(X_scaled,model)
		store_predictions(d4,probs)

        ##########
        # validate previous predictions!
        #validate()




if __name__ == "__main__":
	main()
