from .getData import *
from .buildModel import *
from .tweetIt import *
from .validation import *
import datetime, os

##
# This is the master script which will call functions from the other scripts.
# Currently setup to check for new data, scrape, model, and publish.
# scheduled to run on cron.
## 


def main():
	if not os.path.isdir('{}/jeopardy_model/data/'.format(os.path.expanduser("~"))):os.makedirs('{}/jeopardy_model/data'.format(os.path.expanduser("~")))

	# download 
	getRawData()

	# model - df is from file. df2 is with features. x, scaled, fixed are after processing.
	d = readRawFile()
	d2 = constructFeatures(d)
	d2 = d2.loc[(d2['afterdatecutoff']==1)]
	X, X_scaled, Y, scaler, X_fix = processData(d2,dateFeature = None)
	model = buildLogisticModel(X_scaled,Y,X_fix)

	# predict - df3 is with additional row for predictions. then process in exact same way.
	features,lastWin = getCurrentStatus()
	d3 = addRow(d,features)
	d4 = constructFeatures(d3)
	d4 = d4.loc[(d4['afterdatecutoff']==1)]
	X, X_scaled, Y, scaler, X_fix = processData(d4, None,scaler)
	prob = predict(X_scaled,model)
	features['prob'] = prob

	# tweet it!
	daysOld = datetime.date.today() - datetime.datetime.strptime(lastWin,'%Y-%m-%d').date()
	print "Last game is ", daysOld, " days old. From ", lastWin

	if (datetime.datetime.today().weekday()==0 and daysOld.days <= 3) or (datetime.datetime.today().weekday()<=4 and daysOld.days <= 1):
		tweetProb(features)
		
		probs = predict_all(X_scaled,model)
		store_predictions(d4,probs)

        ##########
        # validate previous predictions!
        validate()




if __name__ == "__main__":
	main()
