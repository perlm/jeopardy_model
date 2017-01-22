#!/usr/bin/python

##
# This is the master script which will call functions from the other scripts.
# Currently setup to check for new data, scrape, model, and publish.
# scheduled to run on cron.
## 

from getData import *
from buildModel import *
from tweetIt import *
import datetime

def main():

	# download 
	getRawData()

	# model - df is from file. df2 is with features. x, scaled, fixed are after processing.
        d = readRawFile()
        d2 = constructFeatures(d)
        X, X_scaled, Y, scaler, X_fix = processData(d2)
        model = buildLogisticModel(X_scaled,Y,X_fix)

	# predict - df3 is with additional row for predictions. then process in exact same way.
	features = getCurrentStatus()
	d3 = addRow(d,features)
        d4 = constructFeatures(d3)
        X, X_scaled, Y, scaler, X_fix = processData(d4,scaler)
	prob = predict(X_scaled,model)
	features['prob'] = prob

	# tweet it!
	daysOld = datetime.date.today() - datetime.datetime.strptime(features['date'],'%Y-%m-%d').date()
	print "Last game is ", daysOld, " days old. From ", features['date']

	if (datetime.datetime.today().weekday()==0 and daysOld.days <= 3) or (datetime.datetime.today().weekday()<=4 and daysOld.days <= 1):
		tweetProb(features)



if __name__ == "__main__":
	main()
