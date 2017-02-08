#!/usr/bin/python

##
# Take the most recent prediction and tweet it, using the python api
##

from twitter import *
import pandas as pd
import ConfigParser, os, datetime

def tweetProb(features):
	tweet = '{n} is a {day}-day champ with ${dol} winnings: #Jeopardy model predicts {p}% prob to win! (as of {date}) https://hastydata.wordpress.com/2016/11/26/modeling-jeopardy-revisited/'.format(n=str(features['name']), day=str(features['days']),dol=str(features['dollars']),p=int(100*round(float(features['prob']),2)),date=str(features['date']) )

	config = ConfigParser.ConfigParser()
	config.read('{0}/.python_keys.conf'.format(os.path.expanduser("~")))

	t = Twitter(auth=OAuth(token=config.get('twitter','token'), token_secret=config.get('twitter','token_secret'), consumer_key=config.get('twitter','consumer_key'), consumer_secret=config.get('twitter','consumer_secret')))
	t.statuses.update(status=tweet)

	print "Tweet Sent:\n%s" % (str(tweet))


def store_predictions(df,pred):
    ####
    # store predictions for future evaluation!
    ######
    fil = '{}/jeopardy_model/data/predictions.csv'.format(os.path.expanduser("~"))

    df['date'] = pd.to_datetime(df['date'])
    df['prediction'] = pred
    df = df.loc[(df['date']==datetime.date.today())]

    if (os.path.exists(fil)):
        prev = pd.read_csv(fil,delimiter=',',header=0)
        prev['date'] = pd.to_datetime(prev['date'])

        # check if this date is already in there    
        # the max of a datetime column is a timestamp object.
        if datetime.datetime.strptime(str(prev['date'].max())[:10],"%Y-%m-%d").date()<datetime.datetime.today().date():
            df.to_csv(fil, header=False, index=False,mode='a')
    else:
        df.to_csv(fil, header=True, index=False)


if __name__ == '__main__':
	features = {}
	tweetProb(features)



