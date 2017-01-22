#!/usr/bin/python

##
# Take the most recent prediction and tweet it, using the python api
##

from twitter import *
import ConfigParser

def tweetProb(features):
	tweet = '{n} is a {day}-day champ with ${dol} winnings: #Jeopardy model predicts {p}% prob to win! (as of {date}) https://hastydata.wordpress.com/2016/11/26/modeling-jeopardy-revisited/'.format(n=str(features['name']), day=str(features['days']),dol=str(features['dollars']),p=int(100*round(float(features['prob']),2)),date=str(features['date']) )

	config = ConfigParser.ConfigParser()
	config.read('/home/jason/.python_keys.conf')

	t = Twitter(auth=OAuth(token=config.get('twitter','token'), token_secret=config.get('twitter','token_secret'), consumer_key=config.get('twitter','consumer_key'), consumer_secret=config.get('twitter','consumer_secret')))
	t.statuses.update(status=tweet)

	print "Tweet Sent:\n%s" % (str(tweet))


if __name__ == '__main__':
	features = {}
	tweetProb(features)



