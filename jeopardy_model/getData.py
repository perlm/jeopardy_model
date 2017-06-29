# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import requests, string, time, re, os, sys,datetime, bz2, pickle

#####
# The objective of this script is to scrape data from the Jeopardy archive so that it can be analyzed and I can make a predictive model.
#####

def getNameDictionaryCSV():
	# I separately have a dictionary of gender and age by name. Pull in this info for modeling.
	# make these global, so I don't need to remember to pass them around or reread the file
	global nameGenderDict
	global nameAgeDict
	nameGenderDict={}
	nameAgeDict={}
	with open('{0}/jeopardy_model/data/namesAverage.data'.format(os.path.expanduser("~")),'r') as f1:
		for line in f1:
			cols = line.split()
			nameGenderDict[str(cols[0])]=str(cols[1])
			nameAgeDict[str(cols[0])]=float(cols[2])
	#return nameGenderDict, nameAgeDict

def getNameDictionary():
	global nameGenderDict
	global nameAgeDict
	with bz2.BZ2File("{}/jeopardy_model/model_pickles/nameGend.pickle".format(os.path.expanduser("~")),"r") as f:
                        nameGenderDict = pickle.load(f)
	with bz2.BZ2File("{}/jeopardy_model/model_pickles/nameAge.pickle".format(os.path.expanduser("~")),"r") as f:
                        nameAgeDict = pickle.load(f)


def getSoup(url):
	# scraping
	while True:
		try:
			r = requests.get(url)
			break
		except requests.ConnectionError:time.sleep(1)
	return BeautifulSoup(r.text, 'html.parser')


def getRawData():
	# check csv to find most recent game in my data, and search for new games to scrape and add in

	#nameGenderDict, nameAgeDict = getNameDictionary() 
	getNameDictionary() 

	# find last entry in data file
	fil = '{0}/jeopardy_model/data/raw.data'.format(os.path.expanduser("~"))
        if os.path.exists(fil):
                with open(fil,'r') as f1:
                        for line in f1:
				start=int(line.split(',')[0])+1
                f1 = open(fil,'a+')
        else:
                f1 = open(fil,'w')
		start=0

	# look for new games in database
	# check 10 past most recent game
	end	= start
	g	= start
	while g <= end+10:
		epInfo = getEpisodeInfo(g)
		if epInfo is not None:
			f1.write(epInfo)
			end = g
		g += 1



def getEpisodeInfo(g):
	# goes though each game and pulls out features for modeling.

	# download game and check that it's valid
	baseURL1='http://www.j-archive.com/showgame.php?game_id={0}'
	skips = ["Tournament","Kids","College","School Week","Celebrity","Million Dollar Masters","Super Jeopardy","Three-way tie at zero","didn't have a winner"]

	url = (baseURL1.format(g))
	r = getSoup(url)
	errorCheck = r.find('p', attrs={"class": "error"})
	tourCheck = r.find('div', attrs={"id":"game_comments"})
		
	if errorCheck:
		print "nada! %d" % g
		return None
	if any(s in tourCheck.get_text() for s in skips):return None
	if g==504 or g==1088 or g==1092 or g==1309:return None


	# build out feature matrix
	players		= []
	winningDays	= None
	winningDollars	= None
	career		= None
	location	= None
	returnChamp	= None

	# pull out properties for model!
	for p in r.findAll('p', attrs={"class": "contestants"}):
		intro= p.get_text().replace(',','')
		prevWin = (re.search(r'an? (.+) (originally )?from (.+) \(whose ([0-9]+)-day cash winnings total \$([0-9]+)\)$', unicode(intro), re.M|re.I))
		name = p.find('a')

		if prevWin:
			career=prevWin.group(1)
			location=prevWin.group(3)
			winningDays=int(prevWin.group(4))
			winningDollars=int(prevWin.group(5))
			returnChamp = name.get_text().encode('ascii', 'ignore')
		players.append(name.get_text().encode('ascii', 'ignore'))
		if len(players)==3:continue
	players = list(reversed(players))

	if (not winningDays) or (not winningDollars):
		print "No previous winner %d" % g
		return None

	# get episode date
	title = r.find('title').get_text()
	showNumber = (re.search(r'Show #([0-9]*), aired (.*)$', unicode(title), re.M|re.I))
	if showNumber:
		gameNumber = int(showNumber.group(1))
		date = showNumber.group(2).encode('ascii', 'ignore')
	else:
		print "Error: Not valid %d" % g
		return None


	# now look at final scores to see who won!
	scores=[]
	for h3 in r.find_all('h3'):
		if h3.get_text() == 'Final scores:':
			following = h3.next.next.next
			for s in following.find_all('td', attrs={"class":['score_positive', 'score_negative']}):
				dollar= int(s.get_text().replace(',','').replace('$',''))
				if 'positive' in str(s):scores.append(dollar)
				elif 'negative' in str(s):scores.append(dollar*-1)
	if (len(scores)!=3):
		print "Error: no score data %d" % g
		return None

	# identify winner - make sure it matches scores
	# (the prev champ is listed first)
	# winner = 1 success, -1 defeat!
	winner=None
	for w in r.find_all('td', attrs={"class": "score_remarks"}):
		if ('champion' in w.get_text()):
			winner=1
			break
		elif ('place' in w.get_text()):
			winner=-1
			break
	else:
		print "No winner found!", g
		return None

	# check with scores for consistency
	maxScore=-1
	win=None
	for s in xrange(3):
		if (scores[s]>maxScore):
			maxScore=scores[s]
			win=s
	if (winner==1) and (win!=0):
		print "Error: not champion!", g
		return None
	if (winner==-1) and (win==0):
		print "Error: not challenger!", g
		return None



	# if at this point, everything looks valid, then add it to data file!
	print "Adding game", g

	# get some not very helpful demographic info on returning champ
	first = returnChamp.split()[0]
	gender = None
	age = None
	if first in nameGenderDict:gender = nameGenderDict[first]
	else:gender = 'U'
	if first in nameAgeDict:age = round(nameAgeDict[first],1)
	else:age = 36.8

	epInfo = '{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}\n'.format(g, gameNumber, date, winningDays, winningDollars, winner, gender, age, returnChamp.replace(',',' '), career.replace(',',' '), location.replace(',',' ') )
	return epInfo



def getMostRecentLocal():
	# the format the new champ is different- there is no existing page to scrape yet.
	# for the purposes of prediction, I want what the new champ in the same format.

	getNameDictionary()

	maxDate = '2004-10-06'
	maxDate = datetime.datetime.strptime(maxDate,'%Y-%m-%d')
	maxEp 	= None
	entry	= None
	with open('{0}/jeopardy_model/data/raw.data'.format(os.path.expanduser("~")),'r') as f1:
		for line in f1:
			l = line.split(',')
			if datetime.datetime.strptime(l[2],'%Y-%m-%d') > maxDate:
				maxEp = l[0]
				entry = l
				maxDate = datetime.datetime.strptime(l[2],'%Y-%m-%d')

	baseURL1='http://www.j-archive.com/showgame.php?game_id={0}'
	url = (baseURL1.format(maxEp))
	return url
	
def getCurrentStatus(url):
	# the format the new champ is different- there is no existing page to scrape yet.
	# for the purposes of prediction, I want what the new champ in the same format.

	getNameDictionary()
	r = getSoup(url)

	# check that it's a valid ep
        skips = ["Tournament","Kids","College","School Week","Celebrity","Million Dollar Masters","Super Jeopardy","Three-way tie at zero","didn't have a winner"]
        errorCheck = r.find('p', attrs={"class": "error"})
        tourCheck = r.find('div', attrs={"id":"game_comments"})
        if errorCheck:return None
        if any(s in tourCheck.get_text() for s in skips):return None


        # pull out properties pregame
        for p in r.findAll('p', attrs={"class": "contestants"}):
		intro= p.get_text().replace(',','')
                prevWin = (re.search(r'an? (.+) (originally )?from (.+) \(whose ([0-9]+)-day cash winnings total \$([0-9]+)\)$', unicode(intro), re.M|re.I))
                name = p.find('a')

                if prevWin:
                        career=prevWin.group(1)
                        location=prevWin.group(3)
                        winningDays=int(prevWin.group(4))
                        winningDollars=int(prevWin.group(5))
                        returnChamp = name.get_text().encode('ascii', 'ignore')

        if (not winningDays) or (not winningDollars):
                print "No previous winner %d" % g
                return None


        # get episode date
        title = r.find('title').get_text()
        showNumber = (re.search(r'Show #([0-9]*), aired (.*)$', unicode(title), re.M|re.I))
        if showNumber:
                gameNumber = int(showNumber.group(1))
                date = showNumber.group(2).encode('ascii', 'ignore')
        else:
                print "Error: Not valid %d" % g
                return None

	# look at final scores
	scores=[]
	for h3 in r.find_all('h3'):
		if h3.get_text() == 'Final scores:':
			following = h3.next.next.next
			for s in following.find_all('td', attrs={"class":['score_positive', 'score_negative']}):
				dollar= int(s.get_text().replace(',','').replace('$',''))
				if 'positive' in str(s):scores.append(dollar)
				elif 'negative' in str(s):scores.append(dollar*-1)

	assert len(scores)==3

	maxScore=-1
	win=None
	for s in xrange(3):
		if (scores[s]>maxScore):
			maxScore=scores[s]
			win=s

	# changing from loading from csv, to pulling direct, to avoid storing files locally
	#assert ((int(entry[5])==1 and win==0) or (int(entry[5])==-1 and win!=0))
	#if int(entry[5])==1:
	if win==0:
		winningDays 	= winningDays +1 #int(entry[3]) + 1
		winningDollars 	= winningDollars + maxScore #int(entry[4]) + maxScore
		#career 		= entry[9]
		#location	= entry[10]
		name		= returnChamp #entry[8]
		first 		= name.split()[0]
	else:
		winningDays	= 1
		winningDollars	= maxScore
		
		# at the top (where names/careers are) champ is last
		# at the bottom (where final scores are) champ is first
		# if win==0 then it's not here. if win==1 then fine. if win==2, then set to zer0.

		if win==2:win=0

		x=0
		for p in r.findAll('p', attrs={"class": "contestants"}):
			intro= p.get_text().replace(',','')
			contestant = (re.search(r'an? (.+) (originally )?from (.+)', unicode(intro), re.M|re.I))
			namerow = p.find('a')
			if contestant:
				career	= contestant.group(1)
				location= contestant.group(3)
				name 	= namerow.get_text().encode('ascii', 'ignore')
				first 	= name.split()[0]
			if win==x:break
			else:x+=1

	#date of last win
	lastWin = date #entry[2]
	# today's date
	#date = datetime.date.today()

	gender = None
	age = None
	if first in nameGenderDict:gender = nameGenderDict[first]
	else:gender = 'U'
	if first in nameAgeDict:age = round(nameAgeDict[first],1)
	else:age = 36.8

        features = {'date':date,'days':winningDays,'dollars':winningDollars,'gender':gender,'age':age,'name':name.replace(',',' '),'career':career.replace(',',' '),'location':location.replace(',',' ')}

	return features, lastWin

def getMostRecentSoup():
	# instead of referring to dataset, just scrape the most recent page.

	u = 'http://www.j-archive.com/listseasons.php'
	soup = getSoup(u)
	# get most recent season page.
	for link in soup.findAll('a', attrs={'href': re.compile("^http://")}):
    		if link.has_attr('href'):
			if link['href'] =='http://www.j-archive.com':
				continue
			l = link['href']
			break

	# get list of games in order.
	soup = getSoup(l)
	gamePages = [] 
	for link in soup.findAll('a', attrs={'href': re.compile("^http://")}):
		if link.has_attr('href'):
			if 'game' in link['href']:
				gamePages.append(link['href'])

	# This returns list! Other function returns most recent non-tournament!
	return gamePages


if __name__ == '__main__':
	#getRawData()
	getNameDictionary()


