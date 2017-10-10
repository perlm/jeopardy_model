# this is for lambda!
from bs4 import BeautifulSoup
import subprocess, requests, string, time, re, os, sys,datetime, pickle, bz2
import boto3

# for aws lambda:
#  sudo pip install requests datetime pip -t ~/jeopardy_model/aws
#  then need to grant permissions for lambda job to read/write s3 bucket, which is not a default option and surprisingly difficult...
#  basically, good luck asshole... ;-)
#  https://support.asperasoft.com/hc/en-us/articles/216129328-IAM-role-permissions-for-S3-buckets


print('Loading function')


def getRawData():
        # check csv to find most recent game in my data, and search for new games to scrape and add in

        #nameGenderDict, nameAgeDict = getNameDictionary() 
        getNameDictionary()

        # find last entry in data file
        #fil = '{0}/jeopardy_model/data/raw.data'.format(os.path.expanduser("~"))
        fil = '/tmp/raw.data'
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
        end     = start
        g       = start
        while g <= end+10:
                epInfo = getEpisodeInfo(g)
                if epInfo is not None:
                        f1.write(epInfo)
                        end = g
                g += 1


def getNameDictionary():
        global nameGenderDict
        global nameAgeDict
        with bz2.BZ2File("/tmp/nameGend.pickle","r") as f:
                        nameGenderDict = pickle.load(f)
        with bz2.BZ2File("/tmp/nameAge.pickle","r") as f:
                        nameAgeDict = pickle.load(f)


def getSoup(url):
        # scraping
        while True:
                try:
                        r = requests.get(url)
                        break
                except requests.ConnectionError:time.sleep(1)
        return BeautifulSoup(r.text, 'html.parser')

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
        players         = []
        winningDays     = None
        winningDollars  = None
        career          = None
        location        = None
        returnChamp     = None

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



def lambda_handler(event, context):

    BUCKET_NAME = 'jeopardydata' # replace with your bucket name

    # could switch this to read file, rather than download it.
    s3 = boto3.resource('s3')
    s3.Bucket(BUCKET_NAME).download_file('raw.data', '/tmp/raw.data')
    s3.Bucket(BUCKET_NAME).download_file('nameAge.pickle', '/tmp/nameAge.pickle')
    s3.Bucket(BUCKET_NAME).download_file('nameGend.pickle', '/tmp/nameGend.pickle')

    getRawData()

    s3.meta.client.upload_file('/tmp/raw.data', BUCKET_NAME, 'raw.data')
    
