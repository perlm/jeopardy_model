
######
# Jeopary Model
# Goal is to predict probability that returning champ will win match on jeopardy given the information in the Jeopardy archive.
# http://j-archive.com/
########

setwd("/home/jason/Dropbox/gitHubable/jeopardy/modeling_v2")

library(ROCR)
library(ggplot2)
library(glmnet)
library(dygraphs)
library(xts)
library(reshape2)
library(Hmisc)
library(lubridate)



df = read.table("data/raw.data", header=FALSE, sep=",", quote='', nrows=250000)
colnames(df) <- c("index", "gameID","date","ConsecutiveWins","TotalDollars","Outcome","Gender","Age","Name","Occuppation","Location")

###########################
# restructure/add features!

# convert outcome to zero/one
df$Outcome[df$Outcome==-1]<-0

#try restructuring data
df$TotalDollars_buckets[df$TotalDollars<=15000] <- 'a_lt15k'
df$TotalDollars_buckets[df$TotalDollars>15000 & df$TotalDollars<=35000] <- 'b_15-35k'
df$TotalDollars_buckets[df$TotalDollars>35000 & df$TotalDollars<=75000] <- 'c_35-75k'
df$TotalDollars_buckets[df$TotalDollars>75000 & df$TotalDollars<=150000] <- 'd_75-150k'
df$TotalDollars_buckets[df$TotalDollars>150000] <- 'e_gt150k'

df$Avg_Dollars <- df$TotalDollars/df$ConsecutiveWins
df$Avg_Dollars_buckets[df$Avg_Dollars<=10000] <- 'a_lt10k'
df$Avg_Dollars_buckets[df$Avg_Dollars>10000 & df$Avg_Dollars<=30000] <- 'b_10-30k'
df$Avg_Dollars_buckets[df$Avg_Dollars>30000] <- 'c_gt30k'


ggplot(data=df, aes(x=round(Avg_Dollars,digits=-3),y=Outcome)) +  
  stat_summary(fun.data="mean_cl_normal", size=1)+ #stat_summary(fun.y="mean", geom="line", size=1) +  
  xlab('Avg Winnings') + ylab('Win Rate') + xlim(c(0,40000))+ ylim(c(0,1))


ggplot(data=df, aes(x=round(Age,digits=0),y=Outcome)) +  stat_summary(fun.y="mean", geom="line", size=2) +  xlab('Age') + ylab('Win Rate')

df$age_bucket[df$Age<=35] <- 'a_lt35'
df$age_bucket[df$Age>35 & df$Age<=55] <- 'b_35-55'
df$age_bucket[df$Age>55] <- 'c_gt55'

df$TotalDollars_buckets <- as.factor(df$TotalDollars_buckets)
df$age_bucket <- as.factor(df$age_bucket)

df$ConsecutiveWins_capped <- df$ConsecutiveWins
df$ConsecutiveWins_capped[df$ConsecutiveWins>=10]<-10

ggplot(data=df, aes(x=ConsecutiveWins,y=Outcome)) +  stat_summary(fun.y="mean", geom="line", size=2) +  xlab('Wins') + ylab('Win Rate') +  xlim(c(1,20))

df$date <- as.Date(df$date)
df$year <- year(df$date)
df$year_relative <- 2017-df$year
df$decade <- substr(as.character(df$year),0,3)

ggplot(data=df, aes(x=year,y=Outcome)) +  stat_summary(fun.y="mean", geom="line", size=2) +  xlab('Year') + ylab('Win Rate')


# states
temp <- strsplit(as.character(df$Location),split=" ")
lastValue <- function(x) tail(x,n=1)
df$state<- unlist(lapply(as.array(temp),lastValue))

# arbitary city list  
df$location_2way <- ifelse(df$Location %in% c('New York New York','Los Angeles California','Chicago Illinois','Houston Texas','Philadelphia Pennsylvania'),'a_top5city','b_else')
#df$location_2way <- ifelse(df$Location %in% c('New York New York','Los Angeles California','Chicago Illinois'),'a_top3city','b_else')

# cities of champions
a <- subset(df, !duplicated(Name))
b <- as.data.frame(xtabs(a,formula=~Location))
c <- b[order(-b$Freq),]
d <- unlist(head(c$Location,n=20))
df$cityofchampions <- ifelse(df$Location %in% d,'championcity','else')

# occupation of champions -- lawyer is most common!
a <- subset(df, !duplicated(Name))
b <- as.data.frame(xtabs(a,formula=~Occuppation))
c <- b[order(-b$Freq),]
d <- unlist(head(c$Occuppation,n=4))

lawjobs <- c('attorney','lawyer','law student')
df$lawyer<-0
df$lawyer[grep(paste(lawjobs,collapse="|"),df$Occuppation,value=FALSE)]<-1
# head(subset(df,lawyer==1))

df$jobs<-'other'
df$jobs[grep(paste(lawjobs,collapse="|"),df$Occuppation,value=FALSE)]<-'law'
df$jobs[grep("student|Ph.D. candidate",df$Occuppation,value=FALSE)]<-'student'
df$jobs[grep("writer",df$Occuppation,value=FALSE)]<-'writer'
df$jobs[grep("home",df$Occuppation,value=FALSE)]<-'home'
df$jobs[grep("teach|professor|college instructor|librarian",df$Occuppation,value=FALSE)]<-'teacher'
df$jobs[grep("engineer|scien|software|programmer",df$Occuppation,value=FALSE)]<-'tech'


#########################
# Examine data
mean(as.numeric(df$Outcome))
aggregate(df$Outcome, list(Condition=df$ConsecutiveWins), length)
aggregate(df$Outcome, list(Condition=df$ConsecutiveWins), mean)

aggregate(df$Outcome, list(Condition=df$TotalDollars), length)
aggregate(df$Outcome, list(Condition=df$TotalDollars), mean)

aggregate(df$Outcome, list(Condition=df$decade), length)
aggregate(df$Outcome, list(Condition=df$decade), mean)

aggregate(df$Outcome, list(Condition=df$TotalDollars_buckets), length)
aggregate(df$Outcome, list(Condition=df$TotalDollars_buckets), mean)

aggregate(df$Outcome, list(Condition=df$Avg_Dollars_buckets), length)
aggregate(df$Outcome, list(Condition=df$Avg_Dollars_buckets), mean)

aggregate(df$Outcome, list(Condition=df$age_bucket), length)
aggregate(df$Outcome, list(Condition=df$age_bucket), mean)

aggregate(df$Outcome, list(Condition=df$Gender), length)
aggregate(df$Outcome, list(Condition=df$Gender), mean)


a <- merge(as.data.frame(xtabs(df,formula = Outcome ~ Location)/xtabs(df,formula=~Location)),
           as.data.frame(xtabs(df,formula=~Location)),by='Location')
b <- a[order(a$Freq.x),]
head(b[b$Freq.y>10,],n=10)
tail(b[b$Freq.y>10,],n=10)

aggregate(df$Outcome, list(Condition=df$state), length)
aggregate(df$Outcome, list(Condition=df$state), mean)

aggregate(df$Outcome, list(Condition=df$location_2way), length)
aggregate(df$Outcome, list(Condition=df$location_2way), mean)

aggregate(df$Outcome, list(Condition=df$cityofchampions), length)
aggregate(df$Outcome, list(Condition=df$cityofchampions), mean)

aggregate(df$Outcome, list(Condition=df$lawyer), length)
aggregate(df$Outcome, list(Condition=df$lawyer), mean)

aggregate(df$Outcome, list(Condition=df$jobs), length)
aggregate(df$Outcome, list(Condition=df$jobs), mean)



##################################################
###################################################3
#simplify data frame and setup for training  
#r <- names(df) %in% c('index','gameID','date','Name','pred')
#to <- df[!r]
#sapply(to,function(x)length(levels(as.factor(x))))
#bad <- sapply(to,function(x)length(levels(as.factor(x)))<2)
#to <- to[!bad]

to <- df

to$holdout <- runif(nrow(to)) > 0.8
train.data <- to[!to$holdout, ]
validate.data <- to[to$holdout, ]

# model formula
f <- formula(Outcome ~ ConsecutiveWins_capped + Avg_Dollars  + 
               ConsecutiveWins_capped:Avg_Dollars_buckets +
               Gender + age_bucket +
               cityofchampions + jobs +
               year_relative,data=train.data)

#optimize alpha as well!
mm <- sparse.model.matrix(f, data=train.data)
y <- train.data$Outcome

# a <- 10
# runs <- 10
# alphas <- seq(0, 1, by=1/a)
# aucs <- numeric(runs*(a+1))
# for (r in 1:runs){
#   for(i in 1:(a+1)){
#     cvfits <- cv.glmnet(mm[,-1], train.data$Outcome, alpha=alphas[i], family="binomial", standardize=TRUE, 
#       nlambda=10, type.measure="auc", nfold=200,# nfold=dim(mm)[1],
#       maxit=10000)
#     loc <- which(cvfits$lambda==cvfits$lambda.min)
#     n <- i + (r-1)*(a+1)
#     aucs[n] <- cvfits$cvm[loc]
#   }
# }
# 
# this <- data.frame(auc=aucs, alpha=rep(alphas,runs))
# ggplot(this, aes(x=alpha, y=auc)) + 
#   geom_point(shape=1) + geom_smooth() + ylab("CV AUC") + xlab("Alpha Value")
#   
# coef(cvfits, s=cvfits$lambda.min)    #model coefficients for lambda which minimizes error.

# get final model
getLambdaModel <- cv.glmnet(mm[,-1], train.data$Outcome, alpha=0.75, 
  family="binomial", standardize=TRUE, nfold=250,type.measure="auc", 
  nlambda=100, maxit=10000)
finalModel <- glmnet(mm[,-1], train.data$Outcome, alpha=0.75, 
  family="binomial", standardize=TRUE, 
  lambda=getLambdaModel$lambda.min, maxit=10000)

coef(finalModel)    #model coefficients for lambda which minimizes error.


mm <- model.matrix(f, data=validate.data)
validate.data$pred <- predict(finalModel, newx=mm[,-1], type="response", s = "lambda.min")
calc_AUC(validate.data$Outcome, validate.data$pred, "FALSE")
# 0.596


ggplot(validate.data,aes(x=pred,group=ConsecutiveWins_capped,fill=ConsecutiveWins_capped))+
  geom_histogram(binwidth=0.025) +
  xlim(c(0.25,0.75)) + xlab('Predicted Win Probability')+
  scale_fill_gradient(name="Previous Wins",limits=c(1, 5)) +
  ggtitle('Model Predictions by Previous Wins')

ggplot(validate.data,aes(y=pred,group=ConsecutiveWins_capped,x=ConsecutiveWins_capped))+
  geom_boxplot() +
  ylab('Predicted Win Probability') + xlab('Previous Wins')+
  ggtitle('Model Predictions by Previous Wins')+
  scale_x_continuous(breaks=1:10)




# retrain on whole set
mm <- model.matrix(f, data=df)
final <- glmnet(mm[,-1], df$Outcome, alpha=0.75, 
                     family="binomial", standardize=TRUE, 
                     lambda=getLambdaModel$lambda.min, maxit=10000)

# coef(final)    #model coefficients for lambda which minimizes error.

df$pred <- predict(final, newx=mm[,-1], type="response", s = "lambda.min")
calc_AUC(df$Outcome, df$pred, "FALSE")


a <- as.data.frame(xtabs(df,formula=Outcome~round(pred,1))/xtabs(df,formula=~round(pred,1)))
ggplot(a,aes(as.numeric(as.character(round.pred..1.)),y=Freq)) + geom_point(size=3,alpha=0.75)+geom_abline(intercept=0,slope=1)+
  xlab('Predicted Win Rate')+ylab('Actual Win Rate') + ggtitle('Model Sensitivity Analysis')



a <- as.data.frame(xtabs(df,formula=Outcome~round(pred,2))/xtabs(df,formula=~round(pred,2)))
ggplot(a,aes(as.numeric(as.character(round.pred..2.)),y=Freq)) + geom_point(size=3,alpha=0.75)+geom_abline(intercept=0,slope=1)+
  xlab('Predicted Win Rate')+ylab('Actual Win Rate')


###################33############33
# retry the whole thing, but taking out Jennings
train.data2 <- subset(train.data,Name!='Ken Jennings')

mm <- sparse.model.matrix(f, data=train.data2)
y <- train.data$Outcome

a <- 10
runs <- 10
alphas <- seq(0, 1, by=1/a)
aucs <- numeric(runs*(a+1))
for (r in 1:runs){
  for(i in 1:(a+1)){
    cvfits <- cv.glmnet(mm[,-1], train.data2$Outcome, alpha=alphas[i], family="binomial", standardize=TRUE, 
                        nlambda=10, type.measure="auc", nfold=200,# nfold=dim(mm)[1],
                        maxit=10000)
    loc <- which(cvfits$lambda==cvfits$lambda.min)
    n <- i + (r-1)*(a+1)
    aucs[n] <- cvfits$cvm[loc]
  }
}

this <- data.frame(auc=aucs, alpha=rep(alphas,runs))
ggplot(this, aes(x=alpha, y=auc)) + 
  geom_point(shape=1) + geom_smooth() + ylab("CV AUC") + xlab("Alpha Value")


# get final model
getLambdaModel <- cv.glmnet(mm[,-1], train.data2$Outcome, alpha=1., 
                            family="binomial", standardize=TRUE, nfold=250,type.measure="auc", 
                            nlambda=100, maxit=10000)
finalModel2 <- glmnet(mm[,-1], train.data2$Outcome, alpha=1., 
                     family="binomial", standardize=TRUE, 
                     lambda=getLambdaModel$lambda.min, maxit=10000)

coef(finalModel2,s="lambda.min")    #model coefficients for lambda which minimizes error.


mm <- model.matrix(f, data=validate.data)
validate.data$pred2 <- predict(finalModel2, newx=mm[,-1], type="response", s = "lambda.min")
calc_AUC(validate.data$Outcome, validate.data$pred2, "FALSE")
# 0.591

# excluding Jennings has only a small effect
ggplot(validate.data,aes(x=pred,y=pred2,color=ConsecutiveWins_capped))+geom_point()






##########
### previously-
mm <- sparse.model.matrix(f, data=train.data)
cvmn <- cv.glmnet(mm[,-1], train.data$Outcome, family="binomial", standardize=TRUE, 
      nlambda=50,  type.measure="auc", nfold=10,maxit=10000)
plot(cvmn)

coef(cvmn, s=cvmn$lambda.min)    #model coefficients for lambda which minimizes error.
calc_AUC(train.data$Outcome, predict(cvmn, newx=mm[,-1], type="response", s = "lambda.min"), "TRUE")
#old: in-sample AUC 0.61
#new: in-sample AUC 0.62

# predict for holdout
mm <- model.matrix(f, data=validate.data)
calc_AUC(validate.data$Outcome, predict(cvmn, newx=mm[,-1], type="response", s = "lambda.min"), "FALSE")
#out-of-sample AUC 0.58
#out-of-sample AUC 0.59

# predict for whole set
mm <- model.matrix(f, data=df)
df$pred <- predict(cvmn, newx=mm[,-1], type="response", s = "lambda.min")


# Visualize
ggplot(data=df,aes(pred))+geom_histogram(bins=10)+xlab('Predicted Probability')+ggtitle('Distribution of Predictions')
ggplot(data=df,aes(pred,fill=as.factor(TotalDollars_buckets)))+geom_histogram(binwidth=0.05,position="dodge")+xlab('Predicted Probability')+ggtitle('Distribution of Predictions')

ggplot(data=df,aes(x=date,y=pred,color=Name))+ggtitle('30 Years of Jeopardy!')+
  geom_line(size=1,alpha=0.75) + xlab('Date') + ylab('Predicted Win Prob')+scale_colour_discrete(guide=FALSE)


ts <- as.xts(df[c('date','Name','pred')], order.by=as.Date(df[c('date','Name','pred')]$date))
dygraph(ts,main="Model Prediction over Time") %>% dyHighlight(highlightSeriesBackgroundAlpha = 0.5, highlightCircleSize = 5, highlightSeriesOpts = list(strokeWidth = 1),hideOnMouseOut = FALSE) %>% dyLegend(show = "onmouseover") 


       



