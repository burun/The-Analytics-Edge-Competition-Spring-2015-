# Read the files

NewsTrain = read.csv("NYTimesBlogTrain.csv", stringsAsFactors=FALSE)

NewsTest = read.csv("NYTimesBlogTest.csv", stringsAsFactors=FALSE)

# Convert pubdate

NewsTrain$PubDate = strptime(NewsTrain$PubDate, "%Y-%m-%d %H:%M:%S")
NewsTest$PubDate = strptime(NewsTest$PubDate, "%Y-%m-%d %H:%M:%S")

NewsTrain$Weekday = NewsTrain$PubDate$wday
NewsTest$Weekday = NewsTest$PubDate$wday

NewsTrain$hour = NewsTrain$PubDate$hour
NewsTest$hour = NewsTest$PubDate$hour

NewsTrain$Minute = NewsTrain$PubDate$min
NewsTest$Minute = NewsTest$PubDate$min

library(tm)

# Create a corpus from the headline variables.

CorpusHeadline = Corpus(VectorSource(c(NewsTrain$Headline, NewsTest$Headline)))

CorpusHeadline = tm_map(CorpusHeadline, tolower)

CorpusHeadline = tm_map(CorpusHeadline, PlainTextDocument)

CorpusHeadline = tm_map(CorpusHeadline, removePunctuation)

CorpusHeadline = tm_map(CorpusHeadline, removeWords, stopwords("english"))

CorpusHeadline = tm_map(CorpusHeadline, stemDocument)


dtmHeadline = DocumentTermMatrix(CorpusHeadline)

dtmHeadline = removeSparseTerms(dtmHeadline, 0.99)

dtmHeadline = as.data.frame(as.matrix(dtmHeadline))

colnames(dtmHeadline) = paste0("H", colnames(dtmHeadline))


dtmTrain = head(dtmHeadline, nrow(NewsTrain))
dtmTest = tail(dtmHeadline, nrow(NewsTest))


dtmTrain$Popular = as.factor(NewsTrain$Popular)

dtmTrain$LogWordCount = log(NewsTrain$WordCount)
dtmTest$LogWordCount = log(NewsTest$WordCount)

dtmTrain$Weekday  = as.numeric(NewsTrain$Weekday)
dtmTest$Weekday  = as.numeric(NewsTest$Weekday)

dtmTrain$Hour  = as.numeric(NewsTrain$hour)
dtmTest$Hour  = as.numeric(NewsTest$hour)

# Make RF model

library(randomForest)
blogRF = randomForest(Popular~.,data=dtmTrain, na.rm=TRUE)

PredTest = predict(blogRF, newdata=dtmTest, type="prob")[,2]

library(ROCR)
PredTrain = predict(blogRF, data=dtmTrain, type="prob")[,2]
predROCR = prediction(PredTrain, dtmTrain$Popular)
perfROCR = performance(predROCR, "tpr", "fpr")
plot(perfROCR, colorize=TRUE)
performance(predROCR, "auc")@y.values

# Now we can prepare our submission file for Kaggle:

MySubmission = data.frame(UniqueID = NewsTest$UniqueID, Probability1 = PredTest)

write.csv(MySubmission, "SubmissionRF_01.csv", row.names=FALSE)
