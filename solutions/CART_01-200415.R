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

# Create a corpus from the headline, Snippet and Abstract variables.

CorpusHeadline = Corpus(VectorSource(c(NewsTrain$Headline, NewsTest$Headline)))
CorpusSnippet = Corpus(VectorSource(c(NewsTrain$Snippet, NewsTest$Snippet)))


CorpusHeadline = tm_map(CorpusHeadline, tolower)
CorpusSnippet = tm_map(CorpusSnippet, tolower)

CorpusHeadline = tm_map(CorpusHeadline, PlainTextDocument)
CorpusSnippet = tm_map(CorpusSnippet, PlainTextDocument)

CorpusHeadline = tm_map(CorpusHeadline, removePunctuation)
CorpusSnippet = tm_map(CorpusSnippet, removePunctuation)

CorpusHeadline = tm_map(CorpusHeadline, removeWords, stopwords("english"))
CorpusSnippet = tm_map(CorpusSnippet, removeWords, stopwords("english"))

CorpusHeadline = tm_map(CorpusHeadline, stemDocument)
CorpusSnippet = tm_map(CorpusSnippet, stemDocument)


dtmHeadline = DocumentTermMatrix(CorpusHeadline)
dtmSnippet = DocumentTermMatrix(CorpusSnippet)

dtmHeadline = removeSparseTerms(dtmHeadline, 0.995)
dtmSnippet = removeSparseTerms(dtmSnippet, 0.995)

dtmHeadline = as.data.frame(as.matrix(dtmHeadline))
dtmSnippet = as.data.frame(as.matrix(dtmSnippet))

colnames(dtmHeadline) = paste0("H", colnames(dtmHeadline))
colnames(dtmSnippet) = paste0("S", colnames(dtmSnippet))

dtm = cbind(dtmHeadline, dtmSnippet)


dtmTrain = head(dtm, nrow(NewsTrain))
dtmTest = tail(dtm, nrow(NewsTest))


dtmTrain$Popular = NewsTrain$Popular

dtmTrain$WordCount = NewsTrain$WordCount
dtmTest$WordCount = NewsTest$WordCount

dtmTrain$NewsDesk = NewsTrain$NewsDesk
dtmTest$NewsDesk = NewsTest$NewsDesk

dtmTrain$SectionName = NewsTrain$SectionName
dtmTest$SectionName = NewsTest$SectionName

dtmTrain$SubsectionName  = NewsTrain$SubsectionName 
dtmTest$SubsectionName  = NewsTest$SubsectionName

dtmTrain$Weekday  = NewsTrain$Weekday 
dtmTest$Weekday  = NewsTest$Weekday

dtmTrain$Hour  = NewsTrain$hour 
dtmTest$Hour  = NewsTest$hour


library(rpart)
library(rpart.plot)
blogCART = rpart(Popular~., data=dtmTrain, method="class")

PredTest = predict(blogCART, newdata=dtmTest)[,2]

library(ROCR)
PredTrain = predict(blogCART, data=dtmTrain)[,2]
predROCR = prediction(PredTrain, dtmTrain$Popular)
perfROCR = performance(predROCR, "tpr", "fpr")
plot(perfROCR, colorize=TRUE)
performance(predROCR, "auc")@y.values

# Now we can prepare our submission file for Kaggle:

MySubmission = data.frame(UniqueID = NewsTest$UniqueID, Probability1 = PredTest)

write.csv(MySubmission, "SubmissionCAR_01.csv", row.names=FALSE)
