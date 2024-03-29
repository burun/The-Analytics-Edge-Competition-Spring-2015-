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

# Create a corpus from the headline variable.

CorpusHeadline = Corpus(VectorSource(c(NewsTrain$Headline, NewsTest$Headline)))

CorpusHeadline = tm_map(CorpusHeadline, tolower)

CorpusHeadline = tm_map(CorpusHeadline, PlainTextDocument)

CorpusHeadline = tm_map(CorpusHeadline, removePunctuation)

CorpusHeadline = tm_map(CorpusHeadline, removeWords, stopwords("english"))

CorpusHeadline = tm_map(CorpusHeadline, stemDocument)


dtm = DocumentTermMatrix(CorpusHeadline)

sparse = removeSparseTerms(dtm, 0.99)

HeadlineWords = as.data.frame(as.matrix(sparse))

colnames(HeadlineWords) = make.names(colnames(HeadlineWords))

HeadlineWordsTrain = head(HeadlineWords, nrow(NewsTrain))

HeadlineWordsTest = tail(HeadlineWords, nrow(NewsTest))

HeadlineWordsTrain$Popular = NewsTrain$Popular

HeadlineWordsTrain$WordCount = NewsTrain$WordCount
HeadlineWordsTest$WordCount = NewsTest$WordCount

HeadlineWordsTrain$NewsDesk = NewsTrain$NewsDesk
HeadlineWordsTest$NewsDesk = NewsTest$NewsDesk

HeadlineWordsTrain$SectionName = NewsTrain$SectionName
HeadlineWordsTest$SectionName = NewsTest$SectionName

HeadlineWordsTrain$SubsectionName  = NewsTrain$SubsectionName 
HeadlineWordsTest$SubsectionName  = NewsTest$SubsectionName

HeadlineWordsTrain$Weekday  = NewsTrain$Weekday 
HeadlineWordsTest$Weekday  = NewsTest$Weekday

HeadlineWordsTrain$Hour  = NewsTrain$hour 
HeadlineWordsTest$Hour  = NewsTest$hour

HeadlineWordsLog = glm(Popular ~. -X2015+big+billion+daili+make+obama+pari+pictur+small+springsumm, data=HeadlineWordsTrain, family=binomial)


PredTest = predict(HeadlineWordsLog, newdata=HeadlineWordsTest, type="response")

library(ROCR)
PredTrain = predict(HeadlineWordsLog,data=HeadlineWordsTrain, type="response")
predROCR = prediction(PredTrain, HeadlineWordsTrain$Popular)
perfROCR = performance(predROCR, "tpr", "fpr")
plot(perfROCR, colorize=TRUE)
performance(predROCR, "auc")@y.values

# Now we can prepare our submission file for Kaggle:

MySubmission = data.frame(UniqueID = NewsTest$UniqueID, Probability1 = PredTest)

write.csv(MySubmission, "Submissionglm_05.csv", row.names=FALSE)
