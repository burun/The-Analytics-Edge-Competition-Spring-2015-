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
CorpusAbstract = Corpus(VectorSource(c(NewsTrain$Abstract, NewsTest$Abstract)))


CorpusHeadline = tm_map(CorpusHeadline, tolower)
CorpusSnippet = tm_map(CorpusSnippet, tolower)
CorpusAbstract = tm_map(CorpusAbstract, tolower)

CorpusHeadline = tm_map(CorpusHeadline, PlainTextDocument)
CorpusSnippet = tm_map(CorpusSnippet, PlainTextDocument)
CorpusAbstract = tm_map(CorpusAbstract, PlainTextDocument)

CorpusHeadline = tm_map(CorpusHeadline, removePunctuation)
CorpusSnippet = tm_map(CorpusSnippet, removePunctuation)
CorpusAbstract = tm_map(CorpusAbstract, removePunctuation)

CorpusHeadline = tm_map(CorpusHeadline, removeWords, stopwords("english"))
CorpusSnippet = tm_map(CorpusSnippet, removeWords, stopwords("english"))
CorpusAbstract = tm_map(CorpusAbstract, removeWords, stopwords("english"))

CorpusHeadline = tm_map(CorpusHeadline, stemDocument)
CorpusSnippet = tm_map(CorpusSnippet, stemDocument)
CorpusAbstract = tm_map(CorpusAbstract, stemDocument)


dtmHeadline = DocumentTermMatrix(CorpusHeadline)
dtmSnippet = DocumentTermMatrix(CorpusSnippet)
dtmAbstract = DocumentTermMatrix(CorpusAbstract)

dtmHeadline = removeSparseTerms(dtmHeadline, 0.99)
dtmSnippet = removeSparseTerms(dtmSnippet, 0.99)
dtmAbstract = removeSparseTerms(dtmAbstract, 0.99)

dtmHeadline = as.data.frame(as.matrix(dtmHeadline))
dtmSnippet = as.data.frame(as.matrix(dtmSnippet))
dtmAbstract = as.data.frame(as.matrix(dtmAbstract))

colnames(dtmHeadline) = paste0("H", colnames(dtmHeadline))
colnames(dtmSnippet) = paste0("S", colnames(dtmSnippet))
colnames(dtmAbstract) = paste0("A", colnames(dtmAbstract))

dtm = cbind(dtmHeadline, dtmSnippet, dtmAbstract)


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

# Make glm model
glmLog = glm(Popular ~., data=dtmTrain, family=binomial)


PredTest = predict(glmLog, newdata=dtmTest, type="response")

library(ROCR)
PredTrain = predict(glmLog, data=dtmTrain, type="response")
predROCR = prediction(PredTrain, dtmTrain$Popular)
perfROCR = performance(predROCR, "tpr", "fpr")
plot(perfROCR, colorize=TRUE)
performance(predROCR, "auc")@y.values

# Now we can prepare our submission file for Kaggle:

MySubmission = data.frame(UniqueID = NewsTest$UniqueID, Probability1 = PredTest)

write.csv(MySubmission, "Submissionglm_08.csv", row.names=FALSE)
