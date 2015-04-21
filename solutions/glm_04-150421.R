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

HeadlineWordsTrain$Abstract = NewsTrain$Abstract
HeadlineWordsTest$Abstract = NewsTest$Abstract

# Create a corpus from the Snippet variable.

CorpusAbstract = Corpus(VectorSource(c(HeadlineWordsTrain$Abstract, HeadlineWordsTest$Abstract)))

CorpusAbstract = tm_map(CorpusAbstract, tolower)

CorpusAbstract = tm_map(CorpusAbstract, PlainTextDocument)

CorpusAbstract = tm_map(CorpusAbstract, removePunctuation)

CorpusAbstract = tm_map(CorpusAbstract, removeWords, stopwords("english"))

CorpusAbstract = tm_map(CorpusAbstract, stemDocument)


dtm_Abstract = DocumentTermMatrix(CorpusAbstract)

sparse = removeSparseTerms(dtm_Abstract, 0.99)

AbstratcWords = as.data.frame(as.matrix(sparse))

colnames(AbstratcWords) = make.names(colnames(AbstratcWords))

AbstratcWordsTrain = head(AbstratcWords, nrow(HeadlineWordsTrain))

AbstratcWordsTest = tail(AbstratcWords, nrow(HeadlineWordsTest))

AbstratcWordsTrain$Popular = NewsTrain$Popular

AbstratcWordsTrain$WordCount = NewsTrain$WordCount
AbstratcWordsTest$WordCount = NewsTest$WordCount

AbstratcWordsTrain$NewsDesk = NewsTrain$NewsDesk
AbstratcWordsTest$NewsDesk = NewsTest$NewsDesk

AbstratcWordsTrain$SectionName = NewsTrain$SectionName
AbstratcWordsTest$SectionName = NewsTest$SectionName

AbstratcWordsTrain$SubsectionName  = NewsTrain$SubsectionName 
AbstratcWordsTest$SubsectionName  = NewsTest$SubsectionName

AbstratcWordsTrain$Weekday  = NewsTrain$Weekday 
AbstratcWordsTest$Weekday  = NewsTest$Weekday

AbstratcWordsTrain$Hour  = NewsTrain$hour 
AbstratcWordsTest$Hour  = NewsTest$hour


# Make glm model
AbstractWordsLog = glm(Popular ~.-archiv-bill-book-christma-clip-collect-email-famili-former-found-herald-newsroom-pari-photo-secur-servic-thursday-tribun-use, data=AbstratcWordsTrain, family=binomial)


PredTest = predict(AbstractWordsLog, newdata=AbstratcWordsTest, type="response")

library(ROCR)
PredTrain = predict(AbstractWordsLog, data=AbstratcWordsTrain, type="response")
predROCR = prediction(PredTrain, AbstratcWordsTrain$Popular)
perfROCR = performance(predROCR, "tpr", "fpr")
plot(perfROCR, colorize=TRUE)
performance(predROCR, "auc")@y.values

# Now we can prepare our submission file for Kaggle:

MySubmission = data.frame(UniqueID = NewsTest$UniqueID, Probability1 = PredTest)

write.csv(MySubmission, "Submissionglm_08.csv", row.names=FALSE)
