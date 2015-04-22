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

HeadlineWordsTrain$Snippet = NewsTrain$Snippet
HeadlineWordsTest$Snippet = NewsTest$Snippet

# Create a corpus from the Snippet variable.

CorpusSnippet = Corpus(VectorSource(c(HeadlineWordsTrain$Snippet, HeadlineWordsTest$Snippet)))

CorpusSnippet = tm_map(CorpusSnippet, tolower)

CorpusSnippet = tm_map(CorpusSnippet, PlainTextDocument)

CorpusSnippet = tm_map(CorpusSnippet, removePunctuation)

CorpusSnippet = tm_map(CorpusSnippet, removeWords, stopwords("english"))

CorpusSnippet = tm_map(CorpusSnippet, stemDocument)


dtm_Snippet = DocumentTermMatrix(CorpusSnippet)

sparse = removeSparseTerms(dtm_Snippet, 0.99)

SnippetWords = as.data.frame(as.matrix(sparse))

colnames(SnippetWords) = make.names(colnames(SnippetWords))

SnippetWordsTrain = head(SnippetWords, nrow(HeadlineWordsTrain))

SnippetWordsTest = tail(SnippetWords, nrow(HeadlineWordsTest))

SnippetWordsTrain$Popular = NewsTrain$Popular

SnippetWordsTrain$WordCount = NewsTrain$WordCount
SnippetWordsTest$WordCount = NewsTest$WordCount

SnippetWordsTrain$NewsDesk = NewsTrain$NewsDesk
SnippetWordsTest$NewsDesk = NewsTest$NewsDesk

SnippetWordsTrain$SectionName = NewsTrain$SectionName
SnippetWordsTest$SectionName = NewsTest$SectionName

SnippetWordsTrain$SubsectionName  = NewsTrain$SubsectionName 
SnippetWordsTest$SubsectionName  = NewsTest$SubsectionName

SnippetWordsTrain$Weekday  = NewsTrain$Weekday 
SnippetWordsTest$Weekday  = NewsTest$Weekday

SnippetWordsTrain$Hour  = NewsTrain$hour 
SnippetWordsTest$Hour  = NewsTest$hour


# Make CART model
library(rpart)
library(rpart.plot)
SnippetWordsCART = rpart(Popular ~., data=SnippetWordsTrain, method="class")


PredTest = predict(SnippetWordsCART, newdata=SnippetWordsTest, type="prob")[,2]

library(ROCR)
PredTrain = predict(SnippetWordsCART, data=SnippetWordsTrain, type="prob")[,2]
predROCR = prediction(PredTrain, SnippetWordsTrain$Popular)
perfROCR = performance(predROCR, "tpr", "fpr")
plot(perfROCR, colorize=TRUE)
performance(predROCR, "auc")@y.values

# Now we can prepare our submission file for Kaggle:

MySubmission = data.frame(UniqueID = NewsTest$UniqueID, Probability1 = PredTest)

write.csv(MySubmission, "SubmissionCART_02.csv", row.names=FALSE)
