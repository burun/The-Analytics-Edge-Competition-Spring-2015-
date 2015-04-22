# Read the files

NewsTrain = read.csv("NYTimesBlogTrain.csv", stringsAsFactors=FALSE)

NewsTest = read.csv("NYTimesBlogTest.csv", stringsAsFactors=FALSE)

# Bind the Train and Test
BindTrain = NewsTrain
BindTest = NewsTest
BindTrain$Popular = NULL
BindTrain$UniqueID = NULL
BindTest$UniqueID = NULL
NewsAll = rbind(BindTrain, BindTest)

# Convert pubdate

NewsAll$PubDate = strptime(NewsAll$PubDate, "%Y-%m-%d %H:%M:%S")
NewsAll$Weekday = NewsAll$PubDate$wday
NewsAll$hour = NewsAll$PubDate$hour
NewsAll$PubDate = NULL

# Remove "Abstract" variable
NewsAll$Abstract = NULL
str(NewsAll)

# Create a corpus from the headline, Snippet and Abstract variables.

library(tm)

CorpusHeadline = Corpus(VectorSource(NewsAll$Headline))
CorpusSnippet = Corpus(VectorSource(NewsAll$Snippet))

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

dtmHeadline = removeSparseTerms(dtmHeadline, 0.99)
dtmSnippet = removeSparseTerms(dtmSnippet, 0.99)

dtmHeadline = as.data.frame(as.matrix(dtmHeadline))
dtmSnippet = as.data.frame(as.matrix(dtmSnippet))

colnames(dtmHeadline) = paste0("H", colnames(dtmHeadline))
colnames(dtmSnippet) = paste0("S", colnames(dtmSnippet))

dtm = cbind(dtmHeadline, dtmSnippet, row.names=NULL)


dtm$NewsDesk = as.factor(NewsAll$NewsDesk)
dtm$SectionName = as.factor(NewsAll$SectionName)
dtm$SubsectionName = as.factor(NewsAll$SubsectionName)
dtm$WordCount = log(NewsAll$WordCount+1)
dtm$Weekday = NewsAll$Weekday
dtm$hour = NewsAll$hour


# Spilte train and test set
dtmTrain = head(dtm, nrow(NewsTrain))
dtmTest = tail(dtm, nrow(NewsTest))

# Add "Popular" variable to train set
dtmTrain$Popular = as.factor(NewsTrain$Popular)

# Make glm model
library(randomForest)
blogRF = randomForest(Popular~.,data=dtmTrain,type="prob",ntree=1000,nodesize=5)


PredTest = predict(blogRF, newdata=dtmTest, type="prob")[,2]

library(ROCR)
PredTrain = predict(blogRF, data=dtmTrain, type="prob")[,2]
predROCR = prediction(PredTrain, dtmTrain$Popular)
perfROCR = performance(predROCR, "tpr", "fpr")
plot(perfROCR, colorize=TRUE)
performance(predROCR, "auc")@y.values

# Now we can prepare our submission file for Kaggle:

MySubmission = data.frame(UniqueID = NewsTest$UniqueID, Probability1 = PredTest)

write.csv(MySubmission, "SubmissionRF_02.csv", row.names=FALSE)
