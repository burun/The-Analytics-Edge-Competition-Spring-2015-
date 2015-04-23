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

# Remove "Snippet" variable
NewsAll$Snippet = NULL
str(NewsAll)

# Create a corpus from the headline, Snippet and Abstract variables.

library(tm)

CorpusHeadline = Corpus(VectorSource(NewsAll$Headline))
CorpusAbstract = Corpus(VectorSource(NewsAll$Abstract))

CorpusHeadline = tm_map(CorpusHeadline, tolower)
CorpusAbstract = tm_map(CorpusAbstract, tolower)

CorpusHeadline = tm_map(CorpusHeadline, PlainTextDocument)
CorpusAbstract = tm_map(CorpusAbstract, PlainTextDocument)

CorpusHeadline = tm_map(CorpusHeadline, removePunctuation)
CorpusAbstract = tm_map(CorpusAbstract, removePunctuation)

CorpusHeadline = tm_map(CorpusHeadline, removeWords, stopwords("english"))
CorpusAbstract = tm_map(CorpusAbstract, removeWords, stopwords("english"))

CorpusHeadline = tm_map(CorpusHeadline, stemDocument)
CorpusAbstract = tm_map(CorpusAbstract, stemDocument)

dtmHeadline = DocumentTermMatrix(CorpusHeadline)
dtmAbstract = DocumentTermMatrix(CorpusAbstract)

dtmHeadline = removeSparseTerms(dtmHeadline, 0.97)
dtmAbstract = removeSparseTerms(dtmAbstract, 0.97)

dtmHeadline = as.data.frame(as.matrix(dtmHeadline))
dtmAbstract = as.data.frame(as.matrix(dtmAbstract))

colnames(dtmHeadline) = paste0("H", colnames(dtmHeadline))
colnames(dtmAbstract) = paste0("A", colnames(dtmAbstract))

dtm = cbind(dtmHeadline, dtmAbstract, row.names=NULL)


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
# blogRF = randomForest(Popular~.,data=dtmTrain,type="prob",ntree=1000,nodesize=5)

Yvals = dtmTrain$Popular
IndependentVars = dtmTrain
IndependentVars$Popular = NULL # Drop the Dependent variable column
library(caret)
fitControl <- trainControl(classProbs = TRUE, summaryFunction = twoClassSummary)
tr = train(IndependentVars, Yvals, method="rf", nodesize=4, ntree=1200, metric="ROC", trControl=fitControl)

PredTest = predict(tr$finalModel, newdata=dtmTest, type="prob")[,2]

library(ROCR)
PredTrain = predict(tr$finalModel, data=dtmTrain, type="prob")[,2]
predROCR = prediction(PredTrain, dtmTrain$Popular)
perfROCR = performance(predROCR, "tpr", "fpr")
plot(perfROCR, colorize=TRUE)
performance(predROCR, "auc")@y.values
# 0.9380701

# Now we can prepare our submission file for Kaggle:

MySubmission = data.frame(UniqueID = NewsTest$UniqueID, Probability1 = PredTest)

write.csv(MySubmission, "SubmissionRF_11.csv", row.names=FALSE)
