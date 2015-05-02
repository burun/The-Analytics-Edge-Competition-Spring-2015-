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

# Inpute missing values
NewsAll$NewsDesk[NewsAll$NewsDesk=="" & NewsAll$SectionName=="Opinion"] = "OpEd"
NewsAll$NewsDesk[NewsAll$NewsDesk=="" & NewsAll$SectionName=="Arts"] = "Culture"
NewsAll$NewsDesk[NewsAll$NewsDesk=="" & NewsAll$SectionName=="Business Day"] = "Business"
NewsAll$NewsDesk[NewsAll$NewsDesk=="" & NewsAll$SectionName=="Crosswords/Games"] = "Business"
NewsAll$NewsDesk[NewsAll$NewsDesk=="" & NewsAll$SectionName=="Travel"] = "Travel"
NewsAll$NewsDesk[NewsAll$NewsDesk=="" & NewsAll$SectionName=="Health"] = "Science"
NewsAll$NewsDesk[NewsAll$NewsDesk=="" & NewsAll$SectionName=="N.Y. / Region"] = "Metro"
NewsAll$NewsDesk[NewsAll$NewsDesk=="" & NewsAll$SectionName=="Technology"] = "Business"
NewsAll$NewsDesk[NewsAll$NewsDesk=="" & NewsAll$SectionName=="World"] = "Foreign"
NewsAll$NewsDesk[NewsAll$NewsDesk=="" & NewsAll$SubsectionName=="Dealbook"] = "Business"

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
blogRF = randomForest(Popular~WordCount+SectionName+NewsDesk+hour+Weekday+SubsectionName+Aweek+Ashare+Atime+Hnew+Ayear+Areport+Hweek+Hreport+Hyork+Awill+Afashion+Aintern+Ayork+Afirst+Acompani+Ashow+Aarticl+Aday+Acan,data=dtmTrain,type="prob",ntree=1500,nodesize=1)
# varImpPlot(blogRF)

PredTest = predict(blogRF, newdata=dtmTest, type="prob")[,2]

library(ROCR)
PredTrain = predict(blogRF, data=dtmTrain, type="prob")[,2]
predROCR = prediction(PredTrain, dtmTrain$Popular)
perfROCR = performance(predROCR, "tpr", "fpr")
plot(perfROCR, colorize=TRUE)
performance(predROCR, "auc")@y.values
# 0.9380701

# Now we can prepare our submission file for Kaggle:

MySubmission = data.frame(UniqueID = NewsTest$UniqueID, Probability1 = PredTest)

write.csv(MySubmission, "SubmissionRF_16.csv", row.names=FALSE)
