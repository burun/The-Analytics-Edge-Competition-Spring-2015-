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
dtmTrain$Popular = NewsTrain$Popular

library(caTools)
set.seed(777)
spl = sample.split(dtmTrain$Popular, SplitRatio = 0.5)
ensembleTrain = subset(dtmTrain, spl == TRUE)
blenderTrain = subset(dtmTrain, spl == FALSE)


labelName = 'Popular'
predictors = names(ensembleTrain)[names(ensembleTrain) != labelName]

library(caret)
myControl = trainControl(method='cv', number=3, returnResamp='none')

# test_model = train(blenderTrain[,predictors], blenderTrain[,labelName], method='rf', trControl=myControl)
# preds = predict(object=test_model, dtmTest[,predictors])

model_gbm <- train(ensembleTrain[,predictors], ensembleTrain[,labelName], method='gbm', trControl=myControl)
model_glm <- train(ensembleTrain[,predictors], ensembleTrain[,labelName], method='glm', trControl=myControl)
model_rf <- train(ensembleTrain[,predictors], ensembleTrain[,labelName], method='rf', trControl=myControl)
model_rpart <- train(ensembleTrain[,predictors], ensembleTrain[,labelName], method='rpart', trControl=myControl)

blenderTrain$gbm_PROB <- predict(object=model_gbm, blenderTrain[,predictors])
blenderTrain$glm_PROB <- predict(object=model_glm, blenderTrain[,predictors])
blenderTrain$rf_PROB <- predict(object=model_rf, blenderTrain[,predictors])
blenderTrain$rpart_PROB <- predict(object=model_rpart, blenderTrain[,predictors])

dtmTest$gbm_PROB <- predict(object=model_gbm, dtmTest[,predictors])
dtmTest$glm_PROB <- predict(object=model_glm, dtmTest[,predictors])
dtmTest$rf_PROB <- predict(object=model_rf, dtmTest[,predictors])
dtmTest$rpart_PROB <- predict(object=model_rpart, dtmTest[,predictors])

predictors <- names(blenderTrain)[names(blenderTrain) != labelName]
final_blender_model <- train(blenderTrain[,predictors], blenderTrain[,labelName], method='rf', trControl=myControl)



PredTest = predict(object=final_blender_model, dtmTest[,predictors])


# Now we can prepare our submission file for Kaggle:

MySubmission = data.frame(UniqueID = NewsTest$UniqueID, Probability1 = PredTest)

write.csv(MySubmission, "Ensemble_04.csv", row.names=FALSE)
