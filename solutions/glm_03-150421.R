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

HeadlineWordsTrain$Abstract = NewsTrain$Abstract
HeadlineWordsTest$Abstract = NewsTest$Abstract

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

SnippetWordsTrain$Abstract = NewsTrain$Abstract
SnippetWordsTest$Abstract = NewsTest$Abstract


# Create a corpus from the Abstract variable.

CorpusAbstract = Corpus(VectorSource(c(SnippetWordsTrain$Abstract, SnippetWordsTest$Abstract)))

CorpusAbstract = tm_map(CorpusAbstract, tolower)

CorpusAbstract = tm_map(CorpusAbstract, PlainTextDocument)

CorpusAbstract = tm_map(CorpusAbstract, removePunctuation)

CorpusAbstract = tm_map(CorpusAbstract, removeWords, stopwords("english"))

CorpusAbstract = tm_map(CorpusAbstract, stemDocument)


dtm_Abstract = DocumentTermMatrix(CorpusAbstract)

sparse = removeSparseTerms(dtm_Abstract, 0.99)

AbstractWords = as.data.frame(as.matrix(sparse))

colnames(AbstractWords) = make.names(colnames(AbstractWords))

AbstractWordsTrain = head(AbstractWords, nrow(SnippetWordsTrain))

AbstractWordsTest = tail(AbstractWords, nrow(SnippetWordsTest))

AbstractWordsTrain$Popular = NewsTrain$Popular

AbstractWordsTrain$WordCount = NewsTrain$WordCount
AbstractWordsTest$WordCount = NewsTest$WordCount

AbstractWordsTrain$NewsDesk = NewsTrain$NewsDesk
AbstractWordsTest$NewsDesk = NewsTest$NewsDesk

AbstractWordsTrain$SectionName = NewsTrain$SectionName
AbstractWordsTest$SectionName = NewsTest$SectionName

AbstractWordsTrain$SubsectionName  = NewsTrain$SubsectionName 
AbstractWordsTest$SubsectionName  = NewsTest$SubsectionName

AbstractWordsTrain$Weekday  = NewsTrain$Weekday 
AbstractWordsTest$Weekday  = NewsTest$Weekday

AbstractWordsTrain$Hour  = NewsTrain$hour 
AbstractWordsTest$Hour  = NewsTest$hour

# Make glm model
AbstractWordsLog = glm(Popular ~.-famili-former-secur-thursday, data=AbstractWordsTrain, family=binomial)


PredTest = predict(AbstractWordsLog, newdata=AbstractWordsTest, type="response")

library(ROCR)
PredTrain = predict(AbstractWordsLog, data=AbstractWordsTrain, type="response")
predROCR = prediction(PredTrain, AbstractWordsTrain$Popular)
perfROCR = performance(predROCR, "tpr", "fpr")
plot(perfROCR, colorize=TRUE)
performance(predROCR, "auc")@y.values

# Now we can prepare our submission file for Kaggle:

MySubmission = data.frame(UniqueID = NewsTest$UniqueID, Probability1 = PredTest)

write.csv(MySubmission, "Submissionglm_07.csv", row.names=FALSE)
