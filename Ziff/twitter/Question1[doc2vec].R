# loading packages
library(tidyverse)
library(text2vec)
library(glmnet)
library(stringr)
library(mltools)

### loading and preprocessing a training set of tweets
# function for converting some symbols
conv_fun <- function(x) iconv(x, "latin1", "ASCII", "")

##### loading classified tweets ######
# 0 - the polarity of the tweet (0 = negative, 1 = positive)
# 1 - the text of the tweet

tweets_classified <- read_tsv('sentiment.tsv',
                              col_names = c('sentiment','text')) %>%

  # converting some symbols
  dmap_at('text', conv_fun) %>%
  
  #REPLACING cLASS VALUES
  mutate(sentiment = ifelse(sentiment == 'neg', 0, 1))

# data splitting on train and test
 set.seed(2341)
  trainIndex <- createDataPartition(tweets_classified$sentiment, p = .99,
                                  list = FALSE, 
                                  times = 1)
tweets_train <- tweets_classified[trainIndex, ]
tweets_test <- tweets_classified[-trainIndex, ]


##### doc2vec Deep Learning Algorithm #####
# define preprocessing function and tokenization function

prep_fun <- tolower
tok_fun <- word_tokenizer

it_train <- itoken(tweets_train$text, 
                   preprocessor = prep_fun, 
                   tokenizer = tok_fun,
                   progressbar = TRUE)
it_test <- itoken(tweets_test$text, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun,
                  progressbar = TRUE)

# creating vocabulary and document-term matrix
vocab <- create_vocabulary(it_train)
vectorizer <- vocab_vectorizer(vocab)
dtm_train <- create_dtm(it_train, vectorizer)
dtm_test <- create_dtm(it_test, vectorizer)

# define tf-idf model
tfidf <- TfIdf$new()

# fit the model to the train data and transform it with the fitted model
dtm_train_tfidf <- fit_transform(dtm_train, tfidf)
dtm_test_tfidf <- fit_transform(dtm_test, tfidf)

# train the model
glmnet_classifier <- cv.glmnet(x = dtm_train_tfidf,
                               y = tweets_train[['sentiment']], 
                               family = 'binomial', 
                               # L1 penalty
                               alpha = 0,
                               # interested in the area under ROC curve
                               type.measure = "auc",
                               # 10-fold cross-validation
                               nfolds = 10,
                               # high value is less accurate, but has faster training
                               thresh = 0.01,
                               # again lower number of iterations for faster training
                               maxit = 1000000
                              )
preds <- predict(glmnet_classifier, dtm_test_tfidf, type = 'response')[,1]
auc_roc(preds, as.numeric(tweets_test$sentiment))
