"doc2vec" is a deep learning algorithm that draws context from phrases. It’s currently one of the best ways of sentiment classification and library(text2vec) is also used thanks to "Dmitriy Selivanov" :p
By running the code with the Given "sentiment.csv" file placed in the given folder  With a 10 k-fold cross-validation study the following output was generated. 

auc_roc(preds, as.numeric(tweets_test$sentiment))
[1] 0.8571429

AUC Score:0.8571
