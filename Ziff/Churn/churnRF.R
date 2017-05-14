library(plyr)
library(dplyr)
library(stringr)
library(ggplot2)
library(reshape2)
library(caret)
churn <- read.csv("churn.csv")

set.seed(12)
trainIndex <- caret::createDataPartition(churn$Churn., p = .75, list = FALSE, times = 1)
churnTrain <- churn[ trainIndex,]
churnTest <- churn[-trainIndex, ]

churnTrain$Phone<-NULL
churnTest$Phone<-NULL
churnTrain$Area.Code<-as.factor(churnTrain$Area.Code)
churnTest$Area.Code<-as.factor(churnTest$Area.Code)
set.seed(12)
rfmodel<-train(churnTrain$Churn.~., data=churnTrain, method="rf", 
               trainControl = c(method = "adaptive_cv", number = 10, repeats = 5, classProbs = TRUE, summaryFunction = twoClassSummary, 
                                adaptive = list(min = 10, alpha = 0.05, method = "gls", complete = TRUE)), metric="Kappa")
confusionMatrix(rfmodel)
pred<-predict(rfmodel, newdata=churnTest)
confusionMatrix(pred, churnTest$Churn.)

modelroc2 = roc(pred, churnTest$Churn)
plot(modelroc2, print.auc=TRUE, auc.polygon=T, grid=c(0.1,0.2),grid.col=c('green','red'),max.auc.polygon=TRUE, auc.polygon.col="skyblue", print.thres=TRUE)

