# Weight Lifting Quality Prediction
Saul Cruz  
4/9/2017  



## Objective

The goal of this project is to predict the manner of how different users did the exercise. 

In summary, our model will predict the following possible outcomes:


* Class A: Correct exercise

* Class B: Incorrect, throwing the elbows to the front 

* Class C: Incorrect, lifting the dumbbell only halfway 

* Class D: Incorrect, lowering the dumbbell only halfway

* Class E: Incorrect, throwing the hips to the front 

The model will use multiple features/attributes that are feed from different sensors:

* Arm sensors orientation

* Belt sensors orientation

* Forearm sensors orientation

* Dumbbell sensors orientation

## Loading the dataset

The WLE Data Set is divided into training and test.

The training dataset contains originally 19622 observations of 160 variables
The testing dataset contains 20 observations of 160 variables


```r
library(caret);library(kernlab);library(RCurl);library(ggplot2);library(corrplot);library(randomForest)

training_source<-getURL('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv',ssl.verifyhost=FALSE,ssl.verifypeer=FALSE)
training <- read.csv(textConnection(training_source), header=T, na.strings = c('#DIV/0', '', 'NA') ,stringsAsFactors = F)

testing_source<-getURL('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv',ssl.verifyhost=FALSE,ssl.verifypeer=FALSE)
testing <- read.csv(textConnection(testing_source),na.strings = c('#DIV/0', '', 'NA') , stringsAsFactors=F,header=T)
```

## Feature Selection

To reduce the highly dimensional dataset, the following steps will be followed:

### 1.-Remove elements with a High Missing Value Ratio (>70%)

In other words, we will remove columns/attributes with more than 70% of its data missing

```r
        testing_clean<-testing[,-which(colMeans(is.na(training)) > 0.7)]
        training_clean<-training[, -which(colMeans(is.na(training)) > 0.7)]
```

After the first step, 62.5% of the features were removed. In other words, there are now only 60 variables.

### 2.- Remove Unnecessary columns like timestamps, new_window,num_window,user_name

```r
        drops <- c("X","user_name","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","new_window","num_window","problem_id")
        training_clean<-training_clean[ , !(names(training_clean) %in% drops)] #53 variables remaining 
        testing_clean<-testing_clean[,!(names(testing_clean) %in% drops)] #52 variables remaining
```
After the second step, there are now 53 variables

### 3.- Apply the High Correlation Filter (Pearson's)
To remove highly correlated data columns, first we need to measure the correlation between pairs of columns using the Linear Correlation node

![](index_files/figure-html/featureselection_3-1.png)<!-- -->
Then we have to apply the Correlation Filter node to remove one of two highly correlated data columns, if any. If two variables have a high correlation, the function looks at the mean absolute correlation of each variable and removes the variable with the largest mean absolute correlation.


```r
        highlyCorrelated<-findCorrelation(correlationMatrix, cutoff=0.75)
        training_clean<-training_clean[,-highlyCorrelated] #32 variables remaining, including the output
        testing_clean<-testing_clean[,-highlyCorrelated]  #31 variables remaining
        #cleaned
        correlationMatrixClean <- cor(training_clean[,1:31])
        corrplot(correlationMatrixClean,order="hclust")
```

![](index_files/figure-html/featureselection_4-1.png)<!-- -->
Once the feature selection is completed, the original dataset has been reduced to only 32 variables, this will help to minimize the risk of overfitting the model.

##Training and Validation datasets
We'll use 70% of the training dataset to train our models,  and 30% of the training dataset to validate the accuracy of each one of our models during the model selection phase.


```r
        inBuild <- createDataPartition(y=training_clean$classe,
                                       p=0.7, list=FALSE)
        validation <- training_clean[-inBuild,]
        training_final <- training_clean[inBuild,]
```

##Model Selection
In this section, we'll use two different algorithms to train our model: Gradient Boosted Machines  and Random Forests

###Gradient Boosted Machines (GBM)


```r
        set.seed(3333)
        mod_gbm <- train(classe ~ ., method="gbm",data=training_final,verbose=FALSE) 
        mod_gbm_predict<-predict(mod_gbm,validation)
        cm<-confusionMatrix(mod_gbm_predict, validation$classe) 
        mod_gbm_accuracy <- cm$overall['Accuracy']
        cm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1640   42    0    2    6
##          B   11 1045   52    5   11
##          C   10   40  953   46   18
##          D   10    6   18  906   24
##          E    3    6    3    5 1023
## 
## Overall Statistics
##                                           
##                Accuracy : 0.946           
##                  95% CI : (0.9399, 0.9516)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9316          
##  Mcnemar's Test P-Value : 1.162e-11       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9797   0.9175   0.9288   0.9398   0.9455
## Specificity            0.9881   0.9834   0.9765   0.9882   0.9965
## Pos Pred Value         0.9704   0.9297   0.8932   0.9398   0.9837
## Neg Pred Value         0.9919   0.9803   0.9848   0.9882   0.9878
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2787   0.1776   0.1619   0.1540   0.1738
## Detection Prevalence   0.2872   0.1910   0.1813   0.1638   0.1767
## Balanced Accuracy      0.9839   0.9504   0.9527   0.9640   0.9710
```

Note that there is an Accuracy of: `mod_gbm_accuracy` 

###Random Forests

```r
        set.seed(3333)
        mod_rf <- train(classe ~.,method="rf",            #Random Forest in the training data
                        data=training_final)
        mod_rf_predict<-predict(mod_rf,validation)
        cm_rf<-confusionMatrix(mod_rf_predict, validation$classe)
        mod_rf_accuracy <- cm_rf$overall['Accuracy']
```

Note that there is an Accuracy of: 0.9915038 which is better than the first model (GBM)

##Predicting testing dataset

```r
        mod_rf_prediction<-predict(mod_rf,testing_clean)
        mod_rf_prediction
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
The above are the predictions for each one of the 20 problems.

#Conclusion
The Random Forest algorithm worked better than the GBM. After determining the accuracy of this algorithm we can conclude that our predictions are 0.9915038 accurate

## References

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
Read more: http://groupware.les.inf.puc-rio.br/har#ixzz4do2ngHXp
