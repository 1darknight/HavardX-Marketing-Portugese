# ---
#     title: "Portugese Bank Marketing Calls"
# author: "Le Thuc Anh"
# date: "6/26/2022"
# ---


# Packages installation & dataset import ----------------------------------

# Require packages
Packages <- c("data.table", "tidyverse", "ggplot2", "caret", "gridExtra", "rpart", "randomForest", "rattle", "AUC")
# Print non-scientific number with maximum 4 signifcant digits 
options("scipen" = 999, "digits" = 4)

lapply(Packages, require, character.only = TRUE)

# MovieLens 10M dataset:
# https://archive.ics.uci.edu/ml/datasets/bank+marketing
# https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip
# https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip


zip1 <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip", zip1)
bank_full <- fread(unzip(zip1,"bank-additional/bank-additional-full.csv"))

# rename column to avoid error while running models
setnames(bank_full, c("emp.var.rate", "cons.price.idx", "cons.conf.idx", "nr.employed") , 
         c("emp_var_rate", "cons_price_idx", "cons_conf_idx", "nr_employed"))
```



# Dataset exploration & Data cleaning -------------------------------------

str(bank_full)

# Data cleaning

sum(is.na(bank_full)) # NA or null values

# Data Exploration

summary(bank_full)

# Data Imbalanced
no_y = bank_full[, .N , by = y]
no_y[, perc_N := (N/ sum(N))*100]
no_y



# Split train/validation/test data ----------------------------------------

# Transform y into factors
bank_full$y = as.factor(bank_full$y)
str(bank_full)


set.seed(123)
spec = c(train = .6, validation = .2, test = .2)

g = sample(cut(
  seq(nrow(bank_full)), 
  nrow(bank_full)*cumsum(c(0,spec)),
  labels = names(spec)
))

# Split into train, test, validation dataset
res = split(bank_full, g)
#Move all dataset to global environment
list2env(res,globalenv())

#Check each dataset
print(train)
print(validation)
print(test)



# Model 1 - Decision Tree classification ----------------------------------

## Decision Tree model training
decis_tree = rpart(formula = y ~ ., data = train)
varImp(decis_tree)

fancyRpartPlot(decis_tree, cex = 0.8, caption = "Decision Tree model results")


## Decision Tree model validation
decis_pred = predict(decis_tree, validation, type = 'class')


## F1 Score of Decision tree model  

confusionMatrix(data = decis_pred, reference = validation$y, positive = "yes")

dtree_sense = caret::sensitivity(decis_pred, validation$y)
print(dtree_sense)

dtree_spec = caret::specificity(decis_pred, validation$y)
print(dtree_spec)

decis_f1 = 2*(dtree_sense * dtree_spec)/(dtree_sense + dtree_spec)


## AUC of Decision Tree model
decis_AUC = round(AUC::auc(roc(decis_pred, validation$y)),2)
print(decis_AUC)



# Model 2 - Random Forest ------------------------------------------------

##Random Forest model training 
rf_fit = randomForest(y~., train, importance = TRUE)


##Random Forest model validation 
rf_pred = predict(rf_fit, validation)
head(rf_pred)

rf_AUC = round(AUC::auc(roc(rf_pred, validation$y)),2)
print(rf_AUC)


##Tuning the model  
rf_tuned_trees <- seq(from=1, to=200, by=10)
print(rf_tuned_trees)

rf_tuned = data.frame(matrix(ncol = 2, nrow = 0))

for (i in rf_tuned_trees) {
  fit = randomForest(y~. , train, ntree = i, importance = TRUE)
  pred =  predict(fit, validation)
  auc = AUC::auc(roc(pred, validation$y))
  result = cbind(i, auc)
  rf_tuned = rbind(rf_tuned, result)
}
rf_tuned[rf_tuned$auc == max(rf_tuned$auc),]

rf_tuned_fit = randomForest(y~. , train, ntree = 101, importance = TRUE)
rf_tuned_pred = predict(fit, validation, type = 'class')


## F1 score of Random Forest model
confusionMatrix(data = rf_tuned_pred, reference = validation$y, positive = "yes")

rf_sense = caret::sensitivity(rf_tuned_pred, validation$y)
print(rf_sense)

rf_spec = caret::specificity(rf_tuned_pred, validation$y)
print(rf_spec)

rf_f1 = 2*(rf_sense * rf_spec)/(rf_sense + rf_spec)
print(rf_f1)


## AUC of Random Forest model
rf_tuned_AUC = round(AUC::auc(roc(rf_pred, validation$y)),2)
print(rf_tuned_AUC)



# Model 3 - Logistic Regression -------------------------------------------

##Logistic Regression model training
glm_fit = glm(y ~ ., family = "binomial", data = train)

varImp(glm_fit)


##Logistic Regression model validation
glm_res = predict(glm_fit, validation, type = 'response')
head(glm_res)


## Calculating threshold value
prob = seq(0, 1, length.out = 10)
error = data.frame(matrix(ncol = 3, nrow = 0))
for (i in prob) {
  split = ifelse(glm_res >= i, 'yes', 'no')
  error_yes = 100*sum(ifelse(split == 'yes' & validation$y=='no', 1,0))/length(split)
  error_no = 100*sum(ifelse(split == 'no' & validation$y=='yes', 1,0))/length(split)
  auc = round(AUC::auc(roc(split, validation$y)),2)
  col = cbind(i,auc, error_yes, error_no, tot_error = sum(error_yes, error_no))
  error = rbind(error,col)
  
} 
print(error)


## Predict with the best threshold
glm_pred = factor(ifelse(glm_res >= 0.1, "yes", "no"), levels = levels(validation$y))


## F1 Score of Logistic Regression model  
confusionMatrix(data = glm_pred, reference = validation$y, positive = "yes")

glm_sense = caret::sensitivity(glm_pred, validation$y)
print(glm_sense)

glm_spec = caret::specificity(glm_pred, validation$y)
print(glm_spec)

glm_f1 = 2*(glm_sense * glm_spec)/(glm_sense + glm_spec)
print(glm_f1)


## AUC of Logistic Regression model
glm_AUC = round(AUC::auc(roc(glm_pred, validation$y)),2)
print(glm_AUC)


# Comparing the models performance ----------------------------------------

df <- data.frame(Cat = c("Decision Tree", "Random Forest", "Logistic Regresion"), 
                 F1_score = c(decis_f1, rf_f1, glm_f1), 
                 AUC = c(decis_AUC, rf_AUC, glm_AUC))
print(df)

model <- data.table(cbind(decis_pred, glm_pred, rf_tuned_pred))

plot(roc(decis_pred, validation$y), col="black")
plot(roc(glm_pred, validation$y), add = TRUE, col="red") 
plot(roc(rf_tuned_pred, validation$y), add = TRUE, col="cyan")

legend("bottomright", legend=c(paste0("Decision Tree: ", round(AUC::auc(roc(decis_pred, validation$y)),2)),
                               paste0("GLM: ", round(AUC::auc(roc(glm_pred, validation$y)),2)),
                               paste0("Random Forest: ", round(AUC::auc(roc(rf_tuned_pred, validation$y)),2))),
       col=c("black","red","cyan"), lty=1, cex=0.8)
title(main = "AUC Comparison", sub = "Multiple Model")



# Testing on test set -----------------------------------------------------

## Predict the test dataset  
glm_test_pred = predict(glm_fit, test, type = 'response')
head(glm_test_pred)


## Calculating threshold value for test dataset
prob_test = seq(0, 1, length.out = 10)
error_test = data.frame(matrix(ncol = 3, nrow = 0))
for (i in prob_test) {
  split = ifelse(glm_test_pred >= i, 'yes', 'no')
  error_yes = 100*sum(ifelse(split == 'yes' & test$y=='no', 1,0))/length(split)
  error_no = 100*sum(ifelse(split == 'no' & test$y=='yes', 1,0))/length(split)
  auc = round(AUC::auc(roc(split, test$y)),2)
  col = cbind(i,auc, error_yes, error_no, tot_error = sum(error_yes, error_no))
  error_test = rbind(error_test,col)
  
} 
print(error_test)

glm_test_pred_res = factor(ifelse(glm_test_pred >= 0.1, "yes", "no"), levels = levels(test$y))


## F1 Score of Logistic Regression model  
confusionMatrix(data = glm_test_pred_res, reference = test$y, positive = "yes")

glm_test_sense = caret::sensitivity(glm_test_pred_res, test$y)
print(glm_sense)

glm_test_spec = caret::specificity(glm_test_pred_res, test$y)
print(glm_spec)

glm_test_f1 = 2*(glm_test_sense * glm_test_spec)/(glm_test_sense + glm_test_spec)


## AUC of Logistic Regression model
glm_test_AUC = round(AUC::auc(roc(glm_test_pred_res, test$y)),2)
print(glm_test_AUC)