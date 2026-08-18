# ============================================================
# Statistical Analysis and Predictive Modeling using R
# ============================================================

# ------------------------------------------------------------
# 1. Load Required Packages
# ------------------------------------------------------------

library(ggplot2)
library(caret)
library(pROC)

# ------------------------------------------------------------
# 2. Load Dataset
# ------------------------------------------------------------

data <- read.csv(
  "G:\\DataScience_Internship\\Week-3\\titanic.csv",
  stringsAsFactors = FALSE
)

# ------------------------------------------------------------
# 3. Initial Dataset Inspection
# ------------------------------------------------------------

cat("\n===== DATASET STRUCTURE =====\n")
str(data)

cat("\n===== SUMMARY STATISTICS =====\n")
print(summary(data))

cat("\n===== MISSING VALUES BEFORE CLEANING =====\n")
print(colSums(is.na(data)))

# ------------------------------------------------------------
# 4. Data Cleaning and Preprocessing
# ------------------------------------------------------------

# Replace missing Age values with median
data$Age[is.na(data$Age)] <- median(
  data$Age,
  na.rm = TRUE
)

# Convert blank Embarked values to NA
data$Embarked[data$Embarked == ""] <- NA

# Replace missing Embarked values with mode
modeEmbarked <- names(
  sort(
    table(data$Embarked),
    decreasing = TRUE
  )
)[1]

data$Embarked[is.na(data$Embarked)] <- modeEmbarked

# Convert categorical variables to factors
data$Survived <- factor(
  data$Survived,
  levels = c(0, 1)
)

data$Pclass <- factor(data$Pclass)

data$Sex <- factor(data$Sex)

data$Embarked <- factor(data$Embarked)

cat("\n===== MISSING VALUES AFTER CLEANING =====\n")
print(colSums(is.na(data)))

cat("\n===== FINAL DATASET STRUCTURE =====\n")
str(data)

# ------------------------------------------------------------
# 5. Descriptive Statistical Analysis
# ------------------------------------------------------------

cat("\n===== DESCRIPTIVE STATISTICS =====\n")

cat("\nAge Mean:\n")
print(mean(data$Age))

cat("\nAge Median:\n")
print(median(data$Age))

cat("\nAge Standard Deviation:\n")
print(sd(data$Age))

cat("\nFare Mean:\n")
print(mean(data$Fare))

cat("\nFare Median:\n")
print(median(data$Fare))

cat("\nFare Standard Deviation:\n")
print(sd(data$Fare))

cat("\n===== SURVIVAL COUNTS =====\n")
print(table(data$Survived))

cat("\n===== PASSENGER CLASS COUNTS =====\n")
print(table(data$Pclass))

cat("\n===== SEX COUNTS =====\n")
print(table(data$Sex))

# ------------------------------------------------------------
# 6. Normality Analysis
# ------------------------------------------------------------

cat("\n===== SHAPIRO-WILK NORMALITY TEST FOR AGE =====\n")

# Shapiro-Wilk supports maximum 5000 observations
age_test <- shapiro.test(data$Age)

print(age_test)

# ------------------------------------------------------------
# 7. Correlation Analysis
# ------------------------------------------------------------

numeric_data <- data[
  c("Age", "SibSp", "Parch", "Fare")
]

correlation_matrix <- cor(
  numeric_data,
  use = "complete.obs"
)

cat("\n===== CORRELATION MATRIX =====\n")
print(correlation_matrix)

# ------------------------------------------------------------
# 8. Hypothesis Testing
# Chi-Square Test of Independence
# ------------------------------------------------------------

survival_class_table <- table(
  data$Pclass,
  data$Survived
)

cat("\n===== SURVIVAL BY PASSENGER CLASS =====\n")
print(survival_class_table)

chi_test <- chisq.test(
  survival_class_table
)

cat("\n===== CHI-SQUARE TEST =====\n")
print(chi_test)

# ------------------------------------------------------------
# 9. Logistic Regression Model
# ------------------------------------------------------------

cat("\n===== LOGISTIC REGRESSION MODEL =====\n")

logistic_model <- glm(
  Survived ~ Pclass + Sex + Age + Fare,
  data = data,
  family = binomial
)

print(summary(logistic_model))

# ------------------------------------------------------------
# 10. Train-Test Split
# ------------------------------------------------------------

set.seed(123)

train_index <- createDataPartition(
  data$Survived,
  p = 0.80,
  list = FALSE
)

train_data <- data[train_index, ]

test_data <- data[-train_index, ]

cat("\n===== TRAIN-TEST SPLIT =====\n")

cat("Training observations:\n")
print(nrow(train_data))

cat("Testing observations:\n")
print(nrow(test_data))

# ------------------------------------------------------------
# 11. Train Logistic Regression Model
# ------------------------------------------------------------

cat("\n===== TRAINING LOGISTIC REGRESSION MODEL =====\n")

train_model <- glm(
  Survived ~ Pclass + Sex + Age + Fare,
  data = train_data,
  family = binomial
)

print(summary(train_model))

# ------------------------------------------------------------
# 12. Prediction on Test Data
# ------------------------------------------------------------

predicted_prob <- predict(
  train_model,
  newdata = test_data,
  type = "response"
)

predicted_class <- ifelse(
  predicted_prob >= 0.5,
  "1",
  "0"
)

predicted_class <- factor(
  predicted_class,
  levels = c("0", "1")
)

cat("\n===== FIRST PREDICTED PROBABILITIES =====\n")
print(head(predicted_prob))

cat("\n===== FIRST PREDICTED CLASSES =====\n")
print(head(predicted_class))

# ------------------------------------------------------------
# 13. Confusion Matrix
# ------------------------------------------------------------

cat("\n===== CONFUSION MATRIX =====\n")

confusion_result <- confusionMatrix(
  predicted_class,
  test_data$Survived,
  positive = "1"
)

print(confusion_result)

# ------------------------------------------------------------
# 14. Model Performance Metrics
# ------------------------------------------------------------

accuracy <- confusion_result$overall["Accuracy"]

sensitivity <- confusion_result$byClass["Sensitivity"]

specificity <- confusion_result$byClass["Specificity"]

precision <- confusion_result$byClass["Pos Pred Value"]

f1_score <- 2 * (
  precision * sensitivity
) / (
  precision + sensitivity
)

cat("\n===== MODEL PERFORMANCE =====\n")

cat("Accuracy:\n")
print(accuracy)

cat("Sensitivity:\n")
print(sensitivity)

cat("Specificity:\n")
print(specificity)

cat("Precision:\n")
print(precision)

cat("F1 Score:\n")
print(f1_score)

# ------------------------------------------------------------
# 15. ROC Curve and AUC
# ------------------------------------------------------------

cat("\n===== ROC CURVE AND AUC =====\n")

roc_model <- roc(
  response = test_data$Survived,
  predictor = predicted_prob,
  levels = c("0", "1"),
  direction = "<"
)

auc_value <- auc(roc_model)

cat("AUC:\n")
print(auc_value)

# Display ROC curve
plot(
  roc_model,
  main = "ROC Curve - Logistic Regression"
)

# ------------------------------------------------------------
# 16. Residual Diagnostics
# ------------------------------------------------------------

cat("\n===== MODEL DIAGNOSTICS =====\n")

# Residuals versus fitted values
plot(
  fitted(train_model),
  residuals(train_model, type = "deviance"),
  xlab = "Fitted Values",
  ylab = "Deviance Residuals",
  main = "Residuals vs Fitted Values"
)

abline(
  h = 0,
  lty = 2
)

# ------------------------------------------------------------
# 17. Cook's Distance
# ------------------------------------------------------------

cook_values <- cooks.distance(
  train_model
)

cat("\n===== COOK'S DISTANCE SUMMARY =====\n")
print(summary(cook_values))

plot(
  cook_values,
  type = "h",
  main = "Cook's Distance",
  xlab = "Observation",
  ylab = "Cook's Distance"
)

# ------------------------------------------------------------
# 18. 10-Fold Cross-Validation
# ------------------------------------------------------------

cat("\n===== 10-FOLD CROSS-VALIDATION =====\n")

set.seed(123)

cv_control <- trainControl(
  method = "cv",
  number = 10
)

cv_model <- train(
  Survived ~ Pclass + Sex + Age + Fare,
  data = train_data,
  method = "glm",
  family = binomial,
  trControl = cv_control,
  metric = "Accuracy"
)

print(cv_model)

cat("\n===== CROSS-VALIDATION RESULTS =====\n")
print(cv_model$results)

# ------------------------------------------------------------
# 19. Final Model Summary
# ------------------------------------------------------------

cat("\n============================================\n")
cat("FINAL MODEL SUMMARY\n")
cat("============================================\n")

cat("Test Accuracy: ")
print(accuracy)

cat("Test Sensitivity: ")
print(sensitivity)

cat("Test Specificity: ")
print(specificity)

cat("Test Precision: ")
print(precision)

cat("Test F1 Score: ")
print(f1_score)

cat("Test AUC: ")
print(auc_value)

cat("\n10-Fold Cross-Validation Accuracy:\n")
print(cv_model$results$Accuracy)

cat("\n===== ANALYSIS COMPLETED SUCCESSFULLY =====\n")