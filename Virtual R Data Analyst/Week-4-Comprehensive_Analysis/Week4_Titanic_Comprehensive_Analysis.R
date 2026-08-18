# ============================================================
# WEEK 4 - COMPREHENSIVE DATA ANALYSIS REPORTING
# Titanic Dataset - R
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
  "G:\\DataScience_Internship\\Week-4\\titanic.csv",
  stringsAsFactors = FALSE
)

cat("\n===== DATASET LOADED =====\n")
cat("Observations:", nrow(data), "\n")
cat("Variables:", ncol(data), "\n")

# ------------------------------------------------------------
# 3. Data Inspection
# ------------------------------------------------------------

cat("\n===== DATA STRUCTURE =====\n")
str(data)

cat("\n===== SUMMARY =====\n")
print(summary(data))

# ------------------------------------------------------------
# 4. Data Cleaning
# ------------------------------------------------------------

# Handle missing Age values
data$Age[is.na(data$Age)] <-
  median(data$Age, na.rm = TRUE)

# Handle missing Embarked values
data$Embarked[data$Embarked == ""] <- NA

modeEmbarked <-
  names(sort(table(data$Embarked), decreasing = TRUE))[1]

data$Embarked[is.na(data$Embarked)] <-
  modeEmbarked

# ------------------------------------------------------------
# 5. Convert Variables to Appropriate Data Types
# ------------------------------------------------------------

# IMPORTANT:
# Use valid R names instead of 0 and 1
data$Survived <- factor(
  data$Survived,
  levels = c(0, 1),
  labels = c("Not_Survived", "Survived")
)

data$Pclass <- factor(data$Pclass)

data$Sex <- factor(data$Sex)

data$Embarked <- factor(data$Embarked)

cat("\n===== MISSING VALUES AFTER CLEANING =====\n")
print(colSums(is.na(data)))

# ------------------------------------------------------------
# 6. Descriptive Statistics
# ------------------------------------------------------------

cat("\n===== AGE STATISTICS =====\n")
cat("Mean:", mean(data$Age), "\n")
cat("Median:", median(data$Age), "\n")
cat("Standard Deviation:", sd(data$Age), "\n")

cat("\n===== FARE STATISTICS =====\n")
cat("Mean:", mean(data$Fare), "\n")
cat("Median:", median(data$Fare), "\n")
cat("Standard Deviation:", sd(data$Fare), "\n")

cat("\n===== SURVIVAL COUNTS =====\n")
print(table(data$Survived))

cat("\n===== PASSENGER CLASS COUNTS =====\n")
print(table(data$Pclass))

cat("\n===== SEX COUNTS =====\n")
print(table(data$Sex))

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
# Chi-Square Test
# ------------------------------------------------------------

survival_class_table <-
  table(data$Pclass, data$Survived)

cat("\n===== SURVIVAL BY PASSENGER CLASS =====\n")
print(survival_class_table)

chi_test <- chisq.test(
  survival_class_table
)

cat("\n===== CHI-SQUARE TEST =====\n")
print(chi_test)

# ------------------------------------------------------------
# 9. Train-Test Split
# ------------------------------------------------------------

set.seed(123)

train_index <- createDataPartition(
  data$Survived,
  p = 0.80,
  list = FALSE
)

train_data <- data[train_index, ]
test_data <- data[-train_index, ]

cat("\n===== DATA SPLIT =====\n")
cat("Training observations:", nrow(train_data), "\n")
cat("Testing observations:", nrow(test_data), "\n")

# ------------------------------------------------------------
# 10. Logistic Regression Model
# ------------------------------------------------------------

cat("\n===== LOGISTIC REGRESSION MODEL =====\n")

logistic_model <- glm(
  Survived ~ Pclass + Sex + Age + Fare,
  data = train_data,
  family = binomial
)

print(summary(logistic_model))

# ------------------------------------------------------------
# 11. Model Prediction
# ------------------------------------------------------------

predicted_prob <- predict(
  logistic_model,
  newdata = test_data,
  type = "response"
)

predicted_class <- ifelse(
  predicted_prob >= 0.5,
  "Survived",
  "Not_Survived"
)

predicted_class <- factor(
  predicted_class,
  levels = levels(test_data$Survived)
)

cat("\n===== FIRST PREDICTED PROBABILITIES =====\n")
print(head(predicted_prob))

cat("\n===== FIRST PREDICTED CLASSES =====\n")
print(head(predicted_class))

# ------------------------------------------------------------
# 12. Confusion Matrix
# ------------------------------------------------------------

conf_matrix <- confusionMatrix(
  predicted_class,
  test_data$Survived,
  positive = "Survived"
)

cat("\n===== CONFUSION MATRIX =====\n")
print(conf_matrix)

# ------------------------------------------------------------
# 13. Accuracy
# ------------------------------------------------------------

accuracy <- conf_matrix$overall["Accuracy"]

cat("\n===== MODEL ACCURACY =====\n")
print(accuracy)

# ------------------------------------------------------------
# 14. ROC Curve and AUC
# ------------------------------------------------------------

roc_result <- roc(
  response = test_data$Survived,
  predictor = predicted_prob,
  levels = c("Not_Survived", "Survived"),
  direction = "<"
)

auc_value <- auc(roc_result)

cat("\n===== AUC =====\n")
print(auc_value)

# ------------------------------------------------------------
# 15. Save ROC Curve
# ------------------------------------------------------------

png(
  "G:\\DataScience_Internship\\Week-4\\ROC_Curve.png",
  width = 900,
  height = 700
)

plot(
  roc_result,
  main = "ROC Curve - Titanic Survival Model",
  lwd = 2
)

abline(
  a = 0,
  b = 1,
  lty = 2
)

dev.off()

# ------------------------------------------------------------
# 16. Cook's Distance
# ------------------------------------------------------------

png(
  "G:\\DataScience_Internship\\Week-4\\Cooks_Distance.png",
  width = 900,
  height = 700
)

plot(
  cooks.distance(logistic_model),
  type = "h",
  main = "Cook's Distance",
  xlab = "Observation",
  ylab = "Cook's Distance"
)

dev.off()

# ------------------------------------------------------------
# 17. Survival Prediction Distribution
# ------------------------------------------------------------

prediction_data <- data.frame(
  Actual = test_data$Survived,
  Predicted_Probability = predicted_prob
)

prediction_plot <- ggplot(
  prediction_data,
  aes(
    x = Actual,
    y = Predicted_Probability
  )
) +
  geom_boxplot() +
  labs(
    title = "Predicted Survival Probability by Actual Outcome",
    x = "Actual Outcome",
    y = "Predicted Survival Probability"
  ) +
  theme_minimal()

ggsave(
  "G:\\DataScience_Internship\\Week-4\\Prediction_Probability.png",
  prediction_plot,
  width = 8,
  height = 6
)
# -----------------------------------
# 18. Fold Cross-Validation
# -----------------------------------

cat("\n===== 5-FOLD CROSS-VALIDATION =====\n")

set.seed(123)

cv_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

cv_data <- data

cv_model <- train(
  Survived ~ Pclass + Sex + Age + Fare,
  data = cv_data,
  method = "glm",
  family = binomial,
  metric = "ROC",
  trControl = cv_control
)

print(cv_model)

cat("\n===== CROSS-VALIDATION RESULTS =====\n")
print(cv_model$results)

# ------------------------------------------------------------
# 19. Final Model Summary
# ------------------------------------------------------------

cat("\n========================================\n")
cat("WEEK 4 MODELING SUMMARY\n")
cat("========================================\n")

cat("Training observations:", nrow(train_data), "\n")
cat("Testing observations:", nrow(test_data), "\n")
cat("Accuracy:", round(as.numeric(accuracy), 4), "\n")
cat("AUC:", round(as.numeric(auc_value), 4), "\n")

cat("\n===== WEEK 4 SCRIPT COMPLETED SUCCESSFULLY =====\n")