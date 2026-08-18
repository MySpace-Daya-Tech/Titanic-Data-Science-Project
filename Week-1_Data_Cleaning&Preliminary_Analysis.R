# ============================================================
# Virtual R Data Analyst Internship
# Week 1 - Data Cleaning and Preliminary Analysis
# Dataset: Titanic Dataset
# ============================================================

# ------------------------------------------------------------
# 1. Load Required Packages
# ------------------------------------------------------------

library(ggplot2)
library(dplyr)

# ------------------------------------------------------------
# 2. Load Dataset
# ------------------------------------------------------------

data <- read.csv(
  "G:\\DataScience_Internship\\Week-1\\titanic.csv",
  stringsAsFactors = FALSE
)

cat("\n========================================\n")
cat("DATASET LOADED SUCCESSFULLY\n")
cat("========================================\n")

# ------------------------------------------------------------
# 3. Initial Dataset Inspection
# ------------------------------------------------------------

cat("\n===== DATA STRUCTURE =====\n")
str(data)

cat("\n===== FIRST SIX RECORDS =====\n")
print(head(data))

cat("\n===== SUMMARY STATISTICS =====\n")
print(summary(data))

# ------------------------------------------------------------
# 4. Check Missing Values
# ------------------------------------------------------------

cat("\n===== MISSING VALUES BEFORE CLEANING =====\n")

missing_values <- colSums(is.na(data))
print(missing_values)

# Check blank values in character columns
blank_values <- sapply(
  data,
  function(x) {
    if (is.character(x)) {
      sum(trimws(x) == "", na.rm = TRUE)
    } else {
      0
    }
  }
)

cat("\n===== BLANK CHARACTER VALUES =====\n")
print(blank_values)

# ------------------------------------------------------------
# 5. Handle Missing Values
# ------------------------------------------------------------

# Age: replace missing values with median
age_median <- median(data$Age, na.rm = TRUE)

data$Age[is.na(data$Age)] <- age_median

# Embarked: convert blank values to NA
data$Embarked[
  trimws(data$Embarked) == ""
] <- NA

# Replace missing Embarked values with mode
mode_embarked <- names(
  sort(
    table(data$Embarked),
    decreasing = TRUE
  )
)[1]

data$Embarked[
  is.na(data$Embarked)
] <- mode_embarked

# ------------------------------------------------------------
# 6. Handle Cabin Missingness
# ------------------------------------------------------------

# Cabin contains a large number of blank values.
# Instead of deleting a large number of observations,
# create an indicator showing whether cabin information exists.

data$CabinAvailable <- ifelse(
  trimws(data$Cabin) == "",
  "No",
  "Yes"
)

# ------------------------------------------------------------
# 7. Convert Categorical Variables
# ------------------------------------------------------------

data$Survived <- factor(
  data$Survived,
  levels = c(0, 1),
  labels = c("Not_Survived", "Survived")
)

data$Pclass <- factor(data$Pclass)

data$Sex <- factor(data$Sex)

data$Embarked <- factor(data$Embarked)

data$CabinAvailable <- factor(
  data$CabinAvailable,
  levels = c("No", "Yes")
)

# ------------------------------------------------------------
# 8. Verify Missing Values After Cleaning
# ------------------------------------------------------------

cat("\n===== MISSING VALUES AFTER CLEANING =====\n")

print(colSums(is.na(data)))

# ------------------------------------------------------------
# 9. Outlier Detection
# ------------------------------------------------------------

cat("\n===== OUTLIER DETECTION =====\n")

# Function to calculate IQR-based outliers
detect_outliers <- function(x) {

  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)

  IQR_value <- Q3 - Q1

  lower_limit <- Q1 - 1.5 * IQR_value
  upper_limit <- Q3 + 1.5 * IQR_value

  outliers <- x[
    x < lower_limit | x > upper_limit
  ]

  return(
    list(
      lower_limit = lower_limit,
      upper_limit = upper_limit,
      count = length(outliers)
    )
  )
}

fare_outliers <- detect_outliers(data$Fare)

age_outliers <- detect_outliers(data$Age)

cat("\nFare outlier count:",
    fare_outliers$count, "\n")

cat("Fare lower limit:",
    fare_outliers$lower_limit, "\n")

cat("Fare upper limit:",
    fare_outliers$upper_limit, "\n")

cat("\nAge outlier count:",
    age_outliers$count, "\n")

cat("Age lower limit:",
    age_outliers$lower_limit, "\n")

cat("Age upper limit:",
    age_outliers$upper_limit, "\n")

# ------------------------------------------------------------
# 10. Outlier Visualization
# ------------------------------------------------------------

ggplot(data, aes(y = Fare)) +
  geom_boxplot() +
  labs(
    title = "Fare Outlier Detection",
    y = "Fare"
  ) +
  theme_minimal()

ggsave(
  "Week1_Fare_Outlier_Detection.png",
  width = 8,
  height = 5,
  dpi = 300
)

# ------------------------------------------------------------
# 11. Normalization
# ------------------------------------------------------------

# Min-Max normalization of Fare
# Original Fare is retained and a normalized version is created.

fare_min <- min(data$Fare, na.rm = TRUE)
fare_max <- max(data$Fare, na.rm = TRUE)

data$Fare_Normalized <- (
  data$Fare - fare_min
) / (
  fare_max - fare_min
)

cat("\n===== NORMALIZATION =====\n")

cat(
  "Normalized Fare range: ",
  min(data$Fare_Normalized),
  "to",
  max(data$Fare_Normalized),
  "\n"
)

# ------------------------------------------------------------
# 12. Descriptive Statistics
# ------------------------------------------------------------

cat("\n===== AGE STATISTICS =====\n")

cat(
  "Mean Age:",
  mean(data$Age, na.rm = TRUE),
  "\n"
)

cat(
  "Median Age:",
  median(data$Age, na.rm = TRUE),
  "\n"
)

cat(
  "Standard Deviation of Age:",
  sd(data$Age, na.rm = TRUE),
  "\n"
)

cat("\n===== FARE STATISTICS =====\n")

cat(
  "Mean Fare:",
  mean(data$Fare, na.rm = TRUE),
  "\n"
)

cat(
  "Median Fare:",
  median(data$Fare, na.rm = TRUE),
  "\n"
)

cat(
  "Standard Deviation of Fare:",
  sd(data$Fare, na.rm = TRUE),
  "\n"
)

# ------------------------------------------------------------
# 13. Frequency Analysis
# ------------------------------------------------------------

cat("\n===== SURVIVAL COUNTS =====\n")
print(table(data$Survived))

cat("\n===== PASSENGER CLASS COUNTS =====\n")
print(table(data$Pclass))

cat("\n===== GENDER COUNTS =====\n")
print(table(data$Sex))

cat("\n===== EMBARKATION COUNTS =====\n")
print(table(data$Embarked))

# ------------------------------------------------------------
# 14. Correlation Analysis
# ------------------------------------------------------------

numeric_data <- data %>%
  select(
    Age,
    SibSp,
    Parch,
    Fare
  )

correlation_matrix <- cor(
  numeric_data,
  use = "complete.obs"
)

cat("\n===== CORRELATION MATRIX =====\n")

print(
  round(
    correlation_matrix,
    3
  )
)

# ------------------------------------------------------------
# 15. Exploratory Visualization - Survival
# ------------------------------------------------------------

ggplot(
  data,
  aes(x = Survived)
) +
  geom_bar() +
  labs(
    title = "Passenger Survival Distribution",
    x = "Survival Status",
    y = "Number of Passengers"
  ) +
  theme_minimal()

ggsave(
  "Week1_Survival_Distribution.png",
  width = 8,
  height = 5,
  dpi = 300
)

# ------------------------------------------------------------
# 16. Exploratory Visualization - Age Distribution
# ------------------------------------------------------------

ggplot(
  data,
  aes(x = Age)
) +
  geom_histogram(
    bins = 30
  ) +
  labs(
    title = "Age Distribution of Passengers",
    x = "Age",
    y = "Number of Passengers"
  ) +
  theme_minimal()

ggsave(
  "Week1_Age_Distribution.png",
  width = 8,
  height = 5,
  dpi = 300
)

# ------------------------------------------------------------
# 17. Exploratory Visualization - Fare Distribution
# ------------------------------------------------------------

ggplot(
  data,
  aes(x = Fare)
) +
  geom_histogram(
    bins = 30
  ) +
  labs(
    title = "Fare Distribution",
    x = "Fare",
    y = "Number of Passengers"
  ) +
  theme_minimal()

ggsave(
  "Week1_Fare_Distribution.png",
  width = 8,
  height = 5,
  dpi = 300
)

# ------------------------------------------------------------
# 18. Exploratory Visualization - Survival by Class
# ------------------------------------------------------------

ggplot(
  data,
  aes(
    x = Pclass,
    fill = Survived
  )
) +
  geom_bar(
    position = "dodge"
  ) +
  labs(
    title = "Survival by Passenger Class",
    x = "Passenger Class",
    y = "Number of Passengers",
    fill = "Survival"
  ) +
  theme_minimal()

ggsave(
  "Week1_Survival_by_Class.png",
  width = 8,
  height = 5,
  dpi = 300
)

# ------------------------------------------------------------
# 19. Final Dataset Structure
# ------------------------------------------------------------

cat("\n===== FINAL DATA STRUCTURE =====\n")

str(data)

cat("\n===== FINAL SUMMARY =====\n")

print(summary(data))

# ------------------------------------------------------------
# 20. Save Cleaned Dataset
# ------------------------------------------------------------

write.csv(
  data,
  "Week1_Titanic_Cleaned.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 21. Completion Message
# ------------------------------------------------------------

cat("\n========================================\n")
cat("WEEK 1 ANALYSIS COMPLETED SUCCESSFULLY\n")
cat("========================================\n")

cat("Dataset cleaning completed.\n")
cat("Missing values handled.\n")
cat("Categorical variables encoded.\n")
cat("Outliers identified.\n")
cat("Fare normalization completed.\n")
cat("Descriptive statistics generated.\n")
cat("Correlation analysis completed.\n")
cat("Exploratory visualizations generated.\n")
cat("Cleaned dataset exported.\n")

cat("========================================\n")