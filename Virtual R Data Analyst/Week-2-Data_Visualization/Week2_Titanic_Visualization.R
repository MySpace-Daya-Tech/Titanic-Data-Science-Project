# ==========================================
# Virtual Data Analyst Internship
# Week 2 - Data Visualization using R
# Titanic Dataset
# ==========================================

# Load visualization library
library(ggplot2)

# Import dataset
data <- read.csv("G:\\DataScience_Internship\\Week-2\\titanic.csv")

# Check dataset structure
str(data)

# Check summary statistics
summary(data)

# Check missing values
colSums(is.na(data))

# Handle missing Age values
data$Age[is.na(data$Age)] <- median(data$Age, na.rm = TRUE)

# Handle missing Embarked values
modeEmbarked <- names(sort(table(data$Embarked),
                            decreasing = TRUE))[1]

data$Embarked[is.na(data$Embarked)] <- modeEmbarked

# Convert categorical variables to factors
data$Sex <- as.factor(data$Sex)
data$Embarked <- as.factor(data$Embarked)
data$Pclass <- as.factor(data$Pclass)

# Check cleaned data
str(data)
summary(data)

# Figure 1: Survival by Passenger Class

plot1 <- ggplot(data, aes(x = Pclass,
                          fill = factor(Survived))) +
  geom_bar(position = "dodge") +
  labs(
    title = "Survival by Passenger Class",
    x = "Passenger Class",
    y = "Number of Passengers",
    fill = "Survival"
  )

ggsave(
  "Week2_Figure1_Survival_by_Class.png",
  plot = plot1,
  width = 8,
  height = 6,
  dpi = 300
)
# Figure 2: Age Distribution of Passengers

plot2 <- ggplot(data, aes(x = Age)) +
  geom_histogram(
    bins = 30,
    fill = "steelblue",
    color = "white"
  ) +
  labs(
    title = "Age Distribution of Titanic Passengers",
    x = "Age",
    y = "Number of Passengers"
  )

ggsave(
  "Week2_Figure2_Age_Distribution.png",
  plot = plot2,
  width = 8,
  height = 6,
  dpi = 300
)
# Figure 3: Fare Distribution of Passengers

plot3 <- ggplot(data, aes(x = Fare)) +
  geom_histogram(
    bins = 30,
    fill = "darkgreen",
    color = "white"
  ) +
  labs(
    title = "Fare Distribution of Titanic Passengers",
    x = "Fare",
    y = "Number of Passengers"
  )

ggsave(
  "Week2_Figure3_Fare_Distribution.png",
  plot = plot3,
  width = 8,
  height = 6,
  dpi = 300
)
# Figure 4: Survival Rate by Passenger Class

survival_rate <- aggregate(
  Survived ~ Pclass,
  data = data,
  FUN = mean
)

plot4 <- ggplot(
  survival_rate,
  aes(
    x = Pclass,
    y = Survived,
    group = 1
  )
) +
  geom_line() +
  geom_point() +
  labs(
    title = "Survival Rate by Passenger Class",
    x = "Passenger Class",
    y = "Survival Rate"
  )

ggsave(
  "Week2_Figure4_Survival_Rate_by_Class.png",
  plot = plot4,
  width = 8,
  height = 6,
  dpi = 300
)