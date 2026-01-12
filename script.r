# script file
install.packages(c("cluster", "rpart", "C50", "tree", "randomForest", "kknn", "ROCR"))

library(cluster)
library(rpart)
library(C50)
library(tree)
library(randomForest)
library(kknn)
library(ROCR)

# Chargement des données
data <- read.csv("Data_Projet.csv", header = TRUE, sep = ",", dec = ".",
                 stringsAsFactors = TRUE)
data_new <- read.csv("Data_Projet_New.csv", header = TRUE, sep = ",", dec = ".",
                     stringsAsFactors = TRUE)


