setwd("C:/Users/matlo/Documents/MIAGE/M2/Analyse_des_donnees/Projet")

# 1. Chargement des librairies
install.packages("cluster")
install.packages("ggplot2")
install.packages("tsne")

library(cluster)
library(ggplot2)
library(tsne)

# 2. Chargement des données
data <- read.csv("Data_Projet.csv", header = TRUE, sep = ",", dec = ".",
                 stringsAsFactors = TRUE)

str(data)
summary(data)

# 3. Séparation par classe
data_fraude     <- subset(data, fraudulent == "Yes")
data_nonfraude  <- subset(data, fraudulent == "No")

# Suppression de la variable de classe
data_fraude$fraudulent <- NULL
data_nonfraude$fraudulent <- NULL

# Conversion des variables catégoriques en numériques pour le clustering
data_fraude_numeric <- data.frame(lapply(data_fraude, function(x) {
  if(is.factor(x)) as.numeric(x) else x
}))
data_nonfraude_numeric <- data.frame(lapply(data_nonfraude, function(x) {
  if(is.factor(x)) as.numeric(x) else x
}))

# Suppression des lignes avec des NA
data_fraude_numeric <- na.omit(data_fraude_numeric)
data_nonfraude_numeric <- na.omit(data_nonfraude_numeric)

# 4. Calcul des matrices de distance
dmatrix_fraude <- daisy(data_fraude)
dmatrix_nonfraude <- daisy(data_nonfraude)

# CLUSTERING PAR K-MEANS
set.seed(10000) # Pour reproductibilité

# 5. K-means sur les FRAUDES
km_fraude <- kmeans(data_fraude_numeric, centers = 3)

data_fraude_km <- data.frame(data_fraude_numeric)
data_fraude_km$Cluster <- km_fraude$cluster

# Effectifs par cluster
table(data_fraude_km$Cluster)

# 6. K-means sur les NON-FRAUDES
km_nonfraude <- kmeans(data_nonfraude_numeric, centers = 3)

data_nonfraude_km <- data.frame(data_nonfraude_numeric)
data_nonfraude_km$Cluster <- km_nonfraude$cluster

table(data_nonfraude_km$Cluster)

# CLUSTERING HIERARCHIQUE (AGNES)
# 7. Clustering hiérarchique – FRAUDES
agn_fraude <- agnes(dmatrix_fraude)

# Découpage en 3 clusters
agn_fraude_3 <- cutree(agn_fraude, k = 3)

data_fraude_hc <- data.frame(data_fraude)
data_fraude_hc$Cluster <- agn_fraude_3

table(data_fraude_hc$Cluster)

# 8. Clustering hiérarchique – NON-FRAUDES
agn_nonfraude <- agnes(dmatrix_nonfraude)

agn_nonfraude_3 <- cutree(agn_nonfraude, k = 3)

data_nonfraude_hc <- data.frame(data_nonfraude)
data_nonfraude_hc$Cluster <- agn_nonfraude_3

table(data_nonfraude_hc$Cluster)

# CARACTERISATION ET COMPARAISON DES CLUSTERS
vars_to_plot <- c("age", "days_to_incident", "claim_amount", "total_policy_claims")
vars_available <- intersect(vars_to_plot, colnames(data_fraude))

# 14. Boxplots - K-means FRAUDES
png("clustering/Boxplot_KMeans_Fraudes.png", width = 1200, height = 800)
par(mfrow = c(2, 2))
for (var in vars_available) {
  boxplot(data_fraude[[var]] ~ data_fraude_km$Cluster,
          main = paste("K-means FRAUDES -", var),
          xlab = "Cluster",
          ylab = var)
}
par(mfrow = c(1, 1))
dev.off()

# 14. Boxplots - K-means NON-FRAUDES
png("clustering/Boxplot_KMeans_NonFraudes.png", width = 1200, height = 800)
par(mfrow = c(2, 2))
for (var in vars_available) {
  boxplot(data_nonfraude[[var]] ~ data_nonfraude_km$Cluster,
          main = paste("K-means NON-FRAUDES -", var),
          xlab = "Cluster",
          ylab = var)
}
par(mfrow = c(1, 1))
dev.off()

# 15. Boxplots - AGNES FRAUDES
png("clustering/Boxplot_AGNES_Fraudes.png", width = 1200, height = 800)
par(mfrow = c(2, 2))
for (var in vars_available) {
  boxplot(data_fraude[[var]] ~ data_fraude_hc$Cluster,
          main = paste("AGNES FRAUDES -", var),
          xlab = "Cluster",
          ylab = var)
}
par(mfrow = c(1, 1))
dev.off()

# 16. Boxplots - AGNES NON-FRAUDES
png("clustering/Boxplot_AGNES_NonFraudes.png", width = 1200, height = 800)
par(mfrow = c(2, 2))
for (var in vars_available) {
  boxplot(data_nonfraude[[var]] ~ data_nonfraude_hc$Cluster,
          main = paste("AGNES NON-FRAUDES -", var),
          xlab = "Cluster",
          ylab = var)
}
par(mfrow = c(1, 1))
dev.off()