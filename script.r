# ================================================================================
# PROJET D'ANALYSE DE DONNÉES - DÉTECTION DE FRAUDES DANS L'ASSURANCE
# ================================================================================
# Auteur: Marvin
# Date: Janvier 2026
# Objectif: Analyse de déclarations frauduleuses et construction d'un modèle
#           de prédiction des fraudes
# ================================================================================

# ================================================================================
# SECTION 0: INITIALISATION
# ================================================================================

# Installation des packages nécessaires (à exécuter une seule fois)
# install.packages(c("cluster", "rpart", "C50", "tree", "randomForest", "kknn", "ROCR", "caret", "ggplot2", "corrplot"))

# Chargement des bibliothèques
library(cluster)
library(rpart)
library(C50)
library(tree)
library(randomForest)
library(kknn)
library(ROCR)
library(caret)
library(ggplot2)
library(corrplot)

# Fixation de la graine aléatoire pour la reproductibilité des résultats
set.seed(123)

# Configuration des options d'affichage
options(scipen = 999) # Désactive la notation scientifique

# ================================================================================
# SECTION 1: CHARGEMENT ET TYPAGE DES DONNÉES
# ================================================================================

# Chargement du jeu de données principal (avec labels)
data <- read.csv("data/Data_Projet.csv", header = TRUE, sep = ",", dec = ".", stringsAsFactors = FALSE)

# Chargement du jeu de nouvelles déclarations à prédire (sans labels)
data_new <- read.csv("data/Data_Projet_New.csv", header = TRUE, sep = ",", dec = ".", stringsAsFactors = FALSE)

# Conversion des noms de colonnes en majuscules
names(data) <- toupper(names(data))
names(data_new) <- toupper(names(data_new))

# Conversion des variables catégorielles en facteurs pour le jeu principal
data$GENDER <- as.factor(data$GENDER)
data$INCIDENT_CAUSE <- as.factor(data$INCIDENT_CAUSE)
data$CLAIM_AREA <- as.factor(data$CLAIM_AREA)
data$POLICE_REPORT <- as.factor(data$POLICE_REPORT)
data$CLAIM_TYPE <- as.factor(data$CLAIM_TYPE)
data$FRAUDULENT <- as.factor(data$FRAUDULENT)

# Conversion des variables catégorielles en facteurs pour les nouvelles données
data_new$GENDER <- as.factor(data_new$GENDER)
data_new$INCIDENT_CAUSE <- as.factor(data_new$INCIDENT_CAUSE)
data_new$CLAIM_AREA <- as.factor(data_new$CLAIM_AREA)
data_new$POLICE_REPORT <- as.factor(data_new$POLICE_REPORT)
data_new$CLAIM_TYPE <- as.factor(data_new$CLAIM_TYPE)

cat("\n=== CHARGEMENT DES DONNÉES TERMINÉ ===\n")
cat("Nombre d'instances d'apprentissage:", nrow(data), "\n")
cat("Nombre d'instances à prédire:", nrow(data_new), "\n")

# ================================================================================
# SECTION 2: EXPLORATION ET ANALYSE DESCRIPTIVE DES DONNÉES
# ================================================================================

cat("\n=== EXPLORATION DES DONNÉES ===\n\n")

# Affichage de la structure des données
cat("--- Structure des données ---\n")
str(data)

# Résumé statistique des variables
cat("\n--- Résumé statistique ---\n")
summary(data)

# Vérification des valeurs manquantes
cat("\n--- Analyse des valeurs manquantes ---\n")
na_count <- sapply(data, function(x) sum(is.na(x)))
print(na_count)
cat("Pourcentage total de valeurs manquantes:", round(sum(na_count) / (nrow(data) * ncol(data)) * 100, 2), "%\n")

# Analyse de la variable cible FRAUDULENT
cat("\n--- Distribution de la variable cible FRAUDULENT ---\n")
table_fraudulent <- table(data$FRAUDULENT)
print(table_fraudulent)
prop_fraudulent <- prop.table(table_fraudulent)
print(round(prop_fraudulent * 100, 2))
cat("Taux de fraude:", round(prop_fraudulent["Yes"] * 100, 2), "%\n")

# Analyse des variables catégorielles
cat("\n--- Distribution des variables catégorielles ---\n")
cat("\nGENDER:\n")
print(table(data$GENDER))

cat("\nINCIDENT_CAUSE:\n")
print(table(data$INCIDENT_CAUSE))

cat("\nCLAIM_AREA:\n")
print(table(data$CLAIM_AREA))

cat("\nPOLICE_REPORT:\n")
print(table(data$POLICE_REPORT))

cat("\nCLAIM_TYPE:\n")
print(table(data$CLAIM_TYPE))

# Analyse des variables numériques
cat("\n--- Statistiques des variables numériques ---\n")
numeric_vars <- c("AGE", "DAYS_TO_INCIDENT", "CLAIM_AMOUNT", "TOTAL_POLICY_CLAIMS")
cat("\nAGE:\n")
summary(data$AGE)

cat("\nDAYS_TO_INCIDENT:\n")
summary(data$DAYS_TO_INCIDENT)

cat("\nCLAIM_AMOUNT:\n")
summary(data$CLAIM_AMOUNT)

cat("\nTOTAL_POLICY_CLAIMS:\n")
summary(data$TOTAL_POLICY_CLAIMS)

# Visualisations de base

# Barplot de la variable cible
png("img/plot_fraudulent_distribution.png", width = 800, height = 600)
barplot(table(data$FRAUDULENT), main = "Distribution de FRAUDULENT", xlab = "Classe", ylab = "Fréquence", col = c("lightgreen", "salmon"), ylim = c(0, max(table(data$FRAUDULENT)) * 1.2))
text(x = c(0.7, 1.9), y = table(data$FRAUDULENT) + 20, labels = table(data$FRAUDULENT))
dev.off()

# Boxplots des variables numériques par classe FRAUDULENT
png("img/plot_age_by_fraudulent.png", width = 800, height = 600)
boxplot(AGE ~ FRAUDULENT, data = data, main = "AGE par classe FRAUDULENT", xlab = "FRAUDULENT", ylab = "AGE", col = c("lightgreen", "salmon"))
dev.off()

png("img/plot_claim_amount_by_fraudulent.png", width = 800, height = 600)
boxplot(CLAIM_AMOUNT ~ FRAUDULENT, data = data, main = "CLAIM_AMOUNT par classe FRAUDULENT", xlab = "FRAUDULENT", ylab = "CLAIM_AMOUNT", col = c("lightgreen", "salmon"))
dev.off()

png("img/plot_days_to_incident_by_fraudulent.png", width = 800, height = 600)
boxplot(DAYS_TO_INCIDENT ~ FRAUDULENT, data = data, main = "DAYS_TO_INCIDENT par classe FRAUDULENT", xlab = "FRAUDULENT", ylab = "DAYS_TO_INCIDENT", col = c("lightgreen", "salmon"))
dev.off()

png("img/plot_total_policy_claims_by_fraudulent.png", width = 800, height = 600)
boxplot(TOTAL_POLICY_CLAIMS ~ FRAUDULENT, data = data, main = "TOTAL_POLICY_CLAIMS par classe FRAUDULENT", xlab = "FRAUDULENT", ylab = "TOTAL_POLICY_CLAIMS", col = c("lightgreen", "salmon"))
dev.off()

# Barplots des variables catégorielles par classe FRAUDULENT
png("img/plot_gender_by_fraudulent.png", width = 800, height = 600)
counts <- table(data$GENDER, data$FRAUDULENT)
barplot(counts, main = "GENDER par classe FRAUDULENT", xlab = "FRAUDULENT", ylab = "Fréquence", col = c("lightblue", "pink"), legend = rownames(counts), beside = TRUE)
dev.off()

png("img/plot_claim_area_by_fraudulent.png", width = 800, height = 600)
counts <- table(data$CLAIM_AREA, data$FRAUDULENT)
barplot(counts, main = "CLAIM_AREA par classe FRAUDULENT", xlab = "FRAUDULENT", ylab = "Fréquence", col = c("skyblue", "orange"), legend = rownames(counts), beside = TRUE)
dev.off()

png("img/plot_police_report_by_fraudulent.png", width = 800, height = 600)
counts <- table(data$POLICE_REPORT, data$FRAUDULENT)
barplot(counts, main = "POLICE_REPORT par classe FRAUDULENT", xlab = "FRAUDULENT", ylab = "Fréquence", col = c("lightgreen", "salmon", "lightyellow"), legend = rownames(counts), beside = TRUE)
dev.off()

png("img/plot_incident_cause_by_fraudulent.png", width = 800, height = 600)
counts <- table(data$INCIDENT_CAUSE, data$FRAUDULENT)
barplot(counts, main = "INCIDENT_CAUSE par classe FRAUDULENT", xlab = "FRAUDULENT", ylab = "Fréquence", col = rainbow(5), legend = rownames(counts), beside = TRUE)
dev.off()

png("img/plot_claim_type_by_fraudulent.png", width = 800, height = 600)
counts <- table(data$CLAIM_TYPE, data$FRAUDULENT)
barplot(counts, main = "CLAIM_TYPE par classe FRAUDULENT", xlab = "FRAUDULENT", ylab = "Fréquence", col = c("coral", "steelblue", "gold"), legend = rownames(counts), beside = TRUE)
dev.off()

# Matrice de corrélation des variables numériques
png("img/plot_correlation_matrix.png", width = 800, height = 800)
numeric_data <- data[, numeric_vars]
cor_matrix <- cor(numeric_data, use = "complete.obs")
corrplot(cor_matrix, method = "color", type = "upper", addCoef.col = "black", tl.col = "black", tl.srt = 45, title = "Matrice de corrélation des variables numériques", mar = c(0, 0, 2, 0))
dev.off()

cat("\n=== VISUALISATIONS SAUVEGARDÉES (fichiers PNG) ===\n")
cat("Note: Les graphiques AGE et GENDER seront régénérés après correction des données\n")

# ================================================================================
# SECTION 3: PRÉ-TRAITEMENTS ET NETTOYAGE DES DONNÉES
# ================================================================================

cat("\n=== PRÉ-TRAITEMENTS DES DONNÉES ===\n")

# --- CORRECTION 1: Normalisation de la casse pour GENDER ---
cat("\n--- Correction de la variable GENDER ---\n")
cat("Avant correction:\n")
print(table(data$GENDER))

# Uniformisation en majuscules
data$GENDER <- toupper(as.character(data$GENDER))
data$GENDER <- as.factor(data$GENDER)

data_new$GENDER <- toupper(as.character(data_new$GENDER))
data_new$GENDER <- as.factor(data_new$GENDER)

cat("\nAprès correction:\n")
print(table(data$GENDER))

# --- CORRECTION 2: Traitement des valeurs aberrantes d'AGE ---
cat("\n--- Correction des valeurs aberrantes pour AGE ---\n")
cat("Distribution AGE avant correction:\n")
cat("  Min:", min(data$AGE), "| Max:", max(data$AGE), "| Moyenne:", round(mean(data$AGE), 2), "\n")

# Identification des valeurs hors du domaine attendu [18, 79]
outliers_age <- data$AGE > 79 | data$AGE < 18
cat("Nombre de valeurs aberrantes AGE:", sum(outliers_age), 
    "(", round(sum(outliers_age) / nrow(data) * 100, 2), "%)\n")

if (sum(outliers_age) > 0) {
  cat("Exemples de valeurs aberrantes:\n")
  print(sort(unique(data$AGE[outliers_age]), decreasing = TRUE)[1:min(10, sum(outliers_age))])
  
  # Stratégie: Capping aux bornes du dictionnaire [18, 79]
  data$AGE[data$AGE > 79] <- 79
  data$AGE[data$AGE < 18] <- 18
  
  cat("\nCorrection appliquée: Capping aux bornes [18, 79]\n")
}

# Application de la même correction à data_new
outliers_age_new <- data_new$AGE > 79 | data_new$AGE < 18
if (sum(outliers_age_new) > 0) {
  data_new$AGE[data_new$AGE > 79] <- 79
  data_new$AGE[data_new$AGE < 18] <- 18
  cat("Correction appliquée à data_new:", sum(outliers_age_new), "valeurs\n")
}

cat("\nDistribution AGE après correction:\n")
cat("  Min:", min(data$AGE), "| Max:", max(data$AGE), "| Moyenne:", round(mean(data$AGE), 2), "\n")

# --- CORRECTION 3: Vérification de CLAIM_AMOUNT ---
cat("\n--- Vérification de CLAIM_AMOUNT ---\n")
cat("Min:", min(data$CLAIM_AMOUNT), "| Max:", max(data$CLAIM_AMOUNT), "\n")

outliers_claim <- data$CLAIM_AMOUNT > 47748
if (sum(outliers_claim) > 0) {
  cat("Valeurs légèrement au-dessus du max attendu (47748):", sum(outliers_claim), "\n")
  cat("Décision: Conservation (écart mineur)\n")
}

# Vérification des valeurs manquantes
cat("\n--- Analyse des valeurs manquantes ---\n")
na_count <- sapply(data, function(x) sum(is.na(x)))
print(na_count)

na_count_new <- sapply(data_new, function(x) sum(is.na(x)))
cat("\nValeurs manquantes dans data_new:\n")
print(na_count_new)

# Fonction pour calculer le mode
get_mode <- function(x) {
  ux <- unique(x[!is.na(x)])
  if (length(ux) == 0) return(NA)
  ux[which.max(tabulate(match(x, ux)))]
}

# Imputation des valeurs manquantes (si présentes)
if (sum(na_count) > 0) {
  cat("\nImputation des valeurs manquantes dans le jeu d'apprentissage...\n")
  
  for (col in names(data)) {
    if (sum(is.na(data[[col]])) > 0) {
      if (is.numeric(data[[col]])) {
        data[[col]][is.na(data[[col]])] <- median(data[[col]], na.rm = TRUE)
        cat("  -", col, ": imputation par la médiane\n")
      } else {
        data[[col]][is.na(data[[col]])] <- get_mode(data[[col]])
        cat("  -", col, ": imputation par le mode\n")
      }
    }
  }
}

if (sum(na_count_new) > 0) {
  cat("\nImputation des valeurs manquantes dans le jeu à prédire...\n")
  
  for (col in names(data_new)) {
    if (sum(is.na(data_new[[col]])) > 0) {
      if (is.numeric(data_new[[col]])) {
        # Utilise la médiane du jeu d'apprentissage pour cohérence
        if (col %in% names(data)) {
          data_new[[col]][is.na(data_new[[col]])] <- median(data[[col]], na.rm = TRUE)
        } else {
          data_new[[col]][is.na(data_new[[col]])] <- median(data_new[[col]], na.rm = TRUE)
        }
        cat("  -", col, ": imputation par la médiane\n")
      } else {
        # Utilise le mode du jeu d'apprentissage pour cohérence
        if (col %in% names(data)) {
          data_new[[col]][is.na(data_new[[col]])] <- get_mode(data[[col]])
        } else {
          data_new[[col]][is.na(data_new[[col]])] <- get_mode(data_new[[col]])
        }
        cat("  -", col, ": imputation par le mode\n")
      }
    }
  }
}

# Détection des valeurs aberrantes (outliers) pour les autres variables numériques
cat("\n--- Détection des valeurs aberrantes (méthode IQR) ---\n")

detect_outliers <- function(x) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower <- Q1 - 1.5 * IQR
  upper <- Q3 + 1.5 * IQR
  return(list(
    count = sum(x < lower | x > upper, na.rm = TRUE),
    lower = lower,
    upper = upper
  ))
}

numeric_vars <- c("AGE", "DAYS_TO_INCIDENT", "CLAIM_AMOUNT", "TOTAL_POLICY_CLAIMS")

for (var in numeric_vars) {
  outlier_info <- detect_outliers(data[[var]])
  cat(var, ": ", outlier_info$count, " valeurs aberrantes (",
      round(outlier_info$count / nrow(data) * 100, 2), "%) ",
      "| Bornes IQR: [", round(outlier_info$lower, 2), ", ", 
      round(outlier_info$upper, 2), "]\n", sep = "")
}

cat("\nDécision: Conservation des outliers (peuvent être informatifs pour la fraude)\n")
cat("Note: AGE déjà corrigé selon le dictionnaire de données\n")

# Harmonisation des niveaux de facteurs entre data et data_new
cat("\n--- Harmonisation des niveaux de facteurs ---\n")

categorical_vars <- c("GENDER", "INCIDENT_CAUSE", "CLAIM_AREA", "POLICE_REPORT", "CLAIM_TYPE")

for (var in categorical_vars) {
  all_levels <- union(levels(data[[var]]), levels(data_new[[var]]))
  data[[var]] <- factor(data[[var]], levels = all_levels)
  data_new[[var]] <- factor(data_new[[var]], levels = all_levels)
  cat(var, ": niveaux harmonisés -", length(all_levels), "niveaux\n")
  if (var == "GENDER" || var == "INCIDENT_CAUSE") {
    cat("  Niveaux:", paste(all_levels, collapse = ", "), "\n")
  }
}

# Préparation pour normalisation (sera appliquée spécifiquement pour k-NN)
# Sauvegarde des paramètres de normalisation APRÈS corrections
cat("\n--- Sauvegarde des paramètres de normalisation ---\n")
normalization_params <- list()
for (var in numeric_vars) {
  normalization_params[[var]] <- list(
    mean = mean(data[[var]], na.rm = TRUE),
    sd = sd(data[[var]], na.rm = TRUE)
  )
  cat(var, ": moyenne =", round(normalization_params[[var]]$mean, 2),
      "| sd =", round(normalization_params[[var]]$sd, 2), "\n")
}

cat("\n=== PRÉ-TRAITEMENTS TERMINÉS ===\n")
cat("Corrections appliquées:\n")
cat("  1. GENDER: normalisation de la casse (4 niveaux -> 2 niveaux)\n")
cat("  2. AGE: capping des valeurs aberrantes aux bornes [18, 79]\n")
cat("  3. Paramètres de normalisation sauvegardés pour k-NN\n")

# Régénération des visualisations avec données corrigées
cat("\n--- Régénération des visualisations avec données corrigées ---\n")

# Boxplot AGE corrigé
png("img/plot_age_by_fraudulent_cleaned.png", width = 800, height = 600)
boxplot(AGE ~ FRAUDULENT, data = data, 
        main = "AGE par classe FRAUDULENT (après correction)", 
        xlab = "FRAUDULENT", ylab = "AGE", 
        col = c("lightgreen", "salmon"))
dev.off()

# Barplot GENDER corrigé
png("img/plot_gender_by_fraudulent_cleaned.png", width = 800, height = 600)
counts <- table(data$GENDER, data$FRAUDULENT)
barplot(counts, 
        main = "GENDER par classe FRAUDULENT (après correction)", 
        xlab = "FRAUDULENT", ylab = "Fréquence", 
        col = c("lightblue", "pink"), 
        legend = rownames(counts), 
        beside = TRUE)
dev.off()

# Histogrammes des variables numériques après nettoyage
png("img/plot_age_histogram_cleaned.png", width = 800, height = 600)
hist(data$AGE, 
     main = "Distribution de AGE (après correction)", 
     xlab = "AGE", ylab = "Fréquence", 
     col = "steelblue", breaks = 20, 
     xlim = c(18, 79))
abline(v = mean(data$AGE), col = "red", lwd = 2, lty = 2)
legend("topright", legend = paste("Moyenne =", round(mean(data$AGE), 2)), 
       col = "red", lty = 2, lwd = 2)
dev.off()

png("img/plot_claim_amount_histogram.png", width = 800, height = 600)
hist(data$CLAIM_AMOUNT, 
     main = "Distribution de CLAIM_AMOUNT", 
     xlab = "CLAIM_AMOUNT", ylab = "Fréquence", 
     col = "coral", breaks = 30)
abline(v = mean(data$CLAIM_AMOUNT), col = "darkred", lwd = 2, lty = 2)
abline(v = median(data$CLAIM_AMOUNT), col = "blue", lwd = 2, lty = 2)
legend("topright", 
       legend = c(paste("Moyenne =", round(mean(data$CLAIM_AMOUNT), 0)),
                  paste("Médiane =", round(median(data$CLAIM_AMOUNT), 0))), 
       col = c("darkred", "blue"), lty = 2, lwd = 2)
dev.off()

png("img/plot_days_to_incident_histogram.png", width = 800, height = 600)
hist(data$DAYS_TO_INCIDENT, 
     main = "Distribution de DAYS_TO_INCIDENT", 
     xlab = "DAYS_TO_INCIDENT", ylab = "Fréquence", 
     col = "lightgreen", breaks = 30)
abline(v = median(data$DAYS_TO_INCIDENT), col = "darkgreen", lwd = 2, lty = 2)
legend("topright", 
       legend = paste("Médiane =", round(median(data$DAYS_TO_INCIDENT), 0)), 
       col = "darkgreen", lty = 2, lwd = 2)
dev.off()

# Comparaison avant/après pour AGE
png("img/plot_age_comparison.png", width = 1000, height = 500)
par(mfrow = c(1, 2))
# On ne peut pas montrer l'avant car déjà modifié, donc juste un texte informatif
plot(1, type = "n", axes = FALSE, xlab = "", ylab = "", 
     main = "AGE AVANT correction")
text(1, 1, "Max: 989 ans\nMoyenne: 87.48 ans\n100 valeurs aberrantes (7.69%)", 
     cex = 1.5, col = "red")
boxplot(data$AGE, main = "AGE APRÈS correction", 
        ylab = "AGE", col = "lightgreen")
text(1, 75, paste("Max:", max(data$AGE), "\nMoyenne:", round(mean(data$AGE), 2)), 
     cex = 1.2, col = "darkgreen")
par(mfrow = c(1, 1))
dev.off()

cat("Visualisations corrigées sauvegardées:\n")
cat("  - plot_age_by_fraudulent_cleaned.png\n")
cat("  - plot_gender_by_fraudulent_cleaned.png\n")
cat("  - plot_age_histogram_cleaned.png\n")
cat("  - plot_claim_amount_histogram.png\n")
cat("  - plot_days_to_incident_histogram.png\n")
cat("  - plot_age_comparison.png\n")

# ================================================================================
# SECTION 4: PARTITIONNEMENT APPRENTISSAGE/TEST
# ================================================================================

cat("\n=== PARTITIONNEMENT DES DONNÉES ===\n")

# Split stratifié 70% apprentissage / 30% test pour préserver la distribution de FRAUDULENT
trainIndex <- createDataPartition(data$FRAUDULENT, p = 0.7, list = FALSE, times = 1)

train_df <- data[trainIndex,]
test_df <- data[-trainIndex,]

cat("Taille du jeu d'apprentissage:", nrow(train_df), "\n")
cat("Taille du jeu de test:", nrow(test_df), "\n")

# Vérification de la distribution de FRAUDULENT dans chaque jeu
cat("\n--- Distribution de FRAUDULENT dans le jeu d'apprentissage ---\n")
print(table(train_df$FRAUDULENT))
print(round(prop.table(table(train_df$FRAUDULENT)) * 100, 2))

cat("\n--- Distribution de FRAUDULENT dans le jeu de test ---\n")
print(table(test_df$FRAUDULENT))
print(round(prop.table(table(test_df$FRAUDULENT)) * 100, 2))

cat("\n=== PARTITIONNEMENT TERMINÉ ===\n")

# ================================================================================
# SECTION 5: CLUSTERING EXPLORATOIRE PAR CLASSE
# ================================================================================

cat("\n=== CLUSTERING EXPLORATOIRE ===\n")
cat("Objectif: Identifier des sous-groupes homogènes dans chaque classe\n")
cat("          (FRAUDULENT = Yes et FRAUDULENT = No)\n\n")

# Séparation des données par classe pour clustering distinct
data_fraud <- data[data$FRAUDULENT == "Yes", ]
data_no_fraud <- data[data$FRAUDULENT == "No", ]

cat("Nombre de déclarations frauduleuses:", nrow(data_fraud), "\n")
cat("Nombre de déclarations non-frauduleuses:", nrow(data_no_fraud), "\n")

# Fonction pour calculer la distance de Gower et effectuer le clustering PAM
perform_clustering <- function(dataset, class_label, k_range = 2:6) {
  
  cat("\n--- Clustering pour la classe:", class_label, "---\n")
  
  # Sélection des variables pour le clustering (exclusion des ID et de la classe)
  vars_for_clustering <- c("AGE", "GENDER", "INCIDENT_CAUSE", "DAYS_TO_INCIDENT",
                          "CLAIM_AREA", "POLICE_REPORT", "CLAIM_TYPE", 
                          "CLAIM_AMOUNT", "TOTAL_POLICY_CLAIMS")
  
  cluster_data <- dataset[, vars_for_clustering]
  
  # Calcul de la matrice de distance de Gower (gère variables numériques et catégorielles)
  cat("Calcul de la matrice de distance de Gower...\n")
  gower_dist <- daisy(cluster_data, metric = "gower")
  
  # Recherche du nombre optimal de clusters via silhouette moyenne
  silhouette_scores <- numeric(length(k_range))
  
  for (i in seq_along(k_range)) {
    k <- k_range[i]
    pam_result <- pam(gower_dist, k = k, diss = TRUE)
    sil <- silhouette(pam_result$clustering, gower_dist)
    silhouette_scores[i] <- mean(sil[, 3])
    cat("  k =", k, "| Silhouette moyenne =", round(silhouette_scores[i], 4), "\n")
  }
  
  # Sélection du k optimal (silhouette maximale)
  optimal_k <- k_range[which.max(silhouette_scores)]
  cat("\nNombre optimal de clusters:", optimal_k, 
      "| Silhouette =", round(max(silhouette_scores), 4), "\n")
  
  # Clustering final avec k optimal
  pam_final <- pam(gower_dist, k = optimal_k, diss = TRUE)
  
  # Visualisation de la silhouette
  png(paste0("img/plot_silhouette_", gsub(" ", "_", class_label), ".png"), 
      width = 800, height = 600)
  plot(silhouette(pam_final$clustering, gower_dist),
       main = paste("Silhouette plot -", class_label),
       col = 2:(optimal_k + 1))
  dev.off()
  
  # Ajout des clusters au dataset
  dataset$Cluster <- pam_final$clustering
  
  # Caractérisation de chaque cluster
  cat("\n--- Caractérisation des clusters ---\n")
  
  for (cluster_id in 1:optimal_k) {
    cat("\n========== CLUSTER", cluster_id, "==========\n")
    cluster_subset <- dataset[dataset$Cluster == cluster_id, ]
    cat("Taille du cluster:", nrow(cluster_subset), 
        "(", round(nrow(cluster_subset) / nrow(dataset) * 100, 2), "%)\n")
    
    # Vérification de l'homogénéité par rapport à FRAUDULENT
    cat("\nDistribution de FRAUDULENT dans ce cluster:\n")
    print(table(cluster_subset$FRAUDULENT))
    cat("Pureté:", round(max(prop.table(table(cluster_subset$FRAUDULENT))) * 100, 2), "%\n")
    
    # Caractéristiques numériques: moyennes
    cat("\nVariables numériques (moyennes):\n")
    cat("  AGE:", round(mean(cluster_subset$AGE), 2), "\n")
    cat("  DAYS_TO_INCIDENT:", round(mean(cluster_subset$DAYS_TO_INCIDENT), 2), "\n")
    cat("  CLAIM_AMOUNT:", round(mean(cluster_subset$CLAIM_AMOUNT), 2), "\n")
    cat("  TOTAL_POLICY_CLAIMS:", round(mean(cluster_subset$TOTAL_POLICY_CLAIMS), 2), "\n")
    
    # Caractéristiques catégorielles: modes
    cat("\nVariables catégorielles (modes):\n")
    cat("  GENDER:", as.character(get_mode(cluster_subset$GENDER)), "\n")
    cat("  INCIDENT_CAUSE:", as.character(get_mode(cluster_subset$INCIDENT_CAUSE)), "\n")
    cat("  CLAIM_AREA:", as.character(get_mode(cluster_subset$CLAIM_AREA)), "\n")
    cat("  POLICE_REPORT:", as.character(get_mode(cluster_subset$POLICE_REPORT)), "\n")
    cat("  CLAIM_TYPE:", as.character(get_mode(cluster_subset$CLAIM_TYPE)), "\n")
    
    # Distribution des variables catégorielles importantes
    cat("\nDistribution GENDER:\n")
    print(table(cluster_subset$GENDER))
    
    cat("\nDistribution INCIDENT_CAUSE:\n")
    print(table(cluster_subset$INCIDENT_CAUSE))
    
    cat("\nDistribution POLICE_REPORT:\n")
    print(table(cluster_subset$POLICE_REPORT))
    
    cat("\nDistribution CLAIM_TYPE:\n")
    print(table(cluster_subset$CLAIM_TYPE))
  }
  
  # Graphique de distribution des clusters
  png(paste0("img/plot_cluster_sizes_", gsub(" ", "_", class_label), ".png"), 
      width = 800, height = 600)
  cluster_counts <- table(dataset$Cluster)
  barplot(cluster_counts,
          main = paste("Distribution des clusters -", class_label),
          xlab = "Cluster",
          ylab = "Nombre d'instances",
          col = rainbow(optimal_k),
          ylim = c(0, max(cluster_counts) * 1.2))
  text(x = seq_along(cluster_counts) * 1.2 - 0.5,
       y = cluster_counts + max(cluster_counts) * 0.05,
       labels = cluster_counts)
  dev.off()
  
  return(list(
    dataset = dataset,
    pam_model = pam_final,
    optimal_k = optimal_k,
    silhouette_scores = silhouette_scores
  ))
}

# Clustering des déclarations frauduleuses (FRAUDULENT = Yes)
cat("\n", paste(rep("=", 80), collapse = ""), "\n", sep = "")
cat("ANALYSE DES DÉCLARATIONS FRAUDULEUSES\n")
cat(paste(rep("=", 80), collapse = ""), "\n", sep = "")
fraud_clustering <- perform_clustering(data_fraud, "FRAUDULENT = Yes", k_range = 2:5)

# Clustering des déclarations non-frauduleuses (FRAUDULENT = No)
cat("\n" , paste(rep("=", 80), collapse = ""), "\n", sep = "")
cat("ANALYSE DES DÉCLARATIONS NON-FRAUDULEUSES\n")
cat(paste(rep("=", 80), collapse = ""), "\n", sep = "")
no_fraud_clustering <- perform_clustering(data_no_fraud, "FRAUDULENT = No", k_range = 2:5)

# Sauvegarde des résultats de clustering enrichis
data_fraud_clustered <- fraud_clustering$dataset
data_no_fraud_clustered <- no_fraud_clustering$dataset

# Reconstitution du jeu complet avec les clusters
data_with_clusters <- rbind(data_fraud_clustered, data_no_fraud_clustered)

# Analyse comparative des clusters entre les deux classes
cat("\n" , paste(rep("=", 80), collapse = ""), "\n", sep = "")
cat("COMPARAISON DES PROFILS DE CLUSTERS\n")
cat(paste(rep("=", 80), collapse = ""), "\n", sep = "")

cat("\nRésumé des clusters frauduleux:\n")
cat("  Nombre de clusters:", fraud_clustering$optimal_k, "\n")
for (i in 1:fraud_clustering$optimal_k) {
  cluster_subset <- data_fraud_clustered[data_fraud_clustered$Cluster == i, ]
  cat("  Cluster Fraud-", i, ": ", nrow(cluster_subset), " instances | ",
      "Age moyen: ", round(mean(cluster_subset$AGE), 1),
      " | Montant moyen: ", round(mean(cluster_subset$CLAIM_AMOUNT), 0), "\n", sep = "")
}

cat("\nRésumé des clusters non-frauduleux:\n")
cat("  Nombre de clusters:", no_fraud_clustering$optimal_k, "\n")
for (i in 1:no_fraud_clustering$optimal_k) {
  cluster_subset <- data_no_fraud_clustered[data_no_fraud_clustered$Cluster == i, ]
  cat("  Cluster NoFraud-", i, ": ", nrow(cluster_subset), " instances | ",
      "Age moyen: ", round(mean(cluster_subset$AGE), 1),
      " | Montant moyen: ", round(mean(cluster_subset$CLAIM_AMOUNT), 0), "\n", sep = "")
}

cat("\n=== CLUSTERING EXPLORATOIRE TERMINÉ ===\n")
cat("Les graphiques de silhouette et de distribution ont été sauvegardés.\n")

# ================================================================================
# SECTION 6: PRÉPARATION POUR LA MODÉLISATION SUPERVISÉE
# ================================================================================

cat("\n=== PRÉPARATION POUR LA MODÉLISATION SUPERVISÉE ===\n")

# Sélection des variables prédictives (exclusion des ID et de la classe)
predictive_vars <- c("AGE", "GENDER", "INCIDENT_CAUSE", "DAYS_TO_INCIDENT",
                     "CLAIM_AREA", "POLICE_REPORT", "CLAIM_TYPE", 
                     "CLAIM_AMOUNT", "TOTAL_POLICY_CLAIMS")

# Formule pour les modèles de classification
formula_classification <- as.formula("FRAUDULENT ~ AGE + GENDER + INCIDENT_CAUSE + 
                                      DAYS_TO_INCIDENT + CLAIM_AREA + POLICE_REPORT + 
                                      CLAIM_TYPE + CLAIM_AMOUNT + TOTAL_POLICY_CLAIMS")

cat("Variables prédictives sélectionnées:", length(predictive_vars), "\n")
print(predictive_vars)

# Préparation des données normalisées pour k-NN
# Normalisation des variables numériques (mean = 0, sd = 1)
train_df_scaled <- train_df
test_df_scaled <- test_df
data_new_scaled <- data_new

numeric_vars <- c("AGE", "DAYS_TO_INCIDENT", "CLAIM_AMOUNT", "TOTAL_POLICY_CLAIMS")

for (var in numeric_vars) {
  # Normalisation basée sur les paramètres du jeu d'apprentissage
  mean_val <- normalization_params[[var]]$mean
  sd_val <- normalization_params[[var]]$sd
  
  train_df_scaled[[var]] <- (train_df[[var]] - mean_val) / sd_val
  test_df_scaled[[var]] <- (test_df[[var]] - mean_val) / sd_val
  data_new_scaled[[var]] <- (data_new[[var]] - mean_val) / sd_val
}

cat("Normalisation des variables numériques effectuée pour k-NN.\n")

# Définition de la matrice de coûts (priorité : minimiser les faux négatifs)
# Coût plus élevé pour prédire No alors que c'est Yes (faux négatif)
cost_matrix <- matrix(c(0, 1,   # Vraie classe = No
                        5, 0),  # Vraie classe = Yes (coût FN = 5)
                      nrow = 2, byrow = TRUE,
                      dimnames = list(c("No", "Yes"), c("No", "Yes")))

cat("\nMatrice de coûts définie (pénalité FN = 5):\n")
print(cost_matrix)

cat("\n=== PRÉPARATION TERMINÉE ===\n")

# ================================================================================
# SECTION 7: ENTRAÎNEMENT DES MODÈLES DE CLASSIFICATION
# ================================================================================

cat("\n=== ENTRAÎNEMENT DES MODÈLES DE CLASSIFICATION ===\n")

# Stockage des modèles et de leurs prédictions
models_list <- list()
predictions_list <- list()

# --------------------------------------------------------------------------------
# Modèle 1: Arbre de décision (rpart)
# --------------------------------------------------------------------------------

cat("\n--- Modèle 1: Arbre de décision (rpart) ---\n")

# Configuration 1.1: rpart avec paramètres par défaut
cat("Configuration 1.1: Paramètres par défaut\n")
model_rpart_1 <- rpart(formula_classification, data = train_df, method = "class")
pred_rpart_1 <- predict(model_rpart_1, test_df, type = "class")
pred_rpart_1_prob <- predict(model_rpart_1, test_df, type = "prob")

# Configuration 1.2: rpart avec ajustement de cp et minsplit
cat("Configuration 1.2: cp = 0.01, minsplit = 20\n")
model_rpart_2 <- rpart(formula_classification, data = train_df, method = "class",
                       control = rpart.control(cp = 0.01, minsplit = 20))
pred_rpart_2 <- predict(model_rpart_2, test_df, type = "class")
pred_rpart_2_prob <- predict(model_rpart_2, test_df, type = "prob")

# Configuration 1.3: rpart avec matrice de coûts
cat("Configuration 1.3: Avec matrice de coûts (pénalité FN)\n")
model_rpart_3 <- rpart(formula_classification, data = train_df, method = "class",
                       parms = list(loss = cost_matrix))
pred_rpart_3 <- predict(model_rpart_3, test_df, type = "class")
pred_rpart_3_prob <- predict(model_rpart_3, test_df, type = "prob")

models_list$rpart_1 <- model_rpart_1
models_list$rpart_2 <- model_rpart_2
models_list$rpart_3 <- model_rpart_3

# --------------------------------------------------------------------------------
# Modèle 2: C5.0
# --------------------------------------------------------------------------------

cat("\n--- Modèle 2: C5.0 ---\n")

# Configuration 2.1: C5.0 avec paramètres par défaut
cat("Configuration 2.1: Paramètres par défaut\n")
model_c50_1 <- C5.0(formula_classification, data = train_df)
pred_c50_1 <- predict(model_c50_1, test_df, type = "class")
pred_c50_1_prob <- predict(model_c50_1, test_df, type = "prob")

# Configuration 2.2: C5.0 avec trials (boosting)
cat("Configuration 2.2: Avec boosting (trials = 10)\n")
model_c50_2 <- C5.0(formula_classification, data = train_df, trials = 10)
pred_c50_2 <- predict(model_c50_2, test_df, type = "class")
pred_c50_2_prob <- predict(model_c50_2, test_df, type = "prob")

# Configuration 2.3: C5.0 avec matrice de coûts
cat("Configuration 2.3: Avec matrice de coûts (pénalité FN)\n")
model_c50_3 <- C5.0(formula_classification, data = train_df, costs = cost_matrix)
pred_c50_3 <- predict(model_c50_3, test_df, type = "class")
# Note: Les probabilités ne sont pas disponibles avec la matrice de coûts dans C5.0
pred_c50_3_prob <- NULL

models_list$c50_1 <- model_c50_1
models_list$c50_2 <- model_c50_2
models_list$c50_3 <- model_c50_3

# --------------------------------------------------------------------------------
# Modèle 3: Random Forest
# --------------------------------------------------------------------------------

cat("\n--- Modèle 3: Random Forest ---\n")

# Configuration 3.1: Random Forest avec paramètres par défaut
cat("Configuration 3.1: Paramètres par défaut\n")
model_rf_1 <- randomForest(formula_classification, data = train_df, ntree = 500)
pred_rf_1 <- predict(model_rf_1, test_df, type = "class")
pred_rf_1_prob <- predict(model_rf_1, test_df, type = "prob")

# Configuration 3.2: Random Forest avec pondération des classes
cat("Configuration 3.2: Avec pondération (classwt pour minimiser FN)\n")
class_weights <- c(No = 1, Yes = 5)  # Poids plus élevé pour la classe Yes
model_rf_2 <- randomForest(formula_classification, data = train_df, ntree = 500,
                           classwt = class_weights)
pred_rf_2 <- predict(model_rf_2, test_df, type = "class")
pred_rf_2_prob <- predict(model_rf_2, test_df, type = "prob")

# Configuration 3.3: Random Forest avec optimisation de mtry
cat("Configuration 3.3: Optimisation de mtry\n")
model_rf_3 <- randomForest(formula_classification, data = train_df, ntree = 500,
                           mtry = 3, classwt = class_weights)
pred_rf_3 <- predict(model_rf_3, test_df, type = "class")
pred_rf_3_prob <- predict(model_rf_3, test_df, type = "prob")

models_list$rf_1 <- model_rf_1
models_list$rf_2 <- model_rf_2
models_list$rf_3 <- model_rf_3

cat("\nImportance des variables (Random Forest avec pondération):\n")
print(importance(model_rf_2))

# --------------------------------------------------------------------------------
# Modèle 4: k-NN
# --------------------------------------------------------------------------------

cat("\n--- Modèle 4: k-NN (sur données normalisées) ---\n")

# Préparation des données pour kknn
train_knn <- train_df_scaled[, c(predictive_vars, "FRAUDULENT")]
test_knn <- test_df_scaled[, c(predictive_vars, "FRAUDULENT")]

# Configuration 4.1: k-NN avec k = 5
cat("Configuration 4.1: k = 5\n")
model_knn_1 <- train.kknn(formula_classification, data = train_knn, kmax = 5, 
                          kernel = "optimal")
pred_knn_1 <- predict(model_knn_1, test_knn)
# Pour les probabilités, utilisation de kknn direct
knn_temp <- kknn(formula_classification, train_knn, test_knn, k = 5, kernel = "optimal")
pred_knn_1_prob <- knn_temp$prob

# Configuration 4.2: k-NN avec k = 7
cat("Configuration 4.2: k = 7\n")
model_knn_2 <- train.kknn(formula_classification, data = train_knn, kmax = 7, 
                          kernel = "optimal")
pred_knn_2 <- predict(model_knn_2, test_knn)
knn_temp <- kknn(formula_classification, train_knn, test_knn, k = 7, kernel = "optimal")
pred_knn_2_prob <- knn_temp$prob

# Configuration 4.3: k-NN avec recherche du k optimal
cat("Configuration 4.3: Recherche du k optimal (3 à 15)\n")
model_knn_3 <- train.kknn(formula_classification, data = train_knn, kmax = 15, 
                          kernel = "optimal")
pred_knn_3 <- predict(model_knn_3, test_knn)
best_k <- model_knn_3$best.parameters$k
cat("Meilleur k trouvé:", best_k, "\n")
knn_temp <- kknn(formula_classification, train_knn, test_knn, k = best_k, kernel = "optimal")
pred_knn_3_prob <- knn_temp$prob

models_list$knn_1 <- model_knn_1
models_list$knn_2 <- model_knn_2
models_list$knn_3 <- model_knn_3

cat("\n=== ENTRAÎNEMENT DES MODÈLES TERMINÉ ===\n")
cat("Nombre total de configurations testées:", length(models_list), "\n")

# ================================================================================
# SECTION 8: ÉVALUATION ET SÉLECTION DU MEILLEUR MODÈLE
# ================================================================================

cat("\n=== ÉVALUATION DES MODÈLES ===\n")
cat("Critère principal: Maximiser le recall de la classe FRAUDULENT = Yes\n")
cat("                   (minimiser les faux négatifs)\n\n")

# Fonction pour calculer les métriques d'évaluation
evaluate_model <- function(pred, pred_prob, actual, model_name, threshold = 0.5) {
  
  # Si seuil différent de 0.5, recalculer la prédiction
  if (threshold != 0.5 && !is.null(pred_prob)) {
    pred <- ifelse(pred_prob[, "Yes"] >= threshold, "Yes", "No")
    pred <- factor(pred, levels = levels(actual))
  }
  
  # Matrice de confusion
  cm <- confusionMatrix(pred, actual, positive = "Yes")
  
  # Extraction des métriques
  accuracy <- cm$overall["Accuracy"]
  sensitivity <- cm$byClass["Sensitivity"]  # Recall pour Yes (TP / (TP + FN))
  specificity <- cm$byClass["Specificity"]  # TN / (TN + FP)
  precision <- cm$byClass["Pos Pred Value"] # TP / (TP + FP)
  f1 <- cm$byClass["F1"]
  
  # Calcul de l'AUC si probabilités disponibles
  auc_value <- NA
  if (!is.null(pred_prob)) {
    pred_obj <- prediction(pred_prob[, "Yes"], actual)
    auc_value <- performance(pred_obj, "auc")@y.values[[1]]
  }
  
  # Matrice de confusion brute
  conf_matrix <- cm$table
  
  # Calcul des faux négatifs et faux positifs
  if ("Yes" %in% rownames(conf_matrix) && "Yes" %in% colnames(conf_matrix)) {
    TP <- conf_matrix["Yes", "Yes"]
    FN <- conf_matrix["No", "Yes"]
    FP <- conf_matrix["Yes", "No"]
    TN <- conf_matrix["No", "No"]
  } else {
    TP <- FN <- FP <- TN <- 0
  }
  
  return(list(
    model_name = model_name,
    threshold = threshold,
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    precision = precision,
    f1 = f1,
    auc = auc_value,
    confusion_matrix = conf_matrix,
    TP = TP, FN = FN, FP = FP, TN = TN,
    cm_object = cm
  ))
}

# Évaluation de tous les modèles
evaluation_results <- list()

cat("Évaluation avec seuil par défaut (0.5):\n")
cat(paste(rep("-", 80), collapse = ""), "\n")

evaluation_results[[1]] <- evaluate_model(pred_rpart_1, pred_rpart_1_prob, test_df$FRAUDULENT, "rpart_1_default")
evaluation_results[[2]] <- evaluate_model(pred_rpart_2, pred_rpart_2_prob, test_df$FRAUDULENT, "rpart_2_tuned")
evaluation_results[[3]] <- evaluate_model(pred_rpart_3, pred_rpart_3_prob, test_df$FRAUDULENT, "rpart_3_costs")
evaluation_results[[4]] <- evaluate_model(pred_c50_1, pred_c50_1_prob, test_df$FRAUDULENT, "c50_1_default")
evaluation_results[[5]] <- evaluate_model(pred_c50_2, pred_c50_2_prob, test_df$FRAUDULENT, "c50_2_boosted")
evaluation_results[[6]] <- evaluate_model(pred_c50_3, pred_c50_3_prob, test_df$FRAUDULENT, "c50_3_costs")
evaluation_results[[7]] <- evaluate_model(pred_rf_1, pred_rf_1_prob, test_df$FRAUDULENT, "rf_1_default")
evaluation_results[[8]] <- evaluate_model(pred_rf_2, pred_rf_2_prob, test_df$FRAUDULENT, "rf_2_weighted")
evaluation_results[[9]] <- evaluate_model(pred_rf_3, pred_rf_3_prob, test_df$FRAUDULENT, "rf_3_optimized")
evaluation_results[[10]] <- evaluate_model(pred_knn_1, pred_knn_1_prob, test_df$FRAUDULENT, "knn_1_k5")
evaluation_results[[11]] <- evaluate_model(pred_knn_2, pred_knn_2_prob, test_df$FRAUDULENT, "knn_2_k7")
evaluation_results[[12]] <- evaluate_model(pred_knn_3, pred_knn_3_prob, test_df$FRAUDULENT, "knn_3_optimal")

# Affichage des résultats
for (result in evaluation_results) {
  cat("\n", result$model_name, " (seuil = ", result$threshold, ")\n", sep = "")
  cat("  Accuracy:    ", round(result$accuracy, 4), "\n")
  cat("  Sensitivity: ", round(result$sensitivity, 4), " (Recall Fraude - À MAXIMISER)\n")
  cat("  Specificity: ", round(result$specificity, 4), "\n")
  cat("  Precision:   ", round(result$precision, 4), "\n")
  cat("  F1-Score:    ", round(result$f1, 4), "\n")
  if (!is.na(result$auc)) {
    cat("  AUC:         ", round(result$auc, 4), "\n")
  }
  cat("  FN (Faux Négatifs): ", result$FN, " | FP (Faux Positifs): ", result$FP, "\n")
}

# Création d'un tableau comparatif
comparison_df <- data.frame(
  Modele = sapply(evaluation_results, function(x) x$model_name),
  Accuracy = sapply(evaluation_results, function(x) round(x$accuracy, 4)),
  Sensitivity = sapply(evaluation_results, function(x) round(x$sensitivity, 4)),
  Specificity = sapply(evaluation_results, function(x) round(x$specificity, 4)),
  Precision = sapply(evaluation_results, function(x) round(x$precision, 4)),
  F1 = sapply(evaluation_results, function(x) round(x$f1, 4)),
  AUC = sapply(evaluation_results, function(x) round(x$auc, 4)),
  FN = sapply(evaluation_results, function(x) x$FN),
  FP = sapply(evaluation_results, function(x) x$FP)
)

cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("TABLEAU COMPARATIF DES PERFORMANCES\n")
cat(paste(rep("=", 80), collapse = ""), "\n\n")
print(comparison_df)

# Sélection du meilleur modèle basé sur le recall (sensitivity)
best_model_idx <- which.max(comparison_df$Sensitivity)
best_model_name <- comparison_df$Modele[best_model_idx]

cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("MEILLEUR MODÈLE SÉLECTIONNÉ (Recall maximal):", best_model_name, "\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat("  Sensitivity (Recall Fraude): ", comparison_df$Sensitivity[best_model_idx], "\n")
cat("  F1-Score:                    ", comparison_df$F1[best_model_idx], "\n")
cat("  Accuracy:                    ", comparison_df$Accuracy[best_model_idx], "\n")
cat("  Faux Négatifs:               ", comparison_df$FN[best_model_idx], "\n")

# Sauvegarde du tableau comparatif
write.csv(comparison_df, "results/model_comparison.csv", row.names = FALSE)
cat("\nTableau comparatif sauvegardé dans: results/model_comparison.csv\n")

# Test d'ajustement du seuil pour le meilleur modèle
cat("\n--- Optimisation du seuil de décision ---\n")
cat("Test de différents seuils pour maximiser le recall\n\n")

# Identification du modèle et des probabilités correspondantes
best_model_config <- strsplit(best_model_name, "_")[[1]][1]
best_pred_prob <- NULL

if (grepl("rpart", best_model_name)) {
  if (grepl("rpart_1", best_model_name)) best_pred_prob <- pred_rpart_1_prob
  else if (grepl("rpart_2", best_model_name)) best_pred_prob <- pred_rpart_2_prob
  else if (grepl("rpart_3", best_model_name)) best_pred_prob <- pred_rpart_3_prob
} else if (grepl("c50", best_model_name)) {
  if (grepl("c50_1", best_model_name)) best_pred_prob <- pred_c50_1_prob
  else if (grepl("c50_2", best_model_name)) best_pred_prob <- pred_c50_2_prob
  else if (grepl("c50_3", best_model_name)) best_pred_prob <- pred_c50_3_prob
} else if (grepl("rf", best_model_name)) {
  if (grepl("rf_1", best_model_name)) best_pred_prob <- pred_rf_1_prob
  else if (grepl("rf_2", best_model_name)) best_pred_prob <- pred_rf_2_prob
  else if (grepl("rf_3", best_model_name)) best_pred_prob <- pred_rf_3_prob
} else if (grepl("knn", best_model_name)) {
  if (grepl("knn_1", best_model_name)) best_pred_prob <- pred_knn_1_prob
  else if (grepl("knn_2", best_model_name)) best_pred_prob <- pred_knn_2_prob
  else if (grepl("knn_3", best_model_name)) best_pred_prob <- pred_knn_3_prob
}

# Test de différents seuils
thresholds <- seq(0.3, 0.7, by = 0.05)
threshold_results <- data.frame()

for (thresh in thresholds) {
  pred_adjusted <- ifelse(best_pred_prob[, "Yes"] >= thresh, "Yes", "No")
  pred_adjusted <- factor(pred_adjusted, levels = c("No", "Yes"))
  
  cm <- confusionMatrix(pred_adjusted, test_df$FRAUDULENT, positive = "Yes")
  
  threshold_results <- rbind(threshold_results, data.frame(
    Threshold = thresh,
    Sensitivity = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"],
    Precision = cm$byClass["Pos Pred Value"],
    F1 = cm$byClass["F1"],
    Accuracy = cm$overall["Accuracy"]
  ))
}

print(threshold_results)

# Sélection du seuil optimal (maximise sensitivity, puis F1)
optimal_threshold_idx <- which.max(threshold_results$Sensitivity)
optimal_threshold <- threshold_results$Threshold[optimal_threshold_idx]

cat("\nSeuil optimal sélectionné:", optimal_threshold, "\n")
cat("  Sensitivity: ", round(threshold_results$Sensitivity[optimal_threshold_idx], 4), "\n")
cat("  F1-Score:    ", round(threshold_results$F1[optimal_threshold_idx], 4), "\n")

cat("\n=== ÉVALUATION TERMINÉE ===\n")

# ================================================================================
# SECTION 9: ENTRAÎNEMENT FINAL ET PRÉDICTIONS
# ================================================================================

cat("\n=== ENTRAÎNEMENT FINAL DU MODÈLE SÉLECTIONNÉ ===\n")

# Refit du meilleur modèle sur l'ensemble complet des données étiquetées
# Cela améliore les performances en utilisant toutes les données disponibles
cat("\nRefit du modèle sur l'ensemble complet (train + test)...\n")

# Détermination du type de modèle sélectionné
final_model <- NULL
final_model_type <- strsplit(best_model_name, "_")[[1]][1]

if (final_model_type == "rpart") {
  cat("Type de modèle: Arbre de décision (rpart)\n")
  if (grepl("costs", best_model_name)) {
    cat("Configuration: Avec matrice de coûts\n")
    final_model <- rpart(formula_classification, data = data, method = "class",
                         parms = list(loss = cost_matrix))
  } else if (grepl("tuned", best_model_name)) {
    cat("Configuration: cp = 0.01, minsplit = 20\n")
    final_model <- rpart(formula_classification, data = data, method = "class",
                         control = rpart.control(cp = 0.01, minsplit = 20))
  } else {
    cat("Configuration: Paramètres par défaut\n")
    final_model <- rpart(formula_classification, data = data, method = "class")
  }
  
} else if (final_model_type == "c50") {
  cat("Type de modèle: C5.0\n")
  if (grepl("costs", best_model_name)) {
    cat("Configuration: Avec matrice de coûts\n")
    final_model <- C5.0(formula_classification, data = data, costs = cost_matrix)
  } else if (grepl("boosted", best_model_name)) {
    cat("Configuration: Boosting (trials = 10)\n")
    final_model <- C5.0(formula_classification, data = data, trials = 10)
  } else {
    cat("Configuration: Paramètres par défaut\n")
    final_model <- C5.0(formula_classification, data = data)
  }
  
} else if (final_model_type == "rf") {
  cat("Type de modèle: Random Forest\n")
  if (grepl("weighted", best_model_name)) {
    cat("Configuration: Avec pondération des classes\n")
    final_model <- randomForest(formula_classification, data = data, ntree = 500,
                                classwt = c(No = 1, Yes = 5))
  } else if (grepl("optimized", best_model_name)) {
    cat("Configuration: mtry = 3 avec pondération\n")
    final_model <- randomForest(formula_classification, data = data, ntree = 500,
                                mtry = 3, classwt = c(No = 1, Yes = 5))
  } else {
    cat("Configuration: Paramètres par défaut\n")
    final_model <- randomForest(formula_classification, data = data, ntree = 500)
  }
  
} else if (final_model_type == "knn") {
  cat("Type de modèle: k-NN\n")
  # Normalisation de l'ensemble complet
  data_scaled <- data
  for (var in numeric_vars) {
    mean_val <- normalization_params[[var]]$mean
    sd_val <- normalization_params[[var]]$sd
    data_scaled[[var]] <- (data[[var]] - mean_val) / sd_val
  }
  
  train_knn_full <- data_scaled[, c(predictive_vars, "FRAUDULENT")]
  
  if (grepl("optimal", best_model_name)) {
    cat("Configuration: k optimal (recherche 3-15)\n")
    final_model <- train.kknn(formula_classification, data = train_knn_full, 
                              kmax = 15, kernel = "optimal")
    cat("k optimal:", final_model$best.parameters$k, "\n")
  } else if (grepl("k7", best_model_name)) {
    cat("Configuration: k = 7\n")
    final_model <- train.kknn(formula_classification, data = train_knn_full, 
                              kmax = 7, kernel = "optimal")
  } else {
    cat("Configuration: k = 5\n")
    final_model <- train.kknn(formula_classification, data = train_knn_full, 
                              kmax = 5, kernel = "optimal")
  }
}

cat("\nModèle final entraîné avec succès.\n")

# Affichage du résumé du modèle final
cat("\n--- Résumé du modèle final ---\n")
if (final_model_type %in% c("rpart", "c50", "rf")) {
  print(summary(final_model))
}

# ================================================================================
# SECTION 10: PRÉDICTIONS SUR LES NOUVELLES DÉCLARATIONS
# ================================================================================

cat("\n=== PRÉDICTIONS SUR LES NOUVELLES DÉCLARATIONS ===\n")
cat("Nombre d'instances à prédire:", nrow(data_new), "\n")

# Application des prédictions sur data_new
predictions_new_prob <- NULL
predictions_new_class <- NULL

if (final_model_type == "rpart") {
  predictions_new_prob <- predict(final_model, data_new, type = "prob")
  predictions_new_class_default <- predict(final_model, data_new, type = "class")
  
} else if (final_model_type == "c50") {
  predictions_new_prob <- predict(final_model, data_new, type = "prob")
  predictions_new_class_default <- predict(final_model, data_new, type = "class")
  
} else if (final_model_type == "rf") {
  predictions_new_prob <- predict(final_model, data_new, type = "prob")
  predictions_new_class_default <- predict(final_model, data_new, type = "class")
  
} else if (final_model_type == "knn") {
  # Utilisation des données normalisées pour k-NN
  test_knn_new <- data_new_scaled[, predictive_vars]
  predictions_new_class_default <- predict(final_model, test_knn_new)
  
  # Calcul des probabilités avec kknn
  if (grepl("optimal", best_model_name)) {
    k_value <- final_model$best.parameters$k
  } else if (grepl("k7", best_model_name)) {
    k_value <- 7
  } else {
    k_value <- 5
  }
  
  knn_pred_new <- kknn(formula_classification, train_knn_full, 
                       cbind(test_knn_new, FRAUDULENT = factor(rep("No", nrow(test_knn_new)), 
                                                                levels = c("No", "Yes"))),
                       k = k_value, kernel = "optimal")
  predictions_new_prob <- knn_pred_new$prob
}

# Application du seuil optimal pour la classe prédite finale
predictions_new_class <- ifelse(predictions_new_prob[, "Yes"] >= optimal_threshold, 
                                "Yes", "No")

# Probabilité associée à la classe prédite
predictions_new_prob_final <- ifelse(predictions_new_class == "Yes",
                                    predictions_new_prob[, "Yes"],
                                    predictions_new_prob[, "No"])

# Création du data frame de résultats
results_df <- data.frame(
  CLAIM_ID = data_new$CLAIM_ID,
  CUSTOMER_ID = data_new$CUSTOMER_ID,
  PREDICTED_CLASS = predictions_new_class,
  PROBABILITY = round(predictions_new_prob_final, 4)
)

# Affichage d'un aperçu des résultats
cat("\n--- Aperçu des prédictions ---\n")
print(head(results_df, 10))

# Statistiques des prédictions
cat("\n--- Statistiques des prédictions ---\n")
cat("Distribution des classes prédites:\n")
print(table(results_df$PREDICTED_CLASS))
prop_pred <- prop.table(table(results_df$PREDICTED_CLASS))
cat("Proportions:\n")
print(round(prop_pred * 100, 2))

cat("\nProbabilités pour la classe FRAUDULENT = Yes:\n")
prob_yes <- predictions_new_prob[, "Yes"]
cat("  Minimum:  ", round(min(prob_yes), 4), "\n")
cat("  Maximum:  ", round(max(prob_yes), 4), "\n")
cat("  Moyenne:  ", round(mean(prob_yes), 4), "\n")
cat("  Médiane:  ", round(median(prob_yes), 4), "\n")

cat("\nProbabilités pour la classe FRAUDULENT = No:\n")
prob_no <- predictions_new_prob[, "No"]
cat("  Minimum:  ", round(min(prob_no), 4), "\n")
cat("  Maximum:  ", round(max(prob_no), 4), "\n")
cat("  Moyenne:  ", round(mean(prob_no), 4), "\n")
cat("  Médiane:  ", round(median(prob_no), 4), "\n")

# Sauvegarde des résultats dans un fichier CSV
write.csv(results_df, "results/predictions_fraudulent.csv", row.names = FALSE)

cat("\n=== PRÉDICTIONS TERMINÉES ===\n")
cat("Fichier de résultats sauvegardé:", output_file, "\n")

# ================================================================================
# SECTION 11: RAPPORT FINAL ET SYNTHÈSE
# ================================================================================

cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("SYNTHÈSE FINALE DU PROJET\n")
cat(paste(rep("=", 80), collapse = ""), "\n\n")

cat("1. DONNÉES\n")
cat("   - Instances d'apprentissage:", nrow(data), "\n")
cat("   - Instances de test:", nrow(test_df), "\n")
cat("   - Instances à prédire:", nrow(data_new), "\n")
cat("   - Taux de fraude dans les données:", 
    round(prop.table(table(data$FRAUDULENT))["Yes"] * 100, 2), "%\n\n")

cat("2. CLUSTERING\n")
cat("   - Clusters frauduleux identifiés:", fraud_clustering$optimal_k, "\n")
cat("   - Clusters non-frauduleux identifiés:", no_fraud_clustering$optimal_k, "\n")
cat("   - Méthode: PAM avec distance de Gower\n\n")

cat("3. MODÉLISATION\n")
cat("   - Nombre de configurations testées:", length(models_list), "\n")
cat("   - Algorithmes utilisés: rpart, C5.0, Random Forest, k-NN\n")
cat("   - Critère de sélection: Maximisation du recall (sensitivity) pour FRAUDULENT = Yes\n\n")

cat("4. MEILLEUR MODÈLE\n")
cat("   - Type:", best_model_name, "\n")
cat("   - Sensitivity (Recall Fraude):", comparison_df$Sensitivity[best_model_idx], "\n")
cat("   - F1-Score:", comparison_df$F1[best_model_idx], "\n")
cat("   - Accuracy:", comparison_df$Accuracy[best_model_idx], "\n")
cat("   - Seuil de décision optimal:", optimal_threshold, "\n")
cat("   - Faux Négatifs sur test:", comparison_df$FN[best_model_idx], "\n")
cat("   - Faux Positifs sur test:", comparison_df$FP[best_model_idx], "\n\n")

cat("5. PRÉDICTIONS SUR NOUVELLES DONNÉES\n")
cat("   - Nombre total de prédictions:", nrow(results_df), "\n")
cat("   - Fraudes prédites:", sum(results_df$PREDICTED_CLASS == "Yes"), 
    "(", round(sum(results_df$PREDICTED_CLASS == "Yes") / nrow(results_df) * 100, 2), "%)\n")
cat("   - Non-fraudes prédites:", sum(results_df$PREDICTED_CLASS == "No"),
    "(", round(sum(results_df$PREDICTED_CLASS == "No") / nrow(results_df) * 100, 2), "%)\n\n")

cat("6. FICHIERS GÉNÉRÉS\n")
cat("   - model_comparison.csv: Comparaison des performances des modèles\n")
cat("   - predictions_fraudulent.csv: Prédictions finales\n")
cat("   - Graphiques PNG: Visualisations exploratoires et clustering\n\n")

cat(paste(rep("=", 80), collapse = ""), "\n")
cat("PROJET TERMINÉ AVEC SUCCÈS\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat("\nDate d'exécution:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("\n=== FIN DU SCRIPT ===\n")