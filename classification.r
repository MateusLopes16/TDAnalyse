setwd("C:/Users/matlo/Documents/MIAGE/M2/Analyse_des_donnees/Projet")

# 1. CHARGEMENT DES LIBRAIRIES
install.packages("rpart")
install.packages("rpart.plot")
install.packages("C50")
install.packages("tree")
install.packages("randomForest")
install.packages("kknn")
install.packages("e1071")
install.packages("nnet")
install.packages("ROCR")

library(rpart)
library(rpart.plot)
library(C50)
library(tree)
library(randomForest)
library(kknn)
library(e1071)
library(nnet)
library(ROCR)

# 2. CHARGEMENT ET PRÉPARATION DES DONNÉES
data <- read.csv("Data_Projet.csv", header = TRUE, sep = ",", dec = ".",
                 stringsAsFactors = TRUE)

str(data)
summary(data)
table(data$fraudulent)

# Suppression des identifiants (non prédictifs)
data <- subset(data, select = -c(claim_id, customer_id))

# Vérification des valeurs manquantes
sum(is.na(data))

# 3. SÉPARATION APPRENTISSAGE / TEST (70% / 30%)
set.seed(12345)  # Pour reproductibilité
n <- nrow(data)
train_index <- sample(1:n, size = 0.7 * n)

data_EA <- data[train_index, ]   # Ensemble d'apprentissage
data_ET <- data[-train_index, ]  # Ensemble de test

cat("Taille ensemble apprentissage:", nrow(data_EA), "\n")
cat("Taille ensemble test:", nrow(data_ET), "\n")
cat("\nDistribution des classes (apprentissage):\n")
print(table(data_EA$fraudulent))
cat("\nDistribution des classes (test):\n")
print(table(data_ET$fraudulent))

# 4. DÉFINITION DES MÉTRIQUES D'ÉVALUATION

evaluer_classifieur <- function(predictions, reelles, nom_modele) {
  # Matrice de confusion
  mc <- table(Réel = reelles, Prédit = predictions)
  print(mc)
  
  # Calcul des métriques
  # Pour la classe "Yes" (fraude)
  VP <- mc["Yes", "Yes"]  # Vrais Positifs (fraudes bien détectées)
  FN <- mc["Yes", "No"]   # Faux Négatifs (fraudes manquées)
  FP <- mc["No", "Yes"]   # Faux Positifs (fausses alertes)
  VN <- mc["No", "No"]    # Vrais Négatifs
  
  # Taux de succès global
  accuracy <- (VP + VN) / sum(mc)
  
  # Rappel (Sensibilité) - Capacité à détecter les fraudes
  recall <- VP / (VP + FN)
  
  # Précision - Parmi les alertes, combien sont vraies
  precision <- VP / (VP + FP)
  
  # F1-Score
  f1 <- 2 * (precision * recall) / (precision + recall)
  
  # Spécificité - Capacité à identifier les non-fraudes
  specificity <- VN / (VN + FP)
  
  return(list(
    accuracy = accuracy,
    recall = recall,
    precision = precision,
    f1 = f1,
    specificity = specificity,
    faux_negatifs = FN,
    matrice = mc
  ))
}

# 5. CONSTRUCTION DES CLASSIFIEURS

# Stockage des résultats
resultats <- list()

# 5.1 ARBRES DE DÉCISION - RPART
# Modèle 1: rpart avec paramètres par défaut
tree_rpart_1 <- rpart(fraudulent ~ ., data = data_EA, method = "class")
pred_rpart_1 <- predict(tree_rpart_1, data_ET, type = "class")
resultats$rpart_default <- evaluer_classifieur(pred_rpart_1, data_ET$fraudulent, "RPART (défaut)")

# Modèle 2: rpart avec gini, minbucket = 10
tree_rpart_2 <- rpart(fraudulent ~ ., data = data_EA, method = "class",
                      parms = list(split = "gini"),
                      control = rpart.control(minbucket = 10))
pred_rpart_2 <- predict(tree_rpart_2, data_ET, type = "class")
resultats$rpart_gini_10 <- evaluer_classifieur(pred_rpart_2, data_ET$fraudulent, "RPART (gini, minbucket=10)")

# Modèle 3: rpart avec information, minbucket = 5
tree_rpart_3 <- rpart(fraudulent ~ ., data = data_EA, method = "class",
                      parms = list(split = "information"),
                      control = rpart.control(minbucket = 5))
pred_rpart_3 <- predict(tree_rpart_3, data_ET, type = "class")
resultats$rpart_info_5 <- evaluer_classifieur(pred_rpart_3, data_ET$fraudulent, "RPART (information, minbucket=5)")

# Visualisation du meilleur arbre rpart
prp(tree_rpart_2, type = 2, extra = 104, fallen.leaves = TRUE, 
    main = "Arbre RPART - Détection de fraude")

# 5.2 ARBRES DE DÉCISION - C5.0
# Modèle 1: C5.0 par défaut
tree_c50_1 <- C5.0(fraudulent ~ ., data = data_EA)
pred_c50_1 <- predict(tree_c50_1, data_ET)
resultats$c50_default <- evaluer_classifieur(pred_c50_1, data_ET$fraudulent, "C5.0 (défaut)")

# Modèle 2: C5.0 avec minCases = 10
tree_c50_2 <- C5.0(fraudulent ~ ., data = data_EA, 
                   control = C5.0Control(minCases = 10))
pred_c50_2 <- predict(tree_c50_2, data_ET)
resultats$c50_min10 <- evaluer_classifieur(pred_c50_2, data_ET$fraudulent, "C5.0 (minCases=10)")

# Modèle 3: C5.0 avec boosting (trials = 10)
tree_c50_3 <- C5.0(fraudulent ~ ., data = data_EA, trials = 10)
pred_c50_3 <- predict(tree_c50_3, data_ET)
resultats$c50_boost <- evaluer_classifieur(pred_c50_3, data_ET$fraudulent, "C5.0 (boosting)")

# 5.3 ARBRES DE DÉCISION - TREE
# Modèle 1: tree par défaut
tree_tree_1 <- tree(fraudulent ~ ., data = data_EA)
pred_tree_1 <- predict(tree_tree_1, data_ET, type = "class")
resultats$tree_default <- evaluer_classifieur(pred_tree_1, data_ET$fraudulent, "TREE (défaut)")

# 5.4 RANDOM FOREST
# Modèle 1: randomForest par défaut
rf_1 <- randomForest(fraudulent ~ ., data = data_EA, ntree = 500)
pred_rf_1 <- predict(rf_1, data_ET)
resultats$rf_500 <- evaluer_classifieur(pred_rf_1, data_ET$fraudulent, "Random Forest (ntree=500)")

# Modèle 2: randomForest avec plus d'arbres
rf_2 <- randomForest(fraudulent ~ ., data = data_EA, ntree = 1000, mtry = 3)
pred_rf_2 <- predict(rf_2, data_ET)
resultats$rf_1000 <- evaluer_classifieur(pred_rf_2, data_ET$fraudulent, "Random Forest (ntree=1000, mtry=3)")

# 5.5 K-PLUS PROCHES VOISINS (KNN)
# Modèle 1: k = 5
knn_1 <- kknn(fraudulent ~ ., train = data_EA, test = data_ET, k = 5)
pred_knn_1 <- fitted(knn_1)
resultats$knn_5 <- evaluer_classifieur(pred_knn_1, data_ET$fraudulent, "KNN (k=5)")

# Modèle 2: k = 10
knn_2 <- kknn(fraudulent ~ ., train = data_EA, test = data_ET, k = 10)
pred_knn_2 <- fitted(knn_2)
resultats$knn_10 <- evaluer_classifieur(pred_knn_2, data_ET$fraudulent, "KNN (k=10)")

# 5.7 NAIVE BAYES
# Modèle 1: Naive Bayes par défaut
nb_1 <- naiveBayes(fraudulent ~ ., data = data_EA)
pred_nb_1 <- predict(nb_1, data_ET)
resultats$nb_default <- evaluer_classifieur(pred_nb_1, data_ET$fraudulent, "Naive Bayes (défaut)")

# Modèle 2: Naive Bayes avec laplace
nb_2 <- naiveBayes(fraudulent ~ ., data = data_EA, laplace = 1)
pred_nb_2 <- predict(nb_2, data_ET)
resultats$nb_laplace <- evaluer_classifieur(pred_nb_2, data_ET$fraudulent, "Naive Bayes (laplace=1)")

# 5.8 RÉSEAU DE NEURONES
# Modèle 1: nnet
nnet_1 <- nnet(fraudulent ~ ., data = data_EA, size = 10, decay = 0.01, 
               maxit = 200, trace = FALSE)
pred_nnet_1 <- predict(nnet_1, data_ET, type = "class")
resultats$nnet_10 <- evaluer_classifieur(pred_nnet_1, data_ET$fraudulent, "NNET (size=10)")

# Modèle 2: nnet avec plus de neurones
nnet_2 <- nnet(fraudulent ~ ., data = data_EA, size = 25, decay = 0.001, 
               maxit = 300, trace = FALSE)
pred_nnet_2 <- predict(nnet_2, data_ET, type = "class")
resultats$nnet_25 <- evaluer_classifieur(pred_nnet_2, data_ET$fraudulent, "NNET (size=25)")

# 6. COMPARAISON DES MODÈLES
# Création du tableau comparatif
comparaison <- data.frame(
  Modele = names(resultats),
  Accuracy = sapply(resultats, function(x) round(x$accuracy, 4)),
  Rappel = sapply(resultats, function(x) round(x$recall, 4)),
  Precision = sapply(resultats, function(x) round(x$precision, 4)),
  F1_Score = sapply(resultats, function(x) round(x$f1, 4)),
  Faux_Negatifs = sapply(resultats, function(x) x$faux_negatifs)
)

# Tri par Rappel décroissant (critère principal)
comparaison <- comparaison[order(-comparaison$Rappel), ]
rownames(comparaison) <- NULL
print(comparaison)

# 7. SAUVEGARDE DES RÉSULTATS
write.csv(comparaison, "classification/comparaison_modeles.csv", row.names = FALSE)

cat("\n\nAnalyse terminée. Résultats sauvegardés dans le dossier 'classification/'\n")