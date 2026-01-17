setwd("C:/Users/matlo/Documents/MIAGE/M2/Analyse_des_donnees/Projet")

# APPLICATION DU MEILLEUR MODÈLE AUX NOUVELLES DONNÉES
# Meilleur modèle sélectionné : C5.0 avec boosting (c50_boost)
# Critère de sélection : Rappel le plus élevé (0.3778)
# Objectif : Minimiser les faux négatifs (fraudes non détectées)
# 1. CHARGEMENT DES LIBRAIRIES
library(C50)

# 2. CHARGEMENT DES DONNÉES D'APPRENTISSAGE
cat("========== CHARGEMENT DES DONNÉES ==========\n")

# Données d'apprentissage (pour entraîner le modèle final)
data_train <- read.csv("Data_Projet.csv", header = TRUE, sep = ",", dec = ".",
                       stringsAsFactors = TRUE)

cat("Données d'apprentissage chargées:", nrow(data_train), "lignes\n")

# Suppression des identifiants (non prédictifs)
data_train <- subset(data_train, select = -c(claim_id, customer_id))

# Distribution des classes
cat("\nDistribution des classes (données d'apprentissage):\n")
print(table(data_train$fraudulent))

# 3. CHARGEMENT DES NOUVELLES DONNÉES À PRÉDIRE
data_new <- read.csv("Data_Projet_New.csv", header = TRUE, sep = ",", dec = ".",
                     stringsAsFactors = TRUE)

# Sauvegarde des identifiants pour le fichier final
claim_ids <- data_new$claim_id
customer_ids <- data_new$customer_id

# Suppression des identifiants pour la prédiction
data_new_pred <- subset(data_new, select = -c(claim_id, customer_id))

# 4. ENTRAÎNEMENT DU MODÈLE C5.0 AVEC BOOSTING
# Entraînement sur TOUTES les données (pour maximiser la performance)
set.seed(12345)  # Pour reproductibilité
modele_final <- C5.0(fraudulent ~ ., data = data_train, trials = 10)

# Affichage du résumé du modèle
print(summary(modele_final))

# 5. PRÉDICTION SUR LES NOUVELLES DONNÉES
# Prédiction des classes
predictions <- predict(modele_final, data_new_pred)

# Prédiction des probabilités
probabilites <- predict(modele_final, data_new_pred, type = "prob")

# 6. CRÉATION DU FICHIER DE RÉSULTATS
# Fichier complet avec probabilités
resultats <- data.frame(
  claim_id = claim_ids,
  customer_id = customer_ids,
  data_new_pred,
  fraudulent_predit = predictions,
  probabilite_fraude = round(probabilites[, "Yes"], 4),
  probabilite_non_fraude = round(probabilites[, "No"], 4)
)

# 7. STATISTIQUES DES PRÉDICTIONS
print(table(predictions))

nb_fraudes <- sum(predictions == "Yes")
nb_non_fraudes <- sum(predictions == "No")
pct_fraudes <- round(100 * nb_fraudes / length(predictions), 2)

cat("\nNombre de fraudes prédites:", nb_fraudes, "(", pct_fraudes, "%)\n")
cat("Nombre de non-fraudes prédites:", nb_non_fraudes, "\n")

# 8. SAUVEGARDE DU FICHIER
# Fichier de prédictions complet
write.csv(resultats, "Predictions_Fraudes.csv", row.names = FALSE)
cat("Fichier créé: Predictions_Fraudes.csv\n")