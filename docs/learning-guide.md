# Comprendre la classification

Les 16 entrées sont des caractéristiques déjà extraites par les auteurs du dataset. Lire leurs unités et noms ; aucune photo nouvelle n'est segmentée ici. La cible est une variété, pas une maladie ni la qualité nutritive.

Lire deduplicate avant split_data : garder des copies identiques dans train et test surestime le résultat. Les 68 copies sont retirées avant le split. Une étiquette contradictoire serait mise en quarantaine ; aucune n'a été observée.

Macro-F1 donne le même poids aux sept variétés. Le scaler apprend sur train seulement. Validation sélectionne le SVM et calibre sa température. Test évalue les choix figés ; ses erreurs ne servent pas à choisir un autre modèle.

Les probabilités sont une transformation calibrée des scores SVM. Le vote peut différer du plus grand score en cas d'égalité de votes ; les deux résultats sont visibles, sans modifier la décision après examen du test. Un faible ECE global ne suffit pas à valider la confiance sur un autre appareil photo.

Exercice : recomposer le macro-F1 depuis les prédictions, examiner DERMASON/SIRA puis expliquer pourquoi un split aléatoire de grains n'est pas une validation sur un lot indépendant.
