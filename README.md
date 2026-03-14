# 🚆 Prédiction des retards ferroviaires et estimation des compensations financières
### Nederlandse Spoorwegen (NS) — Projet Data Science | Centrale Méditerranée 2024

> **Question centrale :** Comment anticiper le risque financier lié aux retards ferroviaires aux Pays-Bas via le Machine Learning ?

---

## Vue d'ensemble

Ce projet prédit les classes de retard des services ferroviaires néerlandais et estime les compensations financières associées, en se basant sur les barèmes légaux NS et le règlement européen EU 2021/782.

Le défi principal est un **déséquilibre de classes extrême** : 91.77% des services arrivent à l'heure, tandis que les retards compensables (classes 2 et 3) représentent moins de 0.4% du dataset. L'évaluation repose donc non pas sur le F1 score classique, mais sur une **fonction de coût financier réel**.

---

## Dataset

**Source :** [Open Data Rijden de Treinen](https://www.rijdendetreinen.nl/en/open-data/train-archive)
**Période :** Année complète 2024
**Volume brut :** 21,857,914 lignes × 20 colonnes (~2.4 millions de services uniques)

Chaque ligne représente **un arrêt** d'un service dans une gare donnée. Les données ont été agrégées au niveau service pour la modélisation.

**Variables clés :**

| Colonne | Description |
|---|---|
| `Service:RDT-ID` | Identifiant unique du service |
| `Service:Date` | Date du service |
| `Service:Type` | Type de train (Intercity, Sprinter, ICE...) |
| `Service:Company` | Opérateur (NS, Arriva, Keolis...) |
| `Service:Maximum delay` | Retard maximum sur tous les arrêts (minutes) |
| `Stop:Arrival/Departure delay` | Retard à l'arrivée/départ par arrêt |
| `Stop:Arrival/Departure cancelled` | Annulations par arrêt |
| `Stop:Platform change` | Changement de quai |

---

## Classes Cibles

| Classe | Retard | Effectif (train) | % | Compensation NS |
|---|---|---|---|---|
| 0 | ≤ 5 min | 1,538,893 | 91.77% | Aucune |
| 1 | 5–30 min | 131,955 | 7.87% | Aucune |
| 2 | 30–60 min | 4,993 | 0.30% | **50% du billet** |
| 3 | > 60 min | 1,104 | 0.07% | **100% du billet** |

>  Le modèle voit **1 exemple de classe 3 pour 1,518 exemples** —> déséquilibre extrême.

---

## Stratégie d'Évaluation Financière

Plutôt que d'optimiser le F1 score, on minimise un **coût financier réel** basé sur les barèmes NS :

```
Prix moyen billet estimé  : €16.07  (calculé depuis les durées de service)
Passagers moyens/service  : 200
Taux de réclamation       : 30%
```

| Type d'erreur | Coût unitaire | Raison |
|---|---|---|
| Faux Positif | €50 | Coût opérationnel d'une alerte inutile |
| Faux Négatif classe 2 | €482 | €16.07 × 200 × 30% × 50% non provisionnés |
| Faux Négatif classe 3 | €964 | €16.07 × 200 × 30% × 100% non provisionnés |

> **Asymétrie clé :** Un FN classe 3 coûte ~19× plus qu'un faux positif.

---

## Architecture — Cascade 3 Étapes

Plutôt qu'un seul modèle multiclasse, on décompose le problème en **3 classificateurs binaires successifs** :

```
Étape 1 : Retard > 5 min ?        
    └── NON → Classe 0
    └── OUI ↓
Étape 2 : Retard > 30 min ?      
    └── NON → Classe 1
    └── OUI ↓
Étape 3 : Retard > 60 min ?     
    └── NON → Classe 2
    └── OUI → Classe 3
```

Chaque étape a son propre **seuil de décision optimisé par coût financier** sur le val set.

**Split chronologique 70/10/20** (pas de data leakage) :
- Train : 70% des services (ordre chronologique)
- Validation : 10% (sélection des hyperparamètres)
- Test : 20% (évaluation finale)

---

## Features (31 au total)

| Catégorie | Features |
|---|---|
| **Temporelles** | month, day_of_week, dep_hour, week, quarter, is_weekend, is_peak_hour, is_monday, is_friday, is_night, winter_month |
| **Structurelles** | n_stops, n_platform_changes, platform_change_rate, n_cancelled_stops, any_arr_cancelled, any_dep_cancelled, Svc_completely_cancelled, Svc_partly_cancelled, peak_x_weekday, long_route, has_cancellation, cancel_severity |
| **Encodées** | Service_Type_enc, Service_Company_enc, first_station_enc, last_station_enc |
| **Durée** | service_duration_min |
| **Historiques** | hist_delay_route, hist_delay_hour, hist_delay_type |

> Les features historiques sont calculées **uniquement sur le train set** pour éviter tout data leakage.

---

## Modèles

### Modèle 1 — Régression Logistique

Modèle linéaire servant de borne inférieure. Gestion du déséquilibre via `class_weight='balanced'`. `StandardScaler` appliqué (fit sur train uniquement).

| Métrique | Valeur |
|---|---|
| F1 Macro | 0.308 |
| F1 Weighted | 0.877 |
| AUC-ROC | 0.786 |
| **Coût financier** | **€1,054,834** |
| **vs. Naïf** | **-3.5% ↑ SURCOÛT ✗** |

> La LR génère trop de faux positifs (7,251) et coûte plus cher que de ne rien faire.

---

### Modèle 2 — LightGBM (Boosting par Gradient)

Arbres séquentiels avec `scale_pos_weight` sélectionné par coût financier. Early stopping sur 50 rounds.

| Métrique | Valeur |
|---|---|
| F1 Macro | 0.281 |
| F1 Weighted | 0.870 |
| AUC-ROC | 0.799 |
| **Coût financier** | **€900,018** |
| **vs. Naïf** | **+11.7% ↓ ÉCONOMIE ✓** |

> Économise €119K vs le naïf. Bien meilleur que la LR grâce à moins de faux positifs (1,050).

---

### Modèle 3 — Balanced Random Forest (Meilleur modèle)

Forêt aléatoire avec **under-sampling intégré par arbre** — conçu nativement pour les datasets déséquilibrés. `sampling_strategy=0.3` à l'étape 2 sélectionné par coût financier.

| Métrique | Valeur |
|---|---|
| F1 Macro | **0.305** |
| F1 Weighted | 0.870 |
| AUC-ROC | **0.808** |
| **Coût financier** | **€845,847** |
| **vs. Naïf** | **+17.0% ↓ ÉCONOMIE ✓** |
| **Économie annualisée** | **+€179,186/an** |

**Détail des erreurs (test set) :**
- FN classe 2 : 1,183 → €570,315 non provisionnés
- FN classe 3 : 214 → €206,335 non provisionnés
- FP : 1,490 → €74,500 de coûts opérationnels

> Seul modèle avec F1 > 0 sur la classe 2 (F1=0.063). Meilleur AUC, meilleur coût.

---

## Comparaison Finale

| Modèle | Coût Test | vs. Naïf | Économie annuelle |
|---|---|---|---|
| Baseline Naïve | €1,019,143 | 0% | — |
| Régression Logistique | €1,054,834 | -3.5% ✗ | -€37K |
| LightGBM | €900,018 | +11.7% ✓ | +€123K |
| **Balanced Random Forest** | **€845,847** | **+17.0% ✓** | **+€179K** |

---

## Application Streamlit

Une application de prédiction interactive a été développée avec **Streamlit**, intégrant le modèle BRF pour une démonstration en conditions réelles.

L'application permet de saisir les caractéristiques d'un service et retourne la classe de retard prédite, la compensation estimée en €, et les probabilités par classe.

---

## Auteurs

**Rym Bouziri, Alix Desmons, Aya Mahjoubi**
Centrale Méditerranée — Projet Data Science 2024
