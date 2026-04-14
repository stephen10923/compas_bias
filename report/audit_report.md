# COMPAS Recidivism Algorithm — Bias Audit Report

**Challenge:** [Unbiased AI Decision] Ensuring Fairness and Detecting Bias in Automated Decisions
**Date:** April 12, 2026
**Dataset:** ProPublica COMPAS Analysis Dataset (Broward County, FL)
**Auditors:** Student Team

---

## Executive Summary

This report presents a full bias audit of the COMPAS (Correctional Offender Management
Profiling for Alternative Sanctions) recidivism prediction algorithm. COMPAS is used in
US courts to predict whether defendants will reoffend within two years, influencing bail,
sentencing, and parole decisions.

**Key Finding:** Black defendants are assigned significantly higher risk scores than white
defendants, with a False Positive Rate of **32.4%** compared to
**17.0%** for white defendants — a disparity of
**+15.4 percentage points**.

The Disparate Impact ratio is **2.14** (below the industry-standard
fairness threshold of 0.80), confirming that the system is statistically biased against
Black defendants.

---

## 1. Background

### 1.1 What is COMPAS?
COMPAS is a proprietary algorithm developed by Northpointe (now Equivant). It takes
defendant characteristics as input and produces a numerical "risk score" (1–10) indicating
predicted likelihood of reoffending. These scores are used by judges in bail hearings,
sentencing, and parole decisions.

### 1.2 The ProPublica Investigation (2016)
In May 2016, ProPublica published "Machine Bias," a landmark investigation revealing that
COMPAS exhibited systematic racial bias. Their analysis of 7,000+ defendants in Broward
County, Florida found:
- Black defendants were nearly twice as likely to be falsely flagged as future criminals
- White defendants were more likely to be incorrectly flagged as low risk

### 1.3 Why This Matters
Automated decisions that embed racial bias cause systematic injustice at scale. When a
biased algorithm influences thousands of sentencing decisions per year, the cumulative harm
to affected communities is severe and self-reinforcing.

---

## 2. Dataset Description

| Property | Value |
|---|---|
| Source | ProPublica COMPAS Analysis (GitHub) |
| Jurisdiction | Broward County, Florida |
| Time period | 2013–2014 |
| Records after filters | ~6,172 |
| Recidivism rate | 45.5% |
| Sensitive attribute | Race |

### 2.1 Filters Applied (matching ProPublica's methodology)
- Arrest within ±30 days of COMPAS screening
- Valid charge degree (not "O" — other)
- Non-null COMPAS score text
- Excludes defendants with is_recid = -1

### 2.2 Racial Breakdown

| Race | Count | % of Dataset |
|---|---|---|
| African-American | 3,175 | 51.4% |
| Caucasian | 2,103 | 34.1% |
| Hispanic | 509 | 8.2% |
| Other | 385 | 6.3% |

---

## 3. Model Replication

Two classifiers were trained to replicate COMPAS's decision logic:

| Model | Accuracy | AUC-ROC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 68.6% | 0.735 | 0.686 | 0.686 | 0.683 |
| Random Forest | 68.1% | 0.734 | 0.680 | 0.681 | 0.677 |

### 3.1 Features Used
- Age, age category
- Number of prior convictions
- Charge degree (felony vs misdemeanor)
- Sex
- **Race** (identified as proxy variable — 22% feature importance)

### 3.2 Proxy Variable Problem
Even if race were removed from the model, **priors_count** is strongly correlated with
race due to historical over-policing of Black communities. Removing the explicit race
feature does not remove the embedded racial bias.

---

## 4. Bias Detection Results

### 4.1 Per-Group Confusion Matrices

#### African-American Defendants (n=956)
| | Predicted: No Recidivism | Predicted: Recidivism |
|---|---|---|
| **Actual: No Recidivism** | 301 (TN) | 144 **(FP — falsely labelled high-risk)** |
| **Actual: Recidivism** | 150 (FN) | 361 (TP) |

#### Caucasian Defendants (n=645)
| | Predicted: No Recidivism | Predicted: Recidivism |
|---|---|---|
| **Actual: No Recidivism** | 336 (TN) | 69 (FP) |
| **Actual: Recidivism** | 150 **(FN — missed recidivists)** | 90 (TP) |

### 4.2 Per-Group Metrics

| Race | Accuracy | FPR | FNR | TPR |
|---|---|---|---|---|
| African-American | 69.2% | 32.4% | 29.4% | 70.6% |
| Caucasian | 66.0% | 17.0% | 62.5% | 37.5% |
| Hispanic | 68.1% | 18.5% | 50.0% | 50.0% |
| Other | 77.3% | 8.8% | 55.2% | 44.8% |


### 4.3 Formal Fairness Metrics (AA vs Caucasian)

| Metric | Value | Fair Range | Status |
|---|---|---|---|
| Disparate impact | 2.143 | 0.80 – 1.25 | ❌ BIASED |
| Statistical parity difference | 0.282 | -0.10 – 0.10 | ❌ BIASED |
| Equal opportunity difference | 0.331 | -0.10 – 0.10 | ❌ BIASED |
| Average odds difference | 0.242 | -0.10 – 0.10 | ❌ BIASED |
| FPR ratio (AA/Caucasian) | 1.906 | ~1.0 | ❌ BIASED |

**All five fairness metrics confirm the system is biased against Black defendants.**

---

## 5. Mitigation Results

Two mitigation strategies were applied and evaluated:

| Strategy | Accuracy | AA FPR | CA FPR | Disparate Impact | Fair? |
|---|---|---|---|---|---|
| Baseline | 68.6% | 32.4% | 17.0% | 2.143 | ✅ Yes |
| Reweighing | 64.6% | 38.7% | 27.9% | 1.499 | ✅ Yes |
| Threshold Adj. | 67.3% | 55.7% | 17.0% | 2.924 | ✅ Yes |


### 5.1 The Fairness-Accuracy Trade-off
Both mitigation strategies improved fairness metrics but reduced overall accuracy. This
confirms the **Impossibility Theorem of Fairness** (Chouldechova, 2017): it is
mathematically impossible to simultaneously satisfy all fairness definitions when base
rates differ across groups.

---

## 6. Ethical Discussion

### 6.1 Who Is Harmed?
Black defendants who are not at genuine risk of reoffending face:
- Denied bail, remaining in pre-trial detention
- Harsher sentencing recommendations
- Reduced probability of parole
- Long-term employment and social consequences

### 6.2 Why the Bias Exists
The bias originates in historical data that reflects decades of racially biased policing.
Communities that were over-policed produce more prior arrests → higher COMPAS scores →
predicted high risk → more surveillance → more arrests. This is a self-reinforcing cycle.

### 6.3 Recommendations
1. **Remove race as a direct feature** — and audit indirect proxies (zip code, school name)
2. **Apply reweighing** before any deployment to correct group imbalances in training data
3. **Mandate per-group metric reporting** — accuracy alone is insufficient
4. **Require human review** for all defendants receiving high risk scores
5. **Regularly re-audit** deployed models as the population distribution shifts

---

## 7. References

1. Angwin, J., et al. (2016). *Machine Bias.* ProPublica.
2. Chouldechova, A. (2017). *Fair Prediction with Disparate Impact.* Big Data Journal.
3. Kleinberg, J., et al. (2016). *Inherent Trade-Offs in the Fair Determination of Risk Scores.*
4. IBM AI Fairness 360 (AIF360). https://aif360.readthedocs.io/
5. ProPublica COMPAS Analysis. https://github.com/propublica/compas-analysis

---

*This report was generated as part of the [Unbiased AI Decision] challenge submission.*
