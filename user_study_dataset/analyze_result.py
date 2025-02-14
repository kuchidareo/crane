import numpy as np
import pandas as pd
import statsmodels.stats.multitest as smm
import statsmodels.stats.inter_rater as irr
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon 
import pingouin as pg


df = pd.read_csv('user-study-CRANE.csv')
df["score"] = pd.to_numeric(df["score"], errors="coerce")
df = df.dropna(subset=["score"])

"""
   label  image_id  annotator_id  score
0    HAR         1             1    2.0
1  Human         1             1    4.0
2    LLM         1             1    3.0
3    HAR         2             1    2.0
4  Human         2             1    4.0
"""

"""
HAR score   percentage
1.0    46.831276
2.0    31.111111
3.0    13.991770
4.0     4.362140
5.0     3.703704
dtype: float64
Human score   percentage
1.0     1.481481
2.0     5.843621
3.0    22.139918
4.0    35.308642
5.0    35.226337
dtype: float64
LLM score   percentage
1.0     0.987654
2.0     5.020576
3.0    22.469136
4.0    37.366255
5.0    34.156379
dtype: float64
"""

def compute_statistics(df):
    return {
        "Mean": np.mean(df["score"]),
        "Std Dev": np.std(df["score"], ddof=1),
        "Median": np.median(df["score"]),
        "IQR": np.percentile(df["score"], 75) - np.percentile(df["score"], 25),
        "Min": np.min(df["score"]),
        "Max": np.max(df["score"])
    }

def compute_icc(df):
    """Computes ICC using ANOVA method"""
    icc_list = []
    for label in df["label"].unique():
        df_label = df[df["label"] == label]
        icc = pg.intraclass_corr(data=df_label, targets='image_id', raters='annotator_id', ratings='score')
        print(icc)
        icc_list.append(icc)
    return 1

def cronbach_alpha(df):
    scores_matrix = df.pivot_table(index="image_id", columns="annotator_id", values="score")
    itemscores = scores_matrix.values  # Shape (num_images, num_raters)

    k = itemscores.shape[1]  # Number of annotators
    variances = np.var(itemscores, axis=1, ddof=1)  # Variance per image
    total_variance = np.var(itemscores.flatten(), ddof=1)  # Total variance

    alpha = (k / (k - 1)) * (1 - (np.sum(variances) / total_variance))
    return alpha

def compute_fleiss_kappa(df):
    """Converts ratings into a category count matrix for Fleiss' Kappa"""
    fleiss_kappa_list = []
    for label in df["label"].unique():
        df_label = df[df["label"] == label]
        matrix = df_label.pivot_table(index="image_id", columns="annotator_id", values="score")
        rating_matrix = matrix.to_numpy()
        print(rating_matrix)
        fleiss_kappa = irr.fleiss_kappa(rating_matrix, method='fleiss')
        print(fleiss_kappa)
        fleiss_kappa_list.append(fleiss_kappa)
    
    return np.mean(fleiss_kappa_list)

def statistial_analysis():
    har_df = df[df["label"] == "HAR"]
    human_df = df[df["label"] == "Human"]
    llm_df = df[df["label"] == "LLM"]
    
    har_stats = compute_statistics(har_df)
    human_stats = compute_statistics(human_df)
    llm_stats = compute_statistics(llm_df)
    stats_df = pd.DataFrame([har_stats, human_stats, llm_stats], index=["HAR", "Human", "LLM"])
    print("Descriptive Statistics:")
    print(stats_df)

    # Wilcoxon Signed-Rank Tests
    har_scores = har_df["score"].values
    human_scores = human_df["score"].values
    llm_scores = llm_df["score"].values
    
    w_har_human, p_har_human = wilcoxon(har_scores, human_scores)
    w_har_llm, p_har_llm = wilcoxon(har_scores, llm_scores)
    w_human_llm, p_human_llm = wilcoxon(human_scores, llm_scores)
    
    # Apply Bonferroni correction
    p_values = [p_har_human, p_har_llm, p_human_llm]
    reject, p_corrected, _, _ = smm.multipletests(p_values, alpha=0.05, method='bonferroni')
    
    print("Wilcoxon Signed-Rank Test Results (Bonferroni Corrected):")
    print(f"HAR vs Human: p={p_corrected[0]}, Reject Null: {reject[0]}")
    print(f"HAR vs LLM: p={p_corrected[1]}, Reject Null: {reject[1]}")
    print(f"Human vs LLM: p={p_corrected[2]}, Reject Null: {reject[2]}")
    
    # Interpretation
    print("\nInterpretation:")
    if any(reject):
        print("There are statistically significant differences among some groups.")
    else:
        print("No statistically significant differences were found among the groups.")
    
    icc_average_value = compute_icc(df)
    fleiss_kappa_average_value = compute_fleiss_kappa(df)
    agreement_df = pd.DataFrame({
        "ICC": [icc_average_value],
        "Fleiss' Kappa": [fleiss_kappa_average_value]
    })
    print("Inter-Rater Agreement Metrics:")
    print(agreement_df)
    
if __name__ == "__main__":
    print(df[(df["label"] == "Human") & (df["image_id"] == 1)])
    statistial_analysis()