"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = pd.read_csv(data.csv_path, sep="\t", na_values="NA")
    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]

    women_df = df[df["Gender"] == "Female"]
    men_df = df[df["Gender"] == "Male"]

    women_count = int(len(women_df))
    men_count = int(len(men_df))

    women_mri_correlation = {
        feat: float(women_df[feat].corr(women_df["MRI_Count"], method="pearson"))
        for feat in features
    }
    men_mri_correlation = {
        feat: float(men_df[feat].corr(men_df["MRI_Count"], method="pearson"))
        for feat in features
    }

    combined = [
        (feat, abs(corr)) for feat, corr in women_mri_correlation.items()
    ] + [
        (feat, abs(corr)) for feat, corr in men_mri_correlation.items()
    ]
    strongest_mri_feature = max(combined, key=lambda item: item[1])[0]

    return BrainCorrelationSummary(
        men_count=men_count,
        women_count=women_count,
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_mri_feature,
    )