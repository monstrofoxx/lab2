"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    df = pd.read_csv(data.csv_path)

    row_count = int(len(df))
    missing_by_column = {
        str(col): int(cnt) for col, cnt in df.isna().sum().items()
    }
    adults_over_30_count = int((df["Age"] > 30).sum())
    mean_age_by_pclass = {
        int(pclass): float(val)
        for pclass, val in df.groupby("Pclass")["Age"].mean().items()
    }
    survival_rate_by_pclass = {
        int(pclass): float(val)
        for pclass, val in df.groupby("Pclass")["Survived"].mean().items()
    }
    highest_fares = [
        float(val) for val in df["Fare"].nlargest(5).tolist()
    ]

    return TitanicSummary(
        row_count=row_count,
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares,
    )