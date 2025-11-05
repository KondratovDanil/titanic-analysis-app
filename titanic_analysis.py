import pandas as pd

def analyze_survivors(df):
    """
    Подсчитывает количество выживших мужчин по каждому классу обслуживания,
    указывая минимальный и максимальный возраст.
    """
    # Проверка на наличие нужных столбцов
    required_cols = {"Survived", "Pclass", "Sex", "Age"}
    if not required_cols.issubset(df.columns):
        raise ValueError("В DataFrame отсутствуют необходимые столбцы!")

    # Фильтруем мужчин
    males = df[df["Sex"] == "male"]

    # Группируем по классу
    stats = males.groupby("Pclass").agg(
        survived_males=("Survived", "sum"),
        min_age=("Age", "min"),
        max_age=("Age", "max")
    ).reset_index()

    return stats
