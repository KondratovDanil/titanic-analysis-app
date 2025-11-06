import pytest
import pandas as pd
from titanic_analysis import analyze_survivors


# Тестовые данные
@pytest.fixture
def sample_data():
    data = {
        "Survived": [1, 0, 1, 1, 0, 1],
        "Pclass": [1, 1, 2, 3, 3, 3],
        "Sex": ["male", "female", "male", "male", "male", "female"],
        "Age": [35, 28, 40, 19, 22, 30]
    }
    return pd.DataFrame(data)


def test_returns_dataframe(sample_data):
    """Функция должна возвращать DataFrame."""
    result = analyze_survivors(sample_data)
    assert isinstance(result, pd.DataFrame)


def test_survived_males_count(sample_data):
    """Проверяем корректность подсчета выживших мужчин по классам."""
    result = analyze_survivors(sample_data)
    # 1 класс: 1 мужчина выжил; 2 класс: 1 мужчина выжил; 3 класс: 1 мужчина выжил
    expected = {1: 1, 2: 1, 3: 1}
    for _, row in result.iterrows():
        assert row["survived_males"] == expected[row["Pclass"]]


def test_age_range(sample_data):
    """Проверяем корректность диапазона возрастов по классам."""
    result = analyze_survivors(sample_data)
    # Для 3 класса: мужчины 19 и 22 лет
    class3 = result[result["Pclass"] == 3].iloc[0]
    assert class3["min_age"] == 19
    assert class3["max_age"] == 22


def test_missing_columns_error():
    """Проверяем, что функция выбрасывает ошибку при отсутствии нужных столбцов."""
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    with pytest.raises(ValueError):
        analyze_survivors(df)
