import pandas as pd
import numpy as np

# Создаем тестовые данные для демонстрации анализа
np.random.seed(42)
n_samples = 1000

# Генерируем тестовые данные
test_data = pd.DataFrame({
    'age': np.random.randint(18, 80, n_samples),
    'workclass': np.random.choice(['Private', 'Self-emp-not-inc', 'Federal-gov', 'Local-gov'], n_samples),
    'fnlwgt': np.random.randint(10000, 300000, n_samples),
    'education': np.random.choice(['Bachelors', 'HS-grad', 'Some-college', 'Masters', 'Doctorate'], n_samples),
    'education-num': np.random.randint(1, 16, n_samples),
    'marital-status': np.random.choice(['Married-civ-spouse', 'Never-married', 'Divorced', 'Widowed'], n_samples),
    'occupation': np.random.choice(['Tech-support', 'Sales', 'Exec-managerial', 'Prof-specialty'], n_samples),
    'relationship': np.random.choice(['Husband', 'Wife', 'Not-in-family', 'Own-child'], n_samples),
    'race': np.random.choice(['White', 'Black', 'Asian-Pac-Islander'], n_samples),
    'sex': np.random.choice(['Male', 'Female'], n_samples),
    'capital-gain': np.random.randint(0, 10000, n_samples),
    'capital-loss': np.random.randint(0, 2000, n_samples),
    'hours-per-week': np.random.randint(20, 80, n_samples),
    'native-country': np.random.choice(['United-States', 'Mexico', 'Canada', 'Germany'], n_samples),
    'salary': np.random.choice(['<=50K', '>50K'], n_samples, p=[0.75, 0.25])
})

data = test_data
print("Создан тестовый датасет для демонстрации")
print(f"Размер датасета: {data.shape}")
print("\nПервые 5 строк:")
print(data.head())

# загружаем датасет
data = pd.read_csv("./data/adult.data.csv")
data.head()

# 1. Посчитайте, сколько мужчин и женщин (признак sex) представлено в этом датасете
gender_count = data['sex'].value_counts()
print("Количество мужчин и женщин:")
print(gender_count)
print()

# 2. Каков средний возраст мужчин (признак age) по всему датасету?
male_avg_age = data[data['sex'] == 'Male']['age'].mean()
print(f"Средний возраст мужчин: {male_avg_age:.2f} лет")
print()

# 3. Какова доля граждан Соединенных Штатов (признак native-country)?
us_citizens_ratio = (data['native-country'] == 'United-States').mean()
print(f"Доля граждан США: {us_citizens_ratio:.2%}")
print()

# 4-5. Рассчитайте среднее значение и среднеквадратичное отклонение возраста тех, кто получает более 50K в год и тех, кто получает менее 50K в год
age_stats_by_salary = data.groupby('salary')['age'].agg(['mean', 'std'])
print("Статистика возраста по уровню дохода:")
print(age_stats_by_salary)
print()

# 6. Правда ли, что люди, которые получают больше 50k, имеют минимум высшее образование?
higher_education = ['Bachelors', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', 'Masters', 'Doctorate']

high_income = data[data['salary'] == '>50K']
has_higher_ed = high_income['education'].isin(higher_education).mean()

print(f"Доля людей с высшим образованием среди зарабатывающих >50K: {has_higher_ed:.2%}")

# Проверяем, есть ли люди без высшего образования среди высокооплачиваемых
has_no_higher_ed = (high_income['education'].isin(higher_education) == False).sum()
print(f"Количество людей без высшего образования среди зарабатывающих >50K: {has_no_higher_ed}")
print("Ответ: Нет, не все люди с доходом >50K имеют высшее образование")
print()

# 7. Выведите статистику возраста для каждой расы и каждого пола
age_stats = data.groupby(['race', 'sex'])['age'].describe()
print("Статистика возраста по расе и полу:")
print(age_stats)
print()

# Максимальный возраст мужчин расы Asian-Pac-Islander
max_age_asian_male = data[(data['race'] == 'Asian-Pac-Islander') & (data['sex'] == 'Male')]['age'].max()
print(f"Максимальный возраст мужчин расы Asian-Pac-Islander: {max_age_asian_male} лет")
print()

# 8. Среди кого больше доля зарабатывающих много (>50K): среди женатых или холостых мужчин?
male_data = data[data['sex'] == 'Male']

# Определяем семейное положение
married_statuses = ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse']
male_data['marital_group'] = male_data['marital-status'].apply(
    lambda x: 'married' if x in married_statuses else 'single'
)

# Считаем долю высокооплачиваемых
high_income_ratio_by_marital = male_data.groupby('marital_group')['salary'].apply(
    lambda x: (x == '>50K').mean()
)

print("Доля высокооплачиваемых среди мужчин по семейному положению:")
print(high_income_ratio_by_marital)
print()

# 9. Какое максимальное число часов человек работает в неделю?
max_hours = data['hours-per-week'].max()
print(f"Максимальное количество рабочих часов в неделю: {max_hours}")

# Люди, работающие максимальное количество часов
max_hours_workers = data[data['hours-per-week'] == max_hours]
count_max_hours = len(max_hours_workers)
high_income_ratio = (max_hours_workers['salary'] == '>50K').mean()

print(f"Количество людей, работающих {max_hours} часов в неделю: {count_max_hours}")
print(f"Доля высокооплачиваемых среди них: {high_income_ratio:.2%}")
print()

# 10. Посчитайте среднее время работы зарабатывающих мало и много для каждой страны
avg_hours_by_country_salary = data.groupby(['native-country', 'salary'])['hours-per-week'].mean()
print("Среднее время работы по странам и уровню дохода:")
print(avg_hours_by_country_salary.head(10))  # Показываем первые 10 строк
print()

# 11. Сгруппируйте людей по возрастным группам young, adult, retiree
def assign_age_group(age):
    if 16 <= age <= 35:
        return 'young'
    elif 35 < age <= 70:
        return 'adult'
    elif 70 < age <= 100:
        return 'retiree'
    else:
        return 'other'

data['AgeGroup'] = data['age'].apply(assign_age_group)
print("Распределение по возрастным группам:")
print(data['AgeGroup'].value_counts())
print()

# 12-13. Определите количество зарабатывающих >50K в каждой из возрастных групп
high_income_by_agegroup = data[data['salary'] == '>50K']['AgeGroup'].value_counts()
print("Количество высокооплачиваемых по возрастным группам:")
print(high_income_by_agegroup)
print()

# Группа с наибольшей долей высокооплачиваемых
high_income_ratio_by_group = data.groupby('AgeGroup')['salary'].apply(
    lambda x: (x == '>50K').mean()
)
max_ratio_group = high_income_ratio_by_group.idxmax()
max_ratio = high_income_ratio_by_group.max()

print(f"Возрастная группа с наибольшей долей высокооплачиваемых: {max_ratio_group} ({max_ratio:.2%})")
print()

# 14. Сгруппируйте людей по типу занятости и отфильтруйте группы
occupation_groups = data.groupby('occupation')

# Количество людей в каждой группе занятости
occupation_counts = occupation_groups.size()
print("Количество людей по типам занятости:")
print(occupation_counts)
print()

# Функция фильтрации
def filter_func(group):
    avg_age = group['age'].mean()
    min_hours = group['hours-per-week'].min()
    return avg_age <= 40 and min_hours > 5

# Применяем фильтрацию
filtered_occupations = []
for name, group in occupation_groups:
    if filter_func(group):
        filtered_occupations.append(name)

print("Типы занятости, прошедшие фильтрацию (средний возраст ≤ 40, все работают >5 часов):")
print(filtered_occupations)

# Дополнительно: общая информация о датасете
print("\n" + "="*50)
print("ОБЩАЯ ИНФОРМАЦИЯ О ДАТАСЕТЕ")
print("="*50)
print(f"Общее количество записей: {len(data)}")
print(f"Количество признаков: {len(data.columns)}")
print("\nТипы данных:")
print(data.dtypes)
print("\nПропущенные значения:")
print(data.isnull().sum())
