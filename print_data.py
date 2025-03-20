import pandas as pd

data = pd.read_parquet("dataset.parquet")

print("Rozmiar danych (wiersze, kolumny):", data.shape)
print("\nNazwy kolumn:")
print(data.columns)

print("\nPierwsze 5 wierszy:")
print(data.head())
print("\nOstatnie 5 wierszy:")
print(data.tail())

print("\nInformacje o danych:")
print(data.info())

print("\nStatystyki opisowe:")
print(data.describe())

print("\nLiczba pustych wartości w każdej kolumnie:")
print(data.isnull().sum())

print("\nLiczba unikalnych wartości w każdej kolumnie:")
print(data.nunique())

puste_wartosci = data.isnull().sum().sum()
print(f"\nCzy dane zawierają puste wartości? {'Tak' if puste_wartosci > 0 else 'Nie'}, liczba: {puste_wartosci}")

kolumny_z_zerami = (data == 0).sum()
print("\nLiczba wartości zerowych w każdej kolumnie:")
print(kolumny_z_zerami[kolumny_z_zerami > 0])
print("\nTypy danych w kolumnach:")
print(data.dtypes)