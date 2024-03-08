import fetch_data
import housing as hs

fetch_data.fetch_housing_data()
housing = hs.load_housing_data()

head = housing.head()

print(head)