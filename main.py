import fetch_data
import manipulate_data
import matplotlib.pyplot as plt


fetch_data.fetch_housing_data()
housing = manipulate_data.load_housing_data()

head = housing.head()

print(head)

housing.hist(bins=50, figsize=(20,15))
plt.show()

train_set, test_set = manipulate_data.split_train_test(housing, 0.2)

print(len(train_set))
print(len(test_set))

housing["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = manipulate_data.split_train_test_by_id(housing, 0.2, "id")

head = housing.head()

