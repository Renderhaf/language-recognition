from NeuralNet import NeuralNet
import numpy as np
from sklearn.model_selection import train_test_split

data = np.load('training_data.npy', allow_pickle=True)
values = data[0]
results = data[1]
max_length = data[-1]

print(f"There are {len(values)} words in the dataset")

split_results = train_test_split(values, results, test_size=0.1)

split_results = [np.array(item) for item in split_results]

train_x, validate_x, train_y, validate_y = split_results

print(f"There are {len(train_x)} training items and {len(validate_x)} validation items")
print("Training data shape is ", train_x[0].shape, train_y[0].shape)
print("Validation data shape is ", validate_x[0].shape, validate_y[0].shape)

model = NeuralNet(max_length, len(data)-1)

print("Starting Training...")

model.fit(train_x, train_y, validate_x, validate_y, ephocs=200)

print("Finished Training!")