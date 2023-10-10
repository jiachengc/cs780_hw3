#/usr/bin/python
import os
import sys
import numpy as npy
from tqdm import tqdm
import pickle
from mnist import MNIST
import argparse
import torch
sys.path.append(os.path.dirname(__file__))
from net_ext import dot_product_vectors

def get_accuracy(model, data, lable, train=True):
	acc = 0
	for idx in tqdm(range(data.shape[0])):
		# idx = 0
		s = model.generate_prediction(data[idx])
		if s == npy.argmax(lable[idx]):
			acc +=1
	acc = (acc/data.shape[0])*100
	if train:
		model.training_accuracy = acc
	else:
		model.validation_accuracy = acc
	return acc
def fetch_model(model_name):
    neural_net = None
    model_path = f"./models/{model_name}.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as file:
            neural_net = pickle.load(file)
    else:
        print(f"Model not found at: {os.getcwd()}/models/{model_name}.pkl")
    return neural_net


def fetch_data(is_train_data=True, is_normalized=False):
    dataset = MNIST("./data/")
    d, l = None, None
    data_type = "Training" if is_train_data else "Testing"
    print(f"Loading {data_type} Data ...")
    if is_train_data:
        d, l = dataset.load_training()
    else:
        d, l = dataset.load_testing()

    d = npy.array(d)
    l = npy.array(l)

    if is_normalized:
        d = d / d.max()

    return d, l

def sigmoid_derivative(val):
    return val * (1 - val)
# Loss Function
def cross_entropy(predicted, actual):
    num_samples = actual.shape[0]
    diff = predicted - actual
    return diff / num_samples

# Error Function
def loss (predicted, actual):
    num_samples = actual.shape[0]
    log_prob = - npy.log(predicted[npy.arange(num_samples), actual.argmax(axis=1)])
    loss_value = npy.sum(log_prob) / num_samples
    return loss_value

# Create One Hot Vector Error Function
def convert_to_one_hot(labels):

	print("Creating One Hot Vector for Labels ...")
	one_hot_vector_size = npy.unique(labels).shape[0]
	labels_one_hot_array = npy.zeros((labels.shape[0], one_hot_vector_size))

	for idx in range(labels.shape[0]):
		# idx = 1
		labels_one_hot_array[idx][labels[idx]] = 1

	return labels_one_hot_array

def dot_product(a,b):
  if a.ndim == 1 and b.ndim == 1:
    return npy.dot(a, b)
  else:
    if a.ndim == 1:
      a = npy.reshape(a, (1,-1))
    return dot_product_vectors(npy.ascontiguousarray(a), npy.ascontiguousarray(b))


class NeuralNetwork:
	def __init__(self, data_input, label_output, epoch):
		self.input_data = data_input
		first_layer_neurons = 300
		second_layer_neurons = 100
		self.learning_rate = 0.6

		input_dimension = data_input.shape[1]
		output_dimension = label_output.shape[1]

		self.epoch = epoch
		self.training_accuracy = 0
		self.validation_accuracy = 0
		self.loss_record = []

		# Initializing weights and biases
		self.weight1 = npy.random.randn(input_dimension, first_layer_neurons)
		self.bias1 = npy.zeros((1, first_layer_neurons))
		self.weight2 = npy.random.randn(first_layer_neurons, second_layer_neurons)
		self.bias2 = npy.zeros((1, second_layer_neurons))
		self.weight3 = npy.random.randn(second_layer_neurons, output_dimension)
		self.bias3 = npy.zeros((1, output_dimension))
		self.output_data = label_output

	def forward_pass(self):
		# The rest of the code remains mostly unchanged, removed commented code for clarity
		z1 = torch.from_numpy(dot_product(self.input_data, self.weight1)).cuda() + torch.from_numpy(
			npy.ascontiguousarray(self.bias1)).cuda()
		self.layer1_output = torch.sigmoid(z1).cpu().numpy()

		z2 = torch.from_numpy(dot_product(self.layer1_output, self.weight2)).cuda() + torch.from_numpy(
			npy.ascontiguousarray(self.bias2)).cuda()
		self.layer2_output = torch.sigmoid(z2).cpu().numpy()

		z3 = torch.from_numpy(dot_product(self.layer2_output, self.weight3)).cuda() + torch.from_numpy(
			npy.ascontiguousarray(self.bias3)).cuda()
		self.final_output = torch.nn.functional.softmax(z3, dim=1).cpu().numpy()

	def backward_propagation(self):
		# The rest of the code remains mostly unchanged, removed commented code for clarity
		loss_value = loss(self.final_output, self.output_data)
		self.loss_record.append(loss_value)
		delta_layer3 = cross_entropy(self.final_output, self.output_data)
		delta_layer2 = dot_product(delta_layer3, self.weight3.T) * sigmoid_derivative(self.layer2_output)
		delta_layer1 = dot_product(delta_layer2, self.weight2.T) * sigmoid_derivative(self.layer1_output)

		self.weight3 -= self.learning_rate * dot_product(self.layer2_output.T, delta_layer3)
		self.bias3 -= self.learning_rate * npy.sum(delta_layer3, axis=0, keepdims=True)
		self.weight2 -= self.learning_rate * dot_product(self.layer1_output.T, delta_layer2)
		self.bias2 -= self.learning_rate * npy.sum(delta_layer2, axis=0)
		self.weight1 -= self.learning_rate * dot_product(self.input_data.T, delta_layer1)
		self.bias1 -= self.learning_rate * npy.sum(delta_layer1, axis=0)

	def generate_prediction(self, dataset):
		self.input_data = dataset
		self.forward_pass()
		return self.final_output.argmax()

	def store(self, model_filename):
		if not os.path.exists("./models/"):
			os.makedirs("./models/")
		model_path = f"./models/{model_filename}.pkl"
		print("Storing Neural Network Model ...")
		with open(model_path, "wb") as model_file:
			pickle.dump(self, model_file)



if __name__ == '__main__':

	argument_parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	argument_parser.add_argument("epochs_count", type=int, default=150, help="Number of epochs to train the model")
	parsed_args = argument_parser.parse_args()
	npy.random.seed(11)
	training_data, training_labels = fetch_data(is_normalized=True)
	training_labels = convert_to_one_hot(training_labels)

	neural_net = NeuralNetwork(training_data, npy.array(training_labels), parsed_args.epochs_count)
	for current_epoch in tqdm(range(neural_net.epoch)):
		neural_net.forward_pass()
		neural_net.backward_propagation()

	print(f"Epochs Trained: {neural_net.epoch}, Last Loss: {neural_net.loss_record[-1]}")
	print(f"Training accuracy: {get_accuracy(neural_net, training_data, npy.array(training_labels))}%")
	neural_net.store("mnist_neural_net")

	testing_data, testing_labels = fetch_data(is_train_data=False, is_normalized=True)
	loaded_model = fetch_model("mnist_neural_net")
	testing_labels = convert_to_one_hot(testing_labels)

	if loaded_model:
		print(
			f"Testing accuracy: {get_accuracy(loaded_model, testing_data, npy.array(testing_labels), train=False)}%")
		loaded_model.store("mnist_neural_net_updated")




