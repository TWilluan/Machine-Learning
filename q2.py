#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 20:54:50 2023

@author: tuanvo
"""

import numpy as np
from math import exp

#generate trainning dataset
def generate_training_dataset(L, N):
    values = np.random.exponential(L, N)
    labels = np.where(values > (1/L), "G", "S")
    dataset = list(zip(values, labels))
    return dataset
#generate testing dataset
def generate_test_dataset(L):
    values = np.random.exponential(L, 100)
    labels = np.where(values > (1/L), "G", "S")
    dataset = list(zip(values, labels))
    return dataset

# training model by calculating lambda and probabilities
def fit(dataset):
    # Calculate the lambda and count for each column in a dataset
    def calculate_lambda(dataset):
        unpacked = [column for column in zip(*dataset)]
        lambda_value = np.sum(unpacked[0])/len(unpacked[0])
        summaries = [(lambda_value, len(unpacked[0]))]
        return summaries
    
    #split the dataset according to label and calculate lambda and count of each label
    def lambda_by_class(dataset):
        # Split the dataset by class values, produce a dictionary
        separated = dict()
        for i in range(len(dataset)):
            data = dataset[i]
            label = data[-1]
            if (label not in separated):
                separated[label] = list()
            separated[label].append(data)
        
        #calculate lambda and count of each label G and S
        feature_lambda = dict()
        for class_value, rows in separated.items():
            feature_lambda[class_value] = calculate_lambda(rows)
        return feature_lambda
    
    # process fitting model
    return lambda_by_class(dataset)
    
    
def predict(model_train, row, total_rows):
    # Calculate the probability of exponentially distributed feature
    def calculate_probability(x, lambda_value):
        return lambda_value * exp(-(lambda_value * x))
    
    # Calculate the probability of feature based on class
    def class_probabilities(model_train, row, total_rows):
        probabilities = dict()
        for label, class_summaries in model_train.items():
            probabilities[label] = model_train[label][0][1]/float(total_rows)
            for i in range(len(class_summaries)):
                lambda_value = class_summaries[i][0]
                probabilities[label] *= calculate_probability(row[i], lambda_value)
        return probabilities
    
    # Predict label
    def produce_output(model_train, row, total_rows):
        label_prob = class_probabilities(model_train, row, total_rows)
        
        best_label = None
        best_prob = -1
        for label, prob in label_prob.items():
            if prob > best_prob:
                best_prob = prob
                best_label = label
        return best_label
    
    # process predicting label
    return produce_output(model_train, row, total_rows)

#evaluate the accuracy of model
def evaluate_model(train_dataset, test_dataset, N):
    #run naive_bayes to predict label
    def naive_bayes(train_dataset, test_dataset, N):
        model_train = fit(train_dataset)
        predictions = list()
        for row in test_dataset:
            output = predict(model_train, row, N)
            predictions.append(output)
        return predictions
    
    def accuracy(predictions):
        correct = 0
        for i in range(len(test_dataset)):
            true_output = test_dataset[i][1]
            if true_output == predictions[i]:
                correct += 1
        
        return correct / float(len(test_dataset)) * 100.0
    
    predictions = naive_bayes(train_dataset, test_dataset, N)
    return accuracy(predictions)

#main function
lambdas = [1, 1, 0.5]
nums    = [10, 200, 200]

for L, N in zip(lambdas, nums):
    train_dataset = generate_training_dataset(L, N)
    test_dataset  = generate_test_dataset(L)
    
    print("Trial: Lambda = ", L, ", N = ", N)
    print("The accuracy is: ", evaluate_model(train_dataset, test_dataset, N), "%")
    print("---------------------------")
