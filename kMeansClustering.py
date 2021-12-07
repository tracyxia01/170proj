import numpy as np
import os
from parse import read_input_file, write_output_file
import copy
import random
import math

def compute_euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid)**2))

def assign_label_cluster(distance, data_point, centroids):
    index_of_minimum = min(distance, key=distance.get)
    return [index_of_minimum, data_point, centroids[index_of_minimum]]

def compute_new_centroids(cluster_label, centroids):
    return np.array(cluster_label + centroids)/2

def iterate_k_means(data_points, centroids, total_iteration):
    label = []
    cluster_label = []
    total_points = len(data_points)
    k = len(centroids)
    
    for iteration in range(0, total_iteration):
        for index_point in range(0, total_points):
            distance = {}
            for index_centroid in range(0, k):
                distance[index_centroid] = compute_euclidean_distance(data_points[index_point][1:], centroids[index_centroid])
            label = assign_label_cluster(distance, data_points[index_point][1:], centroids)
            centroids[label[0]] = compute_new_centroids(label[1], centroids[label[0]])

            if iteration == (total_iteration - 1):
                cluster_label.append(label)

    return [cluster_label, centroids]

def print_label_data(result):
    print("Result of k-Means Clustering: \n")
    for data in result[0]:
        print("data point: {}".format(data[1]))
        print("cluster number: {} \n".format(data[0]))
    print("Last centroids position: \n {}".format(result[1]))

def create_centroids():
    # create some centroids
    centroids = []
    centroids.append([5.0, 0.0])
    centroids.append([45.0, 70.0])
    centroids.append([50.0, 90.0])
    
    return centroids

if __name__ == "__main__":
    filename = os.path.dirname(__file__) + "\data.csv"
    # data_points = np.genfromtxt(filename, delimiter=",")
    data_points = []
    # data points now consists of the id, the duration and the deadline
    for tsk in tasks:
        data_points.append([tsk.get_task_id(), tsk.get_duration(), tsk.get_deadline()])
    data_points = np.array(data_points)
    centroids = create_centroids()
    total_iteration = 100
    
    [cluster_label, new_centroids] = iterate_k_means(data_points, centroids, total_iteration)
    print_label_data([cluster_label, new_centroids])
    print()