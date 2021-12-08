from parse import read_input_file, write_output_file
import os
import copy
import random
import math
import numpy as np

"""
Functions:

Solve function:
    solve(tasks)

Main step functions:
    place_task_before_ddl(id, deadline, duration): step 1
    shift_tasks_forward(deadline, duration): step 2
    shift_tasks_backward(deadline, duration): step 3

Helper functions:
    if_enough_space(deadline, duration): check whether there's enough space to place current task,
        decide whether or not to run step 2
    place_task(id, start, duration): place current task into timeslots
    actual_forward_shift(front, deadline): function that performs the actual front shift work
"""

# global variable
timeslots = [0] * 1440  # timeslots : 0 ~ 1439
tasks_global = None             # current task

def solve(tasks):
    #print(2)
    """
    Args:
        tasks: list[Task], list of igloos to polish
    Returns:
        output: list of igloos in order of polishing
    """

    cluster = get_cluster(tasks)
    unsorted = copy.copy(tasks)
    # sort all tasks in increasing profit/duration ratio
    tasks.sort(reverse = True, key = lambda x: (x.get_max_benefit() / x.get_duration()))
    base = basic_greedy(tasks) # We first find the base
    base_objects = []
    for id in base:
        base_objects = base_objects + [unsorted[id-1]]
    base = base_objects # now base itself is a list of task objects
    #last_task = len(base)
    combined = base + tasks
    combined = set(combined)
    #seq = [x.get_task_id() for x in combined]
    # TEST TEST TEST TODO TODO
    last_task = find_last_task(tasks)
    res, index = simulated_annealing(list(tasks), last_task, cluster) # apply simulated annealing to the base array
    #print([task.get_task_id() for task in res[:index]])
    index = find_last_task(res)
    res = [task.get_task_id() for task in res[:index]]
    # if len(res) != len(set(res)):
    #     print('nooo!')
    return res

def get_cluster(tasks):
    """
    Args:
        tasks: List[Task]. The list of tasks we are working on
    Returns:
        clusters: A 2d array, where each elem contains everythin in cluster
        Also modifies the cluster of each task. 
    """
    data_points = []
    for tsk in tasks:
        data_points.append([tsk.get_duration(), tsk.get_deadline()])
    data_points = np.array(data_points)
    centroids = create_centroids(len(tasks))
    total_iteration = 100
    
    cluster_label = iterate_k_means(data_points, centroids, total_iteration)
    res = post_process(tasks, cluster_label)
    return res

def create_centroids(tot_len):
    # create some centroids
    # need to change those values. 
    """
    Args:
        tot_len: the total length of the array. Based on that we create different centroid
    """
    centroids = []
    cluster_num = 0
    if tot_len > 150:
        cluster_num = 1
    elif tot_len > 100:
        cluster_num = 1
    else:
        cluster_num = 1
    # small
    # medium
    # large
    for i in range(cluster_num):
        centroids.append([random.randint(0, 60), random.randint(0, 1440)])
    return centroids

def iterate_k_means(data_points, centroids, total_iteration):
    """
    Args:
        data_points: List[Task.duration, Task.deadline]
        centroids: the centroids we currently have
        total_iteration: number of iterations we do. Now it's 100. 
    Returns: 
        cluster_label: List[data_point, cluster_number]
            data_point: each individual data point in data_points
    """
    label = []
    cluster_label = []
    total_points = len(data_points)
    k = len(centroids)
    
    for iteration in range(0, total_iteration):
        for index_point in range(0, total_points):
            distance = {}
            for index_centroid in range(0, k):
                distance[index_centroid] = compute_euclidean_distance(data_points[index_point], centroids[index_centroid])
            label = assign_label_cluster(distance, data_points[index_point], centroids)
            centroids[label[0]] = compute_new_centroids(label[1], centroids[label[0]])

            if iteration == (total_iteration - 1):
                cluster_label.append(label)

    return cluster_label

def compute_euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid)**2))

def assign_label_cluster(distance, data_point, centroids):
    index_of_minimum = min(distance, key=distance.get)
    return [index_of_minimum, data_point, centroids[index_of_minimum]]

def compute_new_centroids(cluster_label, centroids):
    return np.array(cluster_label + centroids)/2

def post_process(tasks, result):
    """
    Does two things: 1. return a dictionary of clusters. 2. update all the clusters for each task
    Args:
        tasks: List[Task]
        result: List[cluster, data_point]
    Return:
        all_clusters: Dict{cluster: List[Tasks]}, tasks are grouped by clusters
    """
    # print("Result of k-Means Clustering: \n")
    res = []
    for data in result:
        # print("data point: {}".format(data[1]))
        # print("cluster number: {} \n".format(data[0]))
        res.append(data[0])
    # print(res)
    all_clusters = dict()
    for i in range(len(res)):
        all_clusters.setdefault(res[i], []).append(tasks[i])
        tasks[i].set_cluster(res[i])
    # print(res)
    # print([len(i) for i in all_clusters.values()])
    return all_clusters

def find_last_task(tasks):
    total_duration = 0
    last_task = 0
    for task in tasks:
        total_duration += task.get_duration()
        if total_duration > 1440:
            break
        last_task += 1
    return last_task


def basic_greedy(tasks):
    # tasks.sort(key = lambda x: (x.get_max_benefit()))
    global timeslots
    #set global tasks:
    global tasks_global
    tasks_global = tasks
    # start iteration; for loop is modified to obtain i
    for i in range(len(tasks)):
        task = tasks[i]
        id = task.get_task_id()
        deadline = task.get_deadline()
        duration = task.get_duration()
        # copy current timeslots for step 4
        timeslots_copy = timeslots.copy()
        # Step 1: place current tasks without any shifting,
        # and if we are not able to find free spaces without shifting, then enter step 2
        # Step 2-4 are for cases where we can't find places if we don't shift
        if not place_task_before_ddl(id, deadline, duration):
            # before running step 2, calcuate the amount of the free space from 0 to current deadline,
            # and if there are less space than duration, then skip step 2 and jump to step 3
            """
            # orignal solution; commented out for now to ignore running step 3
            if if_enough_space(deadline, duration):
                # Step 2: start shifting tasks backwards from t = deadline - 1 to t = 0
                shift_tasks_forward(deadline, duration)
                place_task(id, start, duration)
            # Step 3: shift current task backward so that it finishes after its duration
            if_place_task = shift_tasks_backward(deadline, duration, i)
            # Step 4: if we decide not to place current task, revert timeslot to before step 2 condition
            if not if_place_task:
                timeslots = timeslots_copy
            """

            # for now we don't run step 3
            if if_enough_space(deadline, duration):
                # Step 2: start shifting tasks backwards from t = deadline - 1 to t = 0
                start = shift_tasks_forward(deadline, duration)
                place_task(id, start, duration)
            else:
                timeslots = timeslots_copy

            """
            # nerfed step 3
            if if_enough_space(deadline, duration):
                # Step 2: start shifting tasks backwards from t = deadline - 1 to t = 0
                shift_tasks_forward(deadline, duration)
                place_task(id, start, duration)
            # Step 3: shift current task backward so that it finishes after its duration
            if_place_task = shift_tasks_backward(deadline, duration, i)
            # Step 4: if we decide not to place current task, revert timeslot to before step 2 condition
            if not if_place_task:
                timeslots = timeslots_copy
            """


    # format timeslots for the final output
    output_sequence = set(timeslots)
    output_sequence.discard(0)
    # reset timeslots; TODO: is this necessary here? Afraid that global variables won't reset between EACH PROBLEM
    timeslots = [0] * 1440
    tasks_global = None
    # return our outputs
    return list(output_sequence)


# step 1
# traverse backward in time, place task before deadline AND as late as possible
# returns a boolean of whether we are able to place current task within [0, ddl)
# if True, then timeslots will be updated
def place_task_before_ddl(id, deadline, duration):
    counter = 0
    # iterate backwards
    for i in range(deadline - 1, -1, -1):
        # if current timeslot is empty
        if timeslots[i] == 0:
            counter += 1
            # if there is a chuck of time with length duration
            if counter == duration:
                # update timeslots
                place_task(id, i, duration)
                return True
        # reset the counter of current free time chuck length
        else:
            counter = 0
    return False

# step 2
# shift tasks in backward seqence from t = deadline - 1 to t = 0 in an iterative way
# shift tasks forward
# determined the partial shift needed
# implemented with an one pointer approach
# returns the starting time of empty slots
def shift_tasks_forward(deadline, duration):
    # note that if we run step 2, then there must be enough free space to place task 2
    counter = 0

    # front is ther pointer; iterate backwards
    for front in range(deadline - 1, -1, -1):
        if timeslots[front] == 0:
            counter += 1
            # if the length of the free space is equal to duration, then perform shifts return; note that we want to shift minimally
            if counter == duration:
                # performing shift
                start = actual_forward_shift(front, deadline)
                return start

# TODO: step 3
# shift tasks in forward sequence from t = deadline to t = 1439 in an interative way
# shift tasks backward
# current_i: current iteration
# returns a boolean to determine whether to shift the tasks or not
def shift_tasks_backward(deadline, duration, current_i):
    # shift everything forward from 0 to deadline - 1
    actual_forward_shift(0, deadline)
    # find the start of the empty space
    start_empty = deadline - 1
    while timeslots[start_empty] == 0:
        start_empty -= 1
    start_empty += 1
    extra_space_needed = duration - (deadline - start_empty) # find how much back shifts needed
    delay = extra_space_needed
    # if current task is not the last task
    if current_i != len(tasks_global) - 1:
        # determine if we should place this task with reduced profit by comparing it with next most profitable task
        next_task = tasks_global[current_i + 1]
        next_ratio = next_task.get_max_benefit() / next_task.get_duration()
        current_ratio_with_delay = tasks_global[current_i].get_late_benefit(delay) / duration
        # if current_ratio_with_delay is less than 0.8 of the next best ratio, then skip current task
        if current_ratio_with_delay < (0.8 * next_ratio):  # 0.8 here is arbitrary
            return False
    # back_shift_end: the end on the timeslots for all tasks after deadline to shift backwards
    # timeslot[back_shift_end] is not included in the shifts; back_shift_end - 1 is included
    back_shift_end = deadline
    while back_shift_end < 1440 and extra_space_needed > 0:
        if timeslots[back_shift_end] == 0:
            extra_space_needed -= 1
        back_shift_end += 1



# helper function
# traverse from time 0 to time deadline and see if total empty space from 0 to current task's ddl is
# larger than current task's duration. If not, then skip shifting and do step 3
def if_enough_space(deadline, duration):
    counter = 0
    for i in range(0, deadline):
        if timeslots[i] == 0:
            counter += 1
        if counter >= duration:
            return True
    return False

# helper function
# place current task in timeslots
def place_task(id, start, duration):
    for i in range(start, start + duration):
        # in python, all parameters are passed by reference, thus we are modifying the original array here
        timeslots[i] = id

# helper function
# shift n time slots forward
# back - front = n, but in total there are n + 1 slots involved
# O(n) runtime
# returns the starting time of empty slots
def actual_forward_shift(front, deadline):
    shifted_timeslots = []
    zero_counter = 0
    # add all the non-zero slots to the temp shifted_timeslots first
    for i in range(front, deadline):
        if timeslots[i] != 0:
            shifted_timeslots.append(timeslots[i])
        else:
            zero_counter += 1
    # add the empty slots in the back
    shifted_timeslots = shifted_timeslots + [0] * zero_counter
    # copy the temp shifted_timeslots to the actual timeslots
    for i in range(front, deadline):
        timeslots[i] = shifted_timeslots[i - front]
    return deadline - zero_counter


def simulated_annealing(s, last_task, cluster):
    """
    Args:
        s: the initial lst for SA
        last_task: index of the last task completed before the deadline
        cluster: the dictionary of cluster. Use it to permute
    Returns:
        output: the final task array with optimal values
    """
    t = 100 # should be large. But need further testing
    k = 15000
    BOF = cost(s)[0]
    print("start:")
    print(BOF)
    best_order = s
    for i in range(k): # while we don't have a freeze
        #print(1)
        for j in range(1): # number of neighbors to visite per iteration. try to set it to 15
            old_cost, old_last_task = cost(s)
            # print(old_cost)
            s_prime = permute(s, last_task, cluster)
            new_cost, last_task = cost(s_prime)
            # print(new_cost)
            delta = new_cost - old_cost
            if delta > 0:
                s = s_prime
                if new_cost > BOF:
                    BOF = new_cost
                    best_order = s_prime
            else:
                # TODO
                # replace s = s_price with probability of e^(delta/t)
                if t > 0:
                    p = math.exp(delta/t)
                else:
                    p = 1
                epsilon = random.random() # random decimal between [0,1]
                if epsilon > p:
                    s = s_prime
                else:
                    last_task = old_last_task
        t = t-10

    #seq = [x.get_task_id() for x in s_prime[:last_task]]
    #print(last_task)
    print("end:")
    print(BOF)
    return best_order, last_task


def permute(task_lst, not_used, cluster):
    # TODO
    # given a task, we need to know where the num_b4_ddl_pass is
    """
    Permute the task_lst array
    """
    # first, get the cluster

    cp = copy.copy(task_lst)
    i = random.randint(0, not_used-1)
    task_i = cp[i]
    # find which cluster i belongs to
    cluster_of_i = cluster[task_i.get_cluster()]
    # choose task j within the cluster
    task_j = cluster_of_i[random.randint(0, len(cluster_of_i)-1)]
    # find the index of task j
    j = cp.index(task_j)

    # choose task k within the cluster
    task_k = cluster_of_i[random.randint(0, len(cluster_of_i)-1)]
    # find the index of task j
    k = cp.index(task_k)

    # swap i and j
    temp = cp[i]
    cp[i] = cp[j]
    cp[j] = temp

    # swap j and k. DOUBLE SWAP TO SEEE IF WE CAN GET A BETTER ANSWER
    temp2 = cp[j]
    cp[j] = cp[k]
    cp[k] = temp2
    return cp

def cost(task_lst):
    # TODO
    """
    Returns:
        total_cost: the total cost of this task list
        num_b4_ddl_pass: the index of last tasks can we actually completed before the deadline.
    """
    time_spent, total_cost, num_b4_ddl_pass = 0, 0, 0
    for task in task_lst:
        time_spent += task.get_duration()
        if time_spent > 1440:
            break
        ddl = task.get_deadline()
        total_cost += task.get_late_benefit(max(0, time_spent - ddl))
        num_b4_ddl_pass += 1
    return total_cost, num_b4_ddl_pass


if __name__ == '__main__':
    counter = 0
    for input_path in os.listdir('inputs/'):
        if input_path == '.DS_Store':# or input_path == 'small' or input_path == 'medium':
            continue
        for input_path2 in os.listdir('inputs/' + input_path):
            #if counter >= 10:
            #    break
            if input_path2 == '.ipynb_checkpoints' or input_path2 == '.DS_Store':
                continue
            output_path = 'outputs/' + input_path + '/' + input_path2[:-3] + '.out'
            tasks = read_input_file('inputs/' + input_path + '/' + input_path2[:-3] + '.in')
            output = solve(tasks)
            write_output_file(output_path, output)
            print(output_path)
            #counter += 1
        else:
            continue
