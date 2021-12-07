from parse import read_input_file, write_output_file
import os
import copy
import random
import math

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
tasks_global = None     # current task
# dimensions for the clusters
duration_length = 0
deadline_length = 0

def solve(tasks):
    #print(2)
    """
    Args:
        tasks: list[Task], list of igloos to polish
    Returns:
        output: list of igloos in order of polishing
    """
    global timeslots
    global task_global
    unsorted = copy.copy(tasks)
    # set the range of the 2D cluster in smart swapping needed in simulated annealing
    set_cluster_range(tasks)
    # sort all tasks in increasing profit/duration ratio
    tasks.sort(reverse = True, key = lambda x: (x.get_max_benefit() / x.get_duration()))
    # base = basic_greedy(tasks) # We first find the base
    base = tasks # tasks here are list of id's from the output file; thus we will run simulated annealing from our previous output file
    
    #print(base)
    base_objects = []
    for id in base:
        base_objects = base_objects + [unsorted[id-1]]
    base = base_objects # now base itself is a list of task objects
    old_cost = cost(base) # profit of previous output file
    #last_task = len(base)
    combined = base + tasks
    combined = set(combined)
    #seq = [x.get_task_id() for x in combined]
    last_task = find_last_task(combined)
    res, index = simulated_annealing(list(combined), last_task) # apply simulated annealing to the base array
    #print([task.get_task_id() for task in res[:index]])
    # if the profit computed from current simmulated annealing is worse than that of our old output file, then keep the old task list 
    if cost(res) < old_cost:
        res = base
    
    index = find_last_task(res)
    res = [task.get_task_id() for task in res[:index]]
    if len(res) != len(set(res)):
        print("omg dzdfghm")
    timeslots = [0] * 1440
    tasks_global = None
    return list(res)

# set deadline and duration range for the cluster
# will counstructed cluster based on this
def set_cluster_range(tasks):
    global duration_length
    global deadline_length
    # the dimensions of clusters differ based on the size of the problem (small/medium/large)
    if len(tasks) <= 100: # if small
        # split deadline into 3 parts and duration into 2 parts; 6 clusters total
        deadline_length, duration_length = (480, 30)
    elif len(tasks) <= 150: # if medium
        # split deadline into 4 parts and duration into 3 parts; 12 clusters total
        deadline_length, duration_length = (360, 20)
    else: #if large
        # split deadline into 5 parts and duration into 4 parts; 20 clusters total
        deadline_length, duration_length = (288, 15)

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

            # for now we don't run step 3
            if if_enough_space(deadline, duration):
                # Step 2: start shifting tasks backwards from t = deadline - 1 to t = 0
                start = shift_tasks_forward(deadline, duration)
                place_task(id, start, duration)
            else:
                timeslots = timeslots_copy

    # format timeslots for the final output
    output_sequence = set(timeslots)
    output_sequence.discard(0)
    # reset timeslots; TODO: is this necessary here? Afraid that global variables won't reset between EACH PROBLEM
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

    """

    ###
    ### TODO: simplify logic for shifting "blue region" tasks backwards
    ###

    # shift necessary shifts backwards first
    shifted_timeslots = []
    zero_counter = 0

    # add all the non-zero slots to the temp shifted_timeslots first
    for i in range(deadline, back_shift_end):
        if timeslots[i] != 0:
            shifted_timeslots.append(timeslots[i])
        else:
            zero_counter += 1
    # add the empt y slots to the front
    shifted_timeslots = [0] * zero_counter + shifted_timeslots
    # the two ptrs are starting ptrs
    preshift_ptr = deadline
    postshift_ptr = deadline
    loss = 0.0
    while preshift_ptr < back_shift_end and postshift_ptr < back_shift_end:
        while timeslots[preshift_ptr] == 0:
            preshift_ptr += 1
        while timeslots[postshift_ptr] == 0:
            postshift_ptr += 1
        # pre and post just means pre and post backward shifts
        pre_ID = timeslots[preshift_ptr]
        post_ID = timeslots[postshift_ptr]
        '''
        # TODO: unsure how to handle exception here but pre_ID and post_ID should be the same
        if pre_ID != post_ID:
            return False
        '''
        curr_task = tasks_global[timeslots[preshift_ptr]]
        pre_late_min = preshift_ptr + curr_task.get_duration() - curr_task.get_deadline()
        post_late_min = postshift_ptr + curr_task.get_duration() - curr_task.get_deadline()
        # also need to consider tasks in preshift might also be overtime resulted from a previous iteration
        pre_benefit = curr_task.get_max_benefit() if pre_late_min <= 0 else curr_task.get_late_benefit(pre_late_min)
        post_benefit = curr_task.get_max_benefit() if post_late_min <= 0 else curr_task.get_late_benefit(post_late_min)
        loss = loss + pre_benefit - post_benefit

    # Compare losses, and if the profit loss of later tasks in item caused by delay is more than the
    # partial profit brought by the current tasks, then don't add current task.
    # OPTIMIZATION BE LIKE BRRRRR yeah idk but like if even if we get more profits by adding current delay one still gotta compare next one ig lmao
    if current_ratio_with_delay - loss > 0.6 * next_task.get_max_benefit():
        return False
    # if we do include current task, copy the temp shifted_timeslots to the actual timeslots
    for i in range(deadline, back_shift_end):
        timeslots[i] = shifted_timeslots[i - deadline]
    return True
    """


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


def simulated_annealing(s, last_task):
    """
    Args:
        s: the initial lst for SA
        last_task: index of the last task completed before the deadline
    Returns:
        output: the final task array with optimal values
    """
    t = 5000 # should be large. But need further testing
    k = 50000
    while k > 0:
        s_prime = permute(s, last_task)
        old_cost, old_last_task = cost(s)
        new_cost, last_task = cost(s_prime)
        delta = new_cost - old_cost
        if delta > 0:
            s = s_prime
        else:
            # replace s = s_price with probability of e^(-delta/t)
            if t > 0:
                p = math.exp(delta/t)
            else:
                p = 1
            epsilon = random.random()
            if epsilon > p:
                s = s_prime
            else:
                last_task = old_last_task
        t -= 1
        k -= 1

    return s, last_task

def clusters(tasks, num, limit, dur = False):
    clusters = [[]] * num
    range = limit / num
    cluster_index = 0
    temp = []
    for task in tasks:
        if not dur:
            val = task.get_deadline()
        else:
            val = task.get_duration()
        if val < range:
            temp.append(task)
        else:
            if not dur:
                temp = temp.sort(key = lambda x: (x.duration))
                temp = clusters(temp, 12, 60, dur = True)
            clusters[cluster_index] = temp
            cluster_index += 1
            range += range
        range += 1
    return clusters



def permute(task_lst, not_used):
    # TODO
    # given a task, we need to know where the num_b4_ddl_pass is
    cp = copy.copy(task_lst)
    i = random.randint(0, not_used-1) # task_lst[i] is a task within our current output task list
    task = random_pick_in_cluster(task_lst[i]) # task_lst[j] isn't in the output task list
    j = task_lst.index(task)
    if i == j:
        j = random.randint(0, len(task_lst)-1)
    temp = cp[i]
    cp[i] = cp[j]
    cp[j] = temp
    return cp

# randomly pick another task that is within the cluster range of the current task; core function of the "smart random swap"
def random_pick_in_cluster(curr_task):
    curr_id = curr_task.get_task_id()
    curr_duration = curr_task.get_duration()
    curr_deadline = curr_task.get_deadline()
    potential_tasks = [] # list of tasks that are within the cluster aka the bound
    # iterate through all tasks
    for task in tasks_global:
        task_duration = task.get_duration()
        task_deadline = task.get_deadline()
        # automatically considers tasks that are close to the edge of the ranges of deadline and duration
        if_within_duration = abs(task_duration - curr_duration) <= duration_length / 2.0
        if_within_deadline = abs(task_deadline - curr_deadline) <= deadline_length / 2.0
        # if task within both bounds, then add it to potential_tasks
        if task.get_task_id() != curr_id and if_within_duration and if_within_deadline:
            potential_tasks += [task]
    # return a random task from potential_tasks
    if len(potential_tasks) <= 1:
        return curr_task
    return potential_tasks[random.randint(0, len(potential_tasks)-1)] #NOT SURE ABOUT THE -1 BUT IT WAS OUT OF BOUNDS


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
            # tasks = read_input_file('inputs/' + input_path + '/' + input_path2[:-3] + '.in')
            tasks = read_outputput_file('outputs/' + input_path + '/' + input_path2[:-3] + '.out') # tasks here should be a list of IDs
            
            print(input_path2)
            output = solve(tasks)
            write_output_file(output_path, output)
            #counter += 1
        else:
            continue
