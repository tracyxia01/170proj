from parse import read_input_file, write_output_file
import os

# original implementation:
'''
def solve(tasks):
    """
    Args:
        tasks: list[Task], list of igloos to polish
    Returns:
        output: list of igloos in order of polishing
    """
    # first we have to sort the file
    tasks.sort(key = lambda x: x.get_max_benefit())
    timeslots = [0] * 1440
    for task in tasks:
        id = task.get_task_id()
        deadline = task.get_deadline()
        duration = task.get_duration()
        find_slot(deadline, timeslots, id, duration)
    sequence = set(timeslots)
    sequence.discard(0)
    return list(sequence)

def find_slot(end_time, timeslots, id, duration):
    curr_end = end_time
    curr_start = curr_end - duration
    found = False
    while curr_start >= 0 and not found:
        if sum(timeslots[curr_start : curr_end]) == 0:
            for i in range(curr_start, curr_end):
                timeslots[i] = id
            found = True
        else:
            curr_end -=  1
            curr_start -= 1
'''


###
###
###     TODO: unit-test and input-test each step function and helper function and debug if needed
###     TODO: learn how to use cloud computing platforms if necessary
###     TODO: implement a non-deterministic solution by incorporating randomness
###
###



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
    # sort all tasks in increasing profit/duration ratio
    tasks.sort(key = lambda x: x.get_max_benefit() / x.get_duration())
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
            if if_enough_space(deadline, duration):
                # Step 2: start shifting tasks backwards from t = deadline - 1 to t = 0
                shift_tasks_forward(deadline, duration)
                place_task(id, start, duration)
            # Step 3: shift current task backward so that it finishes after its duration
            if_place_task = shift_tasks_backward(deadline, duration, i)
            # Step 4: if we decide not to place current task, revert timeslot to before step 2 condition
            if not if_place_task:
                timeslots = timeslots_copy
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
def shift_tasks_forward(deadline, duration):
    # note that if we run step 2, then there must be enough free space to place task 2
    counter = 0
    '''
    # two pointers, front and back
    back = deadline - 1
    for front in range(deadline - 1, -1, -1):
        if timeslots[front] == 0:

            # front != back only when there's task(s) in between the two pointers
            if front != back:
                # TODO need to implement correct shift
                shift_by_one(front, back, True)
                timeslots[front] = timeslots[back]
                timeslots[back] = 0 # back will always ending up pointing to the end of a task
            back -= 1
            counter += 1
            # if the length of the free space is equal to duration, then return; note that we want to shift minimally
            if counter == duration:

                return
    '''
    # front is ther pointer; iterate backwards
    for front in range(deadline - 1, -1, -1):
        if timeslots[front] == 0:
            counter += 1
            # if the length of the free space is equal to duration, then perform shifts return; note that we want to shift minimally
            if counter == duration:
                # performing shift
                actual_forward_shift(front, deadline)
                return

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
    # shift necessary shifts backwards first
    shifted_timeslots = []
    zero_counter = 0

    ###
    ### TODO: simplify logic for shifting "blue region" tasks backwards
    ###

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
        """
        # TODO: unsure how to handle exception here but pre_ID and post_ID should be the same
        if pre_ID != post_ID:
            return False
        """
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


# helper function
# traverse from time 0 to time deadline and see if total empty space from 0 to current task's ddl is
# larger than current task's duration. If not, then skip shifting and do step 3
def if_enough_space(deadline, duration):
    counter = 0
    for i in range(0, deadline):
        if timeslots[i] == 0:
            i += 1
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


if __name__ == '__main__':
    counter = 0
    for input_path in os.listdir('inputs/'):
        if input_path == 'small':
            for input_path2 in os.listdir('inputs/' + input_path):
                if counter < 10:
                    counter += 1
                    if input_path2 == '.ipynb_checkpoints' or input_path2 == '.DS_Store':
                        continue
                    output_path = 'outputs/' + input_path + '/' + input_path2[:-3] + '.out'
                    tasks = read_input_file('inputs/' + input_path + '/' + input_path2[:-3] + '.in')
                    output = solve(tasks)
                    write_output_file(output_path, output)
                else:
                    break
        else:
            continue
