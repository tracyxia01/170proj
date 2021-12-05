from parse import read_input_file, write_output_file
import os
import copy
import random
import math

def solve(tasks):
    """
    Args:
        tasks: list[Task], list of igloos to polish
    Returns:
        output: list of igloos in order of polishing
    """
    base, last_task = basic_greedy(tasks) # We first find the base 
    res, index = simulated_annealing(base, last_task) # apply simulated annealing to the base array
    i = for_the_vibe(res)
    x = [task.get_task_id() for task in res[:i]]
    tot = sum([task.get_duration() for task in res[:i]])
    if tot > 1440:
        print(tot)
    if tot == 0:
        print("zero")
    return x
    
def for_the_vibe(tasks):
    time, index = 0, 0
    for i in range(len(tasks)):
        dur = tasks[i].get_duration()
        if time + dur > 1440:
            index += i
            break
        time += dur
    return index

def basic_greedy(tasks):
    """
    Args:
        tasks: list[Task], list of igloos to polish
    Returns:
        output: list[task] A basic solution obtained by greedy. Served as the starting point for SA.
        task_cp: list[task] tasks we haven't used
    """
    # first we have to sort the file
    tasks.sort(key = lambda x: x.get_max_benefit(), reverse = True)
    tasks_cp = copy.copy(tasks)
    timeslots = [0] * 1440
    for i in range(len(tasks)):
        if find_slot(tasks[i], timeslots):
            tasks_cp[i] = 0
    y = set(tasks_cp)
    y.discard(0)
    tasks_cp = list(y)

    # Post processing  
    sequence = set(timeslots)
    sequence.discard(0)
    x = list(sequence)
    last_index = len(x) - 1
    x.extend(tasks_cp)
    return x, last_index

def find_slot(task, timeslots):
    """
    Args:
        task: the task we are trying to put in the timeslot
        timeslots: the current order of tasks already assigned
    Returns:
        found: True if we can find a time_slot, False if we cannot  
    """
    deadline = task.get_deadline()
    duration = task.get_duration()

    curr_end = deadline
    curr_start = curr_end - duration
    found = False

    while curr_start >= 0 and not found:
        if all([isinstance(i,int) for i in timeslots[curr_start + 1 : curr_end]]):
            for i in range(curr_start, curr_end):
                timeslots[i] = task
            found = True
        else:
            # we shift and compare 
            curr_end -=  1
            curr_start -= 1
    return found

def simulated_annealing(s, last_task):
    """
    Args: 
        s: the initial lst for SA
        last_task: index of the last task completed before the deadline
    Returns:
        output: the final task array with optimal values
    """
    t = 200 # should be large. But need further testing
    while t: 
        s_prime = permute(s, last_task)
        old_cost, old_last_task = cost(s)
        new_cost, last_task = cost(s_prime)
        delta = new_cost - old_cost
        if delta > 0:
            s = s_prime
        else:
            # TODO
            # replace s = s_price with probability of e^(-delta/t)
            p = 1
            if t: 
                p = math.exp(delta/t)
            epsilon = random.random()
            if epsilon > p:
                s = s_prime
            else:
                last_task = old_last_task
        t -= 0.5
    return s, last_task

def freeze(task_lst):
    # TODO
    pass

def permute(task_lst, not_used):
    # TODO
    # given a task, we need to know where the num_b4_ddl_pass is 
    cp = copy.copy(task_lst)
    if len(task_lst) == 0:
        print("zero!")
    i = random.randint(0, len(task_lst)-1)
    j = random.randint(0, len(task_lst)-1)
    temp = cp[i]
    cp[i] = cp[j]
    cp[j] = temp
    return cp

def cost(task_lst):
    # TODO
    """
    Returns:
        total_cost: the total cost of this task list
        num_b4_ddl_pass: the index of last tasks can we actually completed before the deadline. 
    """

    time_spent, total_cost, num_b4_ddl_pass = 0, 0, 0
    for i, task in enumerate(task_lst):
        dur = task.get_duration()
        if time_spent + dur > 1440:
            num_b4_ddl_pass = i - 1
            break
        end_time = time_spent + dur
        ddl = task.get_deadline()
        total_cost += task.get_late_benefit(max(0, end_time - ddl))
    return total_cost, num_b4_ddl_pass

if __name__ == '__main__':
    for input_path in os.listdir('inputs/'):
        if input_path == '.DS_Store':
            continue
        for input_path2 in os.listdir('inputs/' + input_path):
             #print(input_path2)
             if input_path2 == '.ipynb_checkpoints' or input_path2 == '.DS_Store':
                 continue
             output_path = 'outputs/' + input_path + '/' + input_path2[:-3] + '.out'
             tasks = read_input_file('inputs/' + input_path + '/' + input_path2[:-3] + '.in')
             output = solve(tasks)
             write_output_file(output_path, output)
