from parse import read_input_file, write_output_file
import os

def solve(tasks):
    """
    Args:
        tasks: list[Task], list of igloos to polish
    Returns:
        output: list of igloos in order of polishing
    """
    base = basic_greedy(tasks) # We first find the bast 
    res = simulated_annealing(base) # apply simulated annealing to the base array
    return res
    


def basic_greedy(tasks):
    """
    Args:
        tasks: list[Task], list of igloos to polish
    Returns:
        output: a basic solution obtained by greedy. Served as the starting point for SA. 
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
    #print(list(sequence))
    return list(sequence)

def find_slot(end_time, timeslots, id, duration):
    """
    Args:
        end_time: the end time of the current task
        timeslots: the current order of tasks already assigned
        id: the id of current task
        duration: duration of the current task
    Returns:
        discrutively modify timeslots to put our current job in it. 
        Doesn't return anything.  
    """
    curr_end = end_time
    curr_start = curr_end - duration
    found = False

    while curr_start >= 0 and not found:
        if sum(timeslots[curr_start : curr_end]) == 0:
            for i in range(curr_start, curr_end):
                timeslots[i] = id
            found = True
        else:
            # we shift and compare 
            curr_end -=  1
            curr_start -= 1


def simulated_annealing(s):
    """
    Args: 
        task_lst: the initial lst for SA
    Returns:
        output: the final task array with optimal values
    """
    t = 20 # should be large. But need further testing
    while not freeze(s): 
        s_prime = permute(s)
        delta = cost(s_prime) - cost(s)
        if delta < 0:
            s = s_prime
        else:
            # TODO
            # replace s = s_price with probability of e^(-delta/t)
            pass
    return 0

def freeze(task_lst):
    # TODO
    pass

def permute(task_lst):
    # TODO
    pass

def cost(task_lst):
    # TODO
    pass

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
