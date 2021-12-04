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
    return [task.id for task in res]
    


def basic_greedy(tasks):
    """
    Args:
        tasks: list[Task], list of igloos to polish
    Returns:
        output: list[task] 
        A basic solution obtained by greedy. Served as the starting point for SA. 
    """
    # first we have to sort the file
    tasks.sort(key = lambda x: x.get_max_benefit())
    timeslots = [0] * 1440
    for task in tasks:
        find_slot(task, timeslots)
    # Question: can we keep a list of jobs that are not used?  
    sequence = set(timeslots)
    sequence.discard(0) # what does this do?  
    return list(sequence)

def find_slot(task, timeslots):
    """
    Args:
        task: the task we are trying to put in the timeslot
        timeslots: the current order of tasks already assigned
    Returns:
        discrutively modify timeslots to put our current job in it. 
        Doesn't return anything.  
    """
    deadline = task.get_deadline()
    duration = task.get_duration()

    curr_end = deadline
    curr_start = curr_end - duration
    found = False

    while curr_start >= 0 and not found:
        if sum(timeslots[curr_start + 1 : curr_end]) == 0:
            for i in range(curr_start, curr_end):
                timeslots[i] = task
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
    """
    Returns:
        total_cost: the total cost of this task list
        num_b4_ddl_pass: the number of tasks we actually completed. 
        I will explain this later lol ik it's confusing. 
    """
    time_spent, total_cost = 0, 0
    num_b4_ddl_pass = 0
    for i, task in enumerate(task_lst):
        if time_spent > 1440:
            num_b4_ddl_pass = i
            break
        end_time = time_spent + task.get_duration()
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
