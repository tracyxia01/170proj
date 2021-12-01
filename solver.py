from parse import read_input_file, write_output_file
import os

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
        found_slot = False
        while not found_slot:
            if deadline <= 0:
                break
            found_slot, time_copy = find_slot(deadline, timeslots, id, duration)
            if found_slot:
                timeslots = time_copy
                break
            deadline -= 1
    sequence = set(timeslots)
    sequence.discard(0)
    #print(list(sequence))
    return list(sequence)

def find_slot(end_time, timeslots, id, duration):
    time_copy = timeslots
    for i in range(end_time-1, max(-1, end_time-duration-1), -1):
        if timeslots[i] == 0:
            time_copy[i] = id
        else:
            return False, timeslots
    return True, time_copy


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
