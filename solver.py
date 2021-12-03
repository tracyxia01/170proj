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
    timeslots = [0] * 1440 # so it ranges from 0 to 1439
    for task in tasks:
        id = task.get_task_id()
        deadline = task.get_deadline()
        duration = task.get_duration()
        find_time(deadline, timeslots, id, duration)
    sequence = set(timeslots)
    sequence.discard(0)
    #print(list(sequence))
    return list(sequence)

# What does this function do? 
def find_slot(end_time, timeslots, id, duration):
    # make a copy of the current times slot
    # OK This function isnt that accurate Imma write a new one
    time_copy = timeslots.copy()
    if end_time-duration < 0:
        return False, timeslots, 0
    for i in range(end_time-1, end_time-duration-1, -1):
        if timeslots[i] == 0:
            time_copy[i] = id
        else:
            return False, timeslots, i+1
    return True, time_copy, 0

# passes in the current task information
# fill in the timeslot if we can find one
# if not, don't do anything
# Return nothing. So we distructively modify the timeslot. 
def find_time(end_time, timeslots, id, duration):
    # set up the start and end time
    curr_end = end_time
    curr_start = curr_end - duration + 1
    found = False
    # we want to look at every reasonable spot 
    # We also have to take into consideration that 
    # the next task starts at the same task the prev was finished
    while curr_start >= 0 and not found:
        # so ignore the start and end, as long as the middle ones are all zero
        if sum(timeslots[curr_start + 1 : curr_end]) == 0:
            for i in range(curr_start, curr_end + 1):
                # We have to check if the entire timeslot if 0
                timeslots[i] = id
                found = True
        else:
            curr_end -=  1
            curr_start -= 1

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
