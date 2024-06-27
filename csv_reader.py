import numpy as np

def read_csv(file_name):
    file = open(file_name,'r')
    
    data = []
    
    for cur_line in file:
        cur_line = cur_line.strip()
        cur_row = cur_line.split(',')
        cur_row = np.array([int(num) for num in cur_row])
        data.append(cur_row)

    file.close()

    return np.array(data)
