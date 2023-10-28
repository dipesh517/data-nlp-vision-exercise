import os 

work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_dir = os.path.join(work_dir, 'data')
output_dir = os.path.join(work_dir, 'output')
logs_dir = os.path.join(work_dir, 'logs')