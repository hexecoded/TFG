import numpy as np
import torch
import os

print(torch.cuda.is_available())

for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i).name)

import multiprocessing

#print(multiprocessing.cpu_count())
print(len(os.sched_getaffinity(0)))
