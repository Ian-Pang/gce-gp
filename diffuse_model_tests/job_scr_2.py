import sys, os, time, fileinput
import numpy as np
import glob
from tqdm import tqdm

dif_mods = [['a', 'f'], ['o', 'f'], ['o', 'a']]
data_mods = ['o', 'a', 'f']
gpu_id = '1'

txt = lambda x: ('\"' + str(x) + '\"')

for i in tqdm(range(len(dif_mods)), desc = 'idx '):
    df = dif_mods[i]
    dm = data_mods[i]
    os.system('python gp_fit_scr.py ' + txt(df) + ' ' + dm + ' ' + gpu_id + '\n')