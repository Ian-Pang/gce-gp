import sys, os, time, fileinput
import numpy as np
import glob
from tqdm import tqdm

dif_mods = [['o'], ['a'], ['f']]
data_mods = ['o', 'a', 'f']
gpu_id = '2'

txt = lambda x: ('\"' + str(x) + '\"')

for df in tqdm(dif_mods, desc = 'dif_mod', position = 0):
    for dm in tqdm(data_mods, desc = 'dat_mod', position = 1, leave = False):
        os.system('python gp_fit_scr_1.py ' + txt(df) + ' ' + dm + ' ' + gpu_id + '\n')