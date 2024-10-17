import sys, os, time, fileinput
import numpy as np
import glob
from tqdm import tqdm

dif_mods = [['o', 'a', 'f']]
data_mods = ['o', 'a', 'f']
gpu_id = '3'

txt = lambda x: ('\"' + str(x) + '\"')

for dm in tqdm(data_mods):
    df = dif_mods[0]
    os.system('python gp_fit_scr_4.py ' + txt(df) + ' ' + dm + ' ' + gpu_id + '\n')