import sys
import argparse
from argparse import RawTextHelpFormatter
import os
import re


def read_lines_to_list(fname):
    list = []
    with open(fname) as f:
        for line in f:
            list.append(line)
    return list

def write_list_to_file(list, fname):
    file = open(fname, "w")
    for line in list:
        file.write(line)

def sort_dimension(list):
    list = re.split("\[*\]", list)
    dims = []
    for item in list:
        dims.append(int(re.sub("[^0-9]", "", item)))
    return dims

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='test', formatter_class=RawTextHelpFormatter)

    parser.add_argument('--file_gpu', '-gpu', action="store", default="../beamforming/test/profiling/results/gpu_result.txt", dest="gpu")
    parser.add_argument('--file_cpu', '-cpu', action="store", default="../beamforming/test/profiling/results/cpu_result.txt", dest="cpu")

    gpu = parser.parse_args().gpu
    cpu = parser.parse_args().cpu

    gpu_lines = read_lines_to_list(gpu)
    cpu_lines = read_lines_to_list(cpu)

    print("GPU file has " + str(len(gpu_lines)) + " lines")
    print("CPU file has " + str(len(cpu_lines)) + " lines")

    gpu_lines.sort(key=sort_dimension)
    write_list_to_file(gpu_lines, "../beamforming/test/profiling/results//test.txt")
