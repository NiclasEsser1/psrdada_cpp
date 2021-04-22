import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

import argparse
from argparse import RawTextHelpFormatter
sys.path.append('')
import benchmark as bench

class Plotter():
    def __init__(self, filename, sep=","):
        self.filename = filename
        self.table = pd.read_csv(self.filename, sep=sep, engine='python')
        self.elements = self.table["elements"].to_numpy()
        self.beams = self.table["beams"].to_numpy()
        self.execution_time = self.table["avg_time"].to_numpy() #* 1000
        self.beam_list = np.unique(self.beams)
        self.elem_list = np.unique(self.elements)

    def plot(self, col="avg_time", ylabel="Time [ms]", xlabel="Beams", save=True):
        i = 0
        y_values = self.table[col].to_numpy()
        for a in self.elem_list:
            plt.plot(self.beams[i:i+len(self.beam_list)], y_values[i:i+len(self.beam_list)], label=("elements 2 x " +str(a)))
            i+=len(self.beam_list)
        for line in self.beam_list:
            plt.axvline(line, color='grey', linestyle="-.", linewidth=0.5)
        if "bandwidth" in col:
            hline = [15.754, 31.508, 63.015]
            hlabel = ["PCIe3x16", "PCIe4x16", "PCIe5x16"]
            hstyle = ["-.", "-.", "-."]
            hcolor = ["black", "grey", "red"]
            for idx, line in enumerate(hline):
                plt.axhline(line, linestyle=hstyle[idx], color=hcolor[idx], linewidth=1, label=hlabel[idx])
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend()
        plt.title(self.table["devicename"][0] + " " + self.table["kernelname"][0])
        plt.savefig(self.filename[:-4] + col + ".png")
        plt.close()
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='test', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--dir', '-d', action="store", dest="dir", help = "Directory to .csv files")
    dir = parser.parse_args().dir
    pplotter = Plotter(dir + "power_kernel.csv")
    vplotter = Plotter(dir + "voltage_kernel.csv")
    uplotter = Plotter(dir + "unpacker_kernel.csv")
    pplotter.plot(col="avg_time", ylabel="Time [ms]", xlabel="beams")
    vplotter.plot(col="avg_time", ylabel="Time [ms]", xlabel="beams")
    uplotter.plot(col="avg_time", ylabel="Time [ms]", xlabel="beams")
    pplotter.plot(col="avg_throughput", ylabel="Throughput [GFLOPs]", xlabel="beams")
    vplotter.plot(col="avg_throughput", ylabel="Throughput [GFLOPs]", xlabel="beams")
    uplotter.plot(col="avg_throughput", ylabel="Throughput [GFLOPs]", xlabel="beams")
    pplotter.plot(col="avg_bandwidth", ylabel="Memory Bandwidth [GB/s]", xlabel="beams")
    vplotter.plot(col="avg_bandwidth", ylabel="Memory Bandwidth [GB/s]", xlabel="beams")
    uplotter.plot(col="avg_bandwidth", ylabel="Memory Bandwidth [GB/s]", xlabel="beams")
    pplotter.plot(col="input_avg_bandwidth", ylabel="Input Bandwidth [GB/s]", xlabel="beams")
    vplotter.plot(col="input_avg_bandwidth", ylabel="Input Bandwidth [GB/s]", xlabel="beams")
    uplotter.plot(col="input_avg_bandwidth", ylabel="Input Bandwidth [GB/s]", xlabel="beams")
    pplotter.plot(col="output_avg_bandwidth", ylabel="Output Bandwidth [GB/s]", xlabel="beams")
    vplotter.plot(col="output_avg_bandwidth", ylabel="Output Bandwidth [GB/s]", xlabel="beams")
    uplotter.plot(col="output_avg_bandwidth", ylabel="Output Bandwidth [GB/s]", xlabel="beams")
