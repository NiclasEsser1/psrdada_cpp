import os
import subprocess
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from argparse import RawTextHelpFormatter

class Params:
    def __init__(self, parser):
        self.tsamp = int(parser.parse_args().tsamp)
        self.channels = int(parser.parse_args().channels)
        self.antennas = [int(i) for i in parser.parse_args().antennas.split(",")]
        self.pol = int(parser.parse_args().pol)
        self.beams = [int(i) for i in parser.parse_args().beams.split(",")]
        self.interval = int(parser.parse_args().interval)
        self.type = int(parser.parse_args().type)
        self.id = int(parser.parse_args().device_id)
        self.iteration = int(parser.parse_args().iteration)
        self.outputfile = parser.parse_args().outputfile
        self.executable = parser.parse_args().executable
        self.elements = [int(i * self.pol) for i in self.antennas]
        self.samples = [int(i * self.tsamp * self.channels) for i in self.elements]
        if self.outputfile[-4:] != ".csv":
            self.outputfile += ".csv"
        self.cmd_args = {"samples" : self.tsamp,
            "channels" : self.channels,
            "antennas" : self.antennas,
            "pol" : self.pol,
            "beams" : self.beams,
            "interval" : self.interval,
            "type" : self.type,
            "id" : self.id,
            "iteration" : self.iteration,
            "outputfile" : self.outputfile}
    def _validate(self):
        if self.elements > 32 and (self.elements & (self.elements-1) == 0):
            raise ValueError("Number of antennas must be a multiple of 4")
        if self.beams%32 !=0:
            raise ValueError("Number of beams must be a multiple of 32")


class Plotter():
    def __init__(self, filename, sep=","):
        self.filename = filename
        self.table = pd.read_csv(self.filename, sep=sep, engine='python')
        self.antennas = self.table["#Nants"].to_numpy()
        self.beams = self.table["#Nbeams"].to_numpy()
        self.input_size = self.table["input_mb"].to_numpy() #/ 1024e2
        self.output_size = self.table["weights_mb"].to_numpy() #/ 1024e2
        self.weight_size = self.table["output_mb"].to_numpy() #/ 1024e2
        self.execution_time = self.table["Benchmark(s)"].to_numpy() #* 1000
        self.beam_list = np.unique(self.beams)

    def calculate(self):
        self.comp_eff = 7* self.input_size *self.beams / (8*self.execution_time)
        self.bw_raw = self.input_size / self.execution_time # MB/ms = GB/s
        self.bw_eff = (self.input_size + self.output_size + self.weight_size) / (self.execution_time)
    def sort_by(self, arg="#Nants"):
        idx = self.table.sort_values(arg).index
        self.table = self.table.loc[idx]
        print(self.table)
    def get_series(self, arg="#Nants"):
        return self.table[arg]
    def plot_time(self, save=True):
        a = min(self.antennas)
        i = 0
        while a <= max(self.antennas):
            plt.plot(self.beams[i:i+len(self.beam_list)], self.execution_time[i:i+len(self.beam_list)], label=("#elements " +str(a*2)))
            plt.ylabel("Execution time (ms)")
            plt.xlabel("#beams")
            plt.legend()
            plt.title("Timing (PowerBF)")
            i+=len(self.beam_list)
            a*=2
        if save:
            plt.savefig(self.filename[:-4] + "PowerBF_timeplot.png")
        plt.close()


    def plot_bw_raw(self, save=True):
        a = min(self.antennas)
        i = 0
        while a <= max(self.antennas):
            plt.plot(self.beams[i:i+len(self.beam_list)], self.bw_raw[i:i+len(self.beam_list)], label=("#elements " +str(a*2)))
            plt.ylabel("Bandwidth (GB/s)")
            plt.xlabel("#beams")
            plt.legend()
            plt.title("Bandwidth of raw voltage (PowerBF)")
            i+=len(self.beam_list)
            a*=2
        if save:
            plt.savefig(self.filename[:-4] + "PowerBF_bw_raw_plot.png")
        plt.close()

    def plot_bw_effective(self, save=True):# MB/ms = GB/s
        a = min(self.antennas)
        i = 0
        while a <= max(self.antennas):
            plt.plot(self.beams[i:i+len(self.beam_list)], self.bw_eff[i:i+len(self.beam_list)], label=("#elements " +str(a*2)))
            plt.ylabel("Bandwidth (GB/s)")
            plt.xlabel("#beams")
            plt.legend()
            plt.title("Effective bandwidth (PowerBF)")
            i+=len(self.beam_list)
            a*=2
        if save:
            plt.savefig(self.filename[:-4] + "PowerBF_bw_eff_plot.png")
        plt.close()


    def plot_comp_eff(self, save=True):
        a = min(self.antennas)
        i = 0
        while a <= max(self.antennas):
            plt.plot(self.beams[i:i+len(self.beam_list)], self.comp_eff[i:i+len(self.beam_list)], label=("#elements " +str(a*2)))
            plt.ylabel("Throughput (GFLOPs)")
            plt.xlabel("#beams")
            plt.legend()
            plt.title("Computational Throughput (PowerBF)")
            i+=len(self.beam_list)
            a*=2
        if save:
            plt.savefig(self.filename[:-4] + "PowerBF_comp_eff_plot.png")
        plt.close()

    def plot_relative_comp_eff(self, ops, save=True):
        a = min(self.antennas)
        i = 0
        while a <= max(self.antennas):
            plt.plot(self.beams[i:i+len(self.beam_list)], self.comp_eff[i:i+len(self.beam_list)]/ops*100, label=("#elements " +str(a*2)))
            plt.ylabel("[%]")
            plt.xlabel("#beams")
            plt.legend()
            plt.title("Percental Throughput of Peak Performance (PowerBF)")
            i+=len(self.beam_list)
            a*=2
        if save:
            plt.savefig(self.filename[:-4] + "PowerBF_relative_comp_eff_plot.png")
        plt.close()


def run(exe, args):
    for i in range(len(args["antennas"])):
        for k in range(len(args["beams"])):
            # Create execute command
            args_string = exe + " "
            for key, value in args.items():
                if key == "antennas":
                    args_string += " --" + key + " " + str(value[i])
                elif key == "beams":
                    args_string += " --" + key + " " + str(value[k])
                else:
                    args_string += " --" + key + " " + str(value)
            subprocess.call(args_string, shell=True)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='test', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--samples', '-s', action="store", default="4096", dest="tsamp", help = "Number of timestamps in databuffer")
    parser.add_argument('--channels', '-c', action="store", default="32", dest="channels", help = "Number of channels")
    parser.add_argument('--antennas', '-a', action="store", default="32,64,128,256,512", dest="antennas", help = "Number of antennas\n    If benchmark with several antennas is desired, seperate them by ',' (e.g 64,128,256,512). Must be a power of 2")
    parser.add_argument('--pol', '-p', action="store", default="2", dest="pol", help = "Number of polarizations")
    parser.add_argument('--beams', '-b', action="store", default="32,64,128,256,512,1024", dest="beams", help = "Number of beams\n   If benchmark with several antennas is desired, seperate them by ',' (e.g 32,64,92). Must be a multiple of 32")
    parser.add_argument('--interval', '-int', action="store", default="64", dest="interval", help = "Integration interval")
    parser.add_argument('--type', '-t', action="store", default="1", dest="type", help = "Beamform type: 0 (naive BF) or 1 (optimzed BF)")
    parser.add_argument('--device_id', '-id', action="store", default="0", dest="device_id", help = "GPU device on which the kernels gets executed")
    parser.add_argument('--iteration', '-i', action="store", default="5", dest="iteration", help = "Number of iterations per cycle")
    parser.add_argument('--outputfile', '-o', action="store", default="/psrdada_cpp/psrdada_cpp/cryopaf/tools/benchmark/benchmark", dest="outputfile", help = "Path to outputfile. If file exists new results are appended to the bottom of the file.")
    parser.add_argument('--executable', '-e', action="store", default="/psrdada_cpp/build/psrdada_cpp/cryopaf/beamforming/test/profiling/profiling", dest="executable", help = "Path to executable. If file exists new results are appended to the bottom of the file.")

    params = Params(parser)
    # run(params.executable, params.cmd_args)
    plotter = Plotter(params.outputfile)
    plotter.calculate()
    plotter.plot_time()
    plotter.plot_bw_raw()
    plotter.plot_bw_effective()
    plotter.plot_comp_eff()
    plotter.plot_relative_comp_eff(13450)
