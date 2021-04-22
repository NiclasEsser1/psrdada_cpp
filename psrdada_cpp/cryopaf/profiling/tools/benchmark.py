import os
import subprocess
import argparse
import numpy as np
from argparse import RawTextHelpFormatter


PROFILING_EXEC = "/homes/nesser/SoftwareDev/Projects/psrdada_cpp/build/psrdada_cpp/cryopaf/profiling/profiling"
class Params:
    def __init__(self, parser):
        self.tsamp = int(parser.parse_args().tsamp)
        self.channels = int(parser.parse_args().channels)
        self.elements = [16,32,64,128,256,512,1024]
        self.beams = np.arange(32,1024+32, step=32).tolist()
        self.integration = int(parser.parse_args().integration)
        self.device_id = int(parser.parse_args().device_id)
        self.iteration = int(parser.parse_args().iteration)
        self.protocol = parser.parse_args().protocol
        self.precision = parser.parse_args().precision
        self.outdir = parser.parse_args().outdir
        self.cmd_args = {"samples" : self.tsamp,
            "channels" : self.channels,
            "elements" : self.elements,
            "beams" : self.beams,
            "integration" : self.integration,
            "device" : self.device_id,
            "iteration" : self.iteration,
            "protocol" : self.protocol,
            "outdir" : self.outdir,
            "precision" : self.precision}


def run(exe, args):
    for i in range(len(args["elements"])):
        for k in range(len(args["beams"])):
            # Create execute command
            args_string = exe + " "
            for key, value in args.items():
                if key == "elements":
                    args_string += " --" + key + " " + str(value[i])
                elif key == "beams":
                    args_string += " --" + key + " " + str(value[k])
                else:
                    args_string += " --" + key + " " + str(value)
            print(args_string)
            subprocess.call(args_string, shell=True)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='test', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--samples', '-s', action="store", default="4096", dest="tsamp", help = "Number of timestamps in databuffer")
    parser.add_argument('--channels', '-c', action="store", default="7", dest="channels", help = "Number of channels")
    parser.add_argument('--integration', '-int', action="store", default="1", dest="integration", help = "Integration integration")
    parser.add_argument('--device_id', '-id', action="store", default="0", dest="device_id", help = "GPU device on which the kernels gets executed")
    parser.add_argument('--iteration', '-i', action="store", default="20", dest="iteration", help = "Number of iterations per cycle")
    parser.add_argument('--protocol', '-proto', action="store", default="codif", dest="protocol", help = "")
    parser.add_argument('--precision', '-p', action="store", default="half", dest="precision", help = "")
    parser.add_argument('--outdir', '-o', action="store", default="Results", dest="outdir", help = "")

    params = Params(parser)
    run(PROFILING_EXEC, params.cmd_args)
