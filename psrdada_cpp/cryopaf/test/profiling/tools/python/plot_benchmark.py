import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
sys.path.append('')
import benchmark as bench


plotter = bench.Plotter("benchmark/benchmark_beanfarmer.csv")
plotter.sort_by(["#Nants","#Nbeams"])
plotter.read_out()
plotter.comp_eff = plotter.get_series("#Performance(Tops/s)")*1000
plotter.plot_time()
plotter.plot_bw_raw()
plotter.plot_bw_effective()
plotter.plot_comp_eff()
