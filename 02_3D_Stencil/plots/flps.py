#!python
import os
from typing import List
import numpy as np

import sys
from pathlib import Path
from itertools import islice, cycle
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib as mpl

def get_measurement_files(rpath:str) -> List[str]:
    m_files = []
    for root, dirs, files in os.walk(rpath, topdown=True):
        for name in files:
            m_files.append(os.path.join(root, name))
        break # scan only top lvl directory
    return m_files


def better_mpl_line_style_cycle(num_styles = 10):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = list(islice(cycle(prop_cycle.by_key()['color']), num_styles))
    linestyles = list(islice(cycle(['-', '--', ':', '-.']), num_styles))
    plt.rc('axes', prop_cycle=(cycler('color', colors) +
                               cycler('linestyle', linestyles)))

def main(argv):
    # Some defaults I like
    mpl.style.use(Path(__file__).resolve().parent / 'mystyle.mplstyle')
    # make overlapping lines easier to see
    better_mpl_line_style_cycle()


    raw_files = get_measurement_files(argv)

    files_cube = [f for f in raw_files if "2-FDTD" in f] # files must contain "3_3"

    files_cube.sort()
    files_ax = [file for file in raw_files if file not in files_cube]

    points_ax = np.zeros(len(files_ax), np.float64)
    unit = "MPoints/s"
    radius = list(range(1,11))

    for i, file in enumerate(files_ax):
        with open(file, "rt") as f:
            identifier = "FDTD3d, Throughput = "
            for line in f:  # iterate over lines
                # Line Pre Processing-------------------
                if line.isspace():  # continue if empty line
                    continue
                if line.startswith(identifier):
                    raw_data = line[len(identifier):-1]
                    data_str = raw_data.split(" ")[0]
                    points_ax[i] = float(data_str)
                    break
    
    points_cube = np.zeros(len(files_cube), np.float64)
    for i, file in enumerate(files_cube):
        with open(file, "rt") as f:
            identifier = "FDTD3d, Throughput = "
            for line in f:  # iterate over lines
                # Line Pre Processing-------------------
                if line.isspace():  # continue if empty line
                    continue
                if line.startswith(identifier):
                    raw_data = line[len(identifier):-1]
                    data_str = raw_data.split(" ")[0]
                    points_cube[i] = float(data_str)
                    break

    points_ax.sort()
    points_cube.sort()
    radius.reverse()
    radius: np.ndarray = np.array(radius)

    plt.plot(radius, points_ax, label="axial stencil", marker = 'o')
    plt.plot(radius, points_cube, label="cubic stencil", marker = 'o')
    plt.yscale('log')
    plt.xlabel("radius")
    plt.ylabel(unit)
    plt.title("FDTD3d")
    plt.legend()

    name = argv.split(sep="/")[-1]
    name = name.split(sep=".")[0]
    plt.savefig("./" + "Points")
    plt.clf()

    flops_ax = np.zeros(len(points_ax))
    elements_in_ax_stenci = 6*radius+1
    mul_flops = elements_in_ax_stenci
    add_flops = elements_in_ax_stenci-1
    flops_p_outelem_ax = mul_flops + add_flops
    flops_ax = np.multiply(points_ax, flops_p_outelem_ax)
    Gflops_ax = flops_ax/1E3

    flops_cube = np.zeros(len(points_cube))
    elements_in_cube_stenci = np.power((2*radius+1),3)
    mul_flops = elements_in_cube_stenci
    add_flops = elements_in_cube_stenci-1
    flops_p_outelem_cube = mul_flops + add_flops
    flops_cube = np.multiply(points_cube, flops_p_outelem_cube)
    Gflops_cube = flops_cube/1E3

    plt.plot(radius, Gflops_ax, label="axial stencil",  marker = 'o')
    plt.plot(radius, Gflops_cube, label="cubic stencil",  marker = 'o')
    plt.xlabel("radius")
    plt.ylabel("Gflops/s")
    plt.title("FDTD3d")
    plt.legend()

    plt.savefig("./" + "Gflop")
    plt.clf()





if __name__ == "__main__":
    args = "../results"
    main(args)
