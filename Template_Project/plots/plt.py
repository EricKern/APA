#!python

import sys
from pathlib import Path
from itertools import islice, cycle
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib as mpl

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

    # read data, skip context information
    # df = pd.read_json(argv)
    df = pd.read_csv(argv)

    
    df_d2d = df[df['Benchmark'] == 'copyd2d']
    print(df_d2d.columns)

    df_d2d_bw = df_d2d.rename({'GlobalMem BW (bytes/sec)': 'BW (GB/s)'}, axis='columns')
    df_d2d_bw = df_d2d_bw["BW (GB/s)"]/1E9

    # df_d2d.loc[:, "GlobalMem BW (bytes/sec)"].plot(legend=True)
    df_d2d_bw.plot(legend=True)
    plt.xlabel("x-lable here")
    plt.ylabel("y-lable here")
    plt.title("Title here")

    name = argv.split(sep="/")[-1]
    name = name.split(sep=".")[0]
    plt.savefig("./" + name)


if __name__ == "__main__":
    args = "../results/bandwidth-20221120_182438.csv"
    main(args)
