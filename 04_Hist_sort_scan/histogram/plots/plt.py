#!python
from itertools import islice, cycle
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib as mpl
#===============

import os



def get_measurement_files() -> List[str]:
    m_files = []
    for root, dirs, files in os.walk("../results", topdown=True):
        for name in files:
            m_files.append(os.path.join(root, name))
        break # scan only top lvl directory
    return m_files

def make_o_path(o_path, o_suffix, i=0) -> str:
    make_path = lambda path, sfx, i : path + "-" + str(i).zfill(2) + sfx
    out_path = make_path(o_path, o_suffix, i)
    while os.path.exists(out_path):
        i += 1
        out_path = make_path(o_path, o_suffix, i)
    return out_path

def better_mpl_line_style_cycle(num_styles = 8):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = list(islice(cycle(prop_cycle.by_key()['color']), num_styles))
    marker = list(islice(cycle(['*', '+', 'v', '^', 'X', 'o', 'v', '^']), num_styles))
    linestyles = list(islice(cycle(['-', '--', ':', '-.','-', '--', ':', '-.']), num_styles))
    plt.rc('axes', prop_cycle=(cycler('color', colors) +
                               cycler('linestyle', linestyles) +
                               cycler('marker', marker)))



if __name__ == "__main__":
    from pathlib import Path
    import collections
    import numpy as np
    from typing import List
    # plt.rcParams.update({'font.size': 11})

    mpl.style.use(Path(__file__).resolve().parent / 'mystyle.mplstyle')
    # make overlapping lines easier to see
    better_mpl_line_style_cycle()

    raw_files = get_measurement_files()

    Point = collections.namedtuple('Point', ['Wc', 'binNum', 'MB_s'])
    data_points: List[Point] = []

    identifier = "histogramBinNum, Throughput = "
    for file in raw_files:
        base_name = os.path.basename(file)
        file_name = os.path.splitext(base_name)[0]
        Wc = file_name[2:4]
        binNum = file_name[11:15]

        with open(file, "rt") as f:
            for line in f:  # iterate over lines
                if line.startswith(identifier):
                    MB_s = line[len(identifier):].split(" ")[0]
                    point = Point(int(Wc), int(binNum), float(MB_s))
                    data_points.append(point)
                    break # next file
                else:
                    continue # next line

    lex_sorter = lambda point : (point.Wc, point.binNum)
    data_points.sort(key=lex_sorter)

    plot_frame = collections.defaultdict(list)
    for Wc, *rest_tpl in data_points:
        plot_frame[Wc].append(rest_tpl)  # add to existing list or create a new one
    dbg=1

    fig, ax = plt.subplots()
    for Wc, data_list in plot_frame.items():
        binNums, MB_s = zip(*data_list)
        GB_s = np.array(MB_s)/1E3
        ax.plot(binNums, GB_s, label="Wc="+str(Wc))

    plt.xscale("log", base=2)
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    # ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    plt.ylabel("Bandwidth in GB/s")
    plt.xlabel("# Bins")
    plt.tight_layout()
    plt.legend()
    
    out_path = make_o_path("./histogram", ".pdf")
    plt.savefig(out_path)
