#!python
from pathlib import Path
from itertools import islice, cycle
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib as mpl
#===============


from dataclasses import dataclass
from typing import List, Callable, TypeVar, Generic, Union
import enum
import os


T = TypeVar('T')

@dataclass
class Member(Generic[T]):

    val_parser: Callable[[str], T]
    """Function transforming data string to the desired type
    Could be for example just int("13") if T is int""" 
    
    value: T
    """Actual value of that member""" 

    id: str
    """Idenitifier string to look for in the file to find the member value""" 
    
# Adapt below -----------------------------------------------------------------

def remove_ms(string: str) -> float:
    """Parser for time measurements removes trailing ms
    from string"""
    return float(string[:-3])

def remove_GBs(string: str) -> float:
    """Parser for time measurements removes trailing GB/s
    from string"""
    return float(string[:-5])

class Meta_Measure:
    """Class holding meta information for measurements"""
    def __init__(self) -> None:
        self.avg_t:       Member[float] = Member(remove_ms, 0.0, "Average time: ")
        self.Bandwidth:   Member[float] = Member(remove_GBs, 0.0, "Bandwidth:    ")
        self.elem:   Member[float] = Member(int, 0, "NOT_PRESENT")
        self.thread:   Member[float] = Member(int, 0, "NOT_PRESENT")



class Data_Measure:
    """Class holding meta information for measurements"""
    def __init__(self) -> None:
        self.exec_time: Member[float] = Member(float, 0.0,
                                                     "Execution Time: ")

# Adapt above -----------------------------------------------------------------

class Measure_Factory:

    def __init__(self, file_path: str) -> None:
        self.file_path: str = file_path
        self.experiment_meta: Meta_Measure = Meta_Measure()
        self.experiment_data: List[Data_Measure] = [Data_Measure()]
        self.meta_progress: int = 0
        self.data_progress: int = 0
        self.n_meta = len(vars(Meta_Measure())) # gives # of fields
        self.n_data = len(vars(Data_Measure()))

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, f_path: str):
        if os.path.isfile(f_path):
            self._file_path = f_path  # makes private variable outside __init__
        else:
            raise ValueError("Passed path seems to be no valid path")

    def read(self):
        ReadStates = enum.Enum('ReadStates', ['readingMeta', 'readingData'])

        file_name = self.file_path.split("/")[-1]
        if file_name.startswith("M-"):
            file_name = file_name[2:]
        name = file_name.split(".")[0]
        self.experiment_meta.elem.value = int(name.split("-")[0])
        self.experiment_meta.thread.value = int(name.split("-")[1])

        with open(self.file_path, "rt") as f:
            curr_state = ReadStates.readingMeta
            for line in f:  # iterate over lines
                # Line Pre Processing-------------------
                if line.isspace():  # continue if empty line
                    continue
                if line.startswith("End-Meta-Info"):
                    curr_state=ReadStates.readingData
                # ----------------------------------------

                if curr_state == ReadStates.readingMeta:
                    self.make_Meta_Measure(line)
                # elif curr_state == ReadStates.readingData:
                #     self.make_Data_Measure(line)
                else:
                    raise NotImplementedError

    def make_Meta_Measure(self, line: str):
        success = self.fill_data_cls(line, self.experiment_meta)
        if success:
            self.meta_progress += 1

    # def make_Data_Measure(self, line: str):
    #     if self.data_progress == self.n_data:
    #         self.experiment_data.append(Data_Measure())
    #         self.data_progress = 0

    #     success = self.fill_data_cls(line, self.experiment_data[-1])
    #     if success:
    #         self.data_progress += 1

    def fill_data_cls(self, line: str, data_cls: Union[Meta_Measure, Data_Measure]):
        # check for every member of the data class if line starts
        # with corresponding identifier
        # Maybe not best performance but least constraints on file format

        for attr_name, member_obj in vars(data_cls).items():
            id = member_obj.id
            # get substring to search for

            if line.startswith(id):
                # if found cast remaining string to type and assign
                member_obj.value = member_obj.val_parser(line[len(id):-1])
                return True
            else:
                continue
        return False # no matching attribute found


    def get_meta(self) -> Meta_Measure:
        return self.experiment_meta

    def get_data(self) -> List[Data_Measure]:
        return self.experiment_data

#-----------------------------------------------------------------------------

def get_measurement_files() -> List[str]:
    m_files = []
    for root, dirs, files in os.walk("../results", topdown=True):
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



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    # plt.rcParams.update({'font.size': 11})
    from collections import namedtuple

    # Some defaults I like
    mpl.style.use(Path(__file__).resolve().parent / 'mystyle.mplstyle')
    # make overlapping lines easier to see
    better_mpl_line_style_cycle()

    raw_files = get_measurement_files()

    files_mpass = [f for f in raw_files if 'M-' in f] # files must contain "M-"
    files = [f for f in raw_files if not 'M-' in f] # files must contain "M-"

    factories_mpass: List[Measure_Factory] = [None]*len(files)
    factories: List[Measure_Factory] = [None]*len(files)

    for i, file in enumerate(files_mpass) :
        factories_mpass[i] = Measure_Factory(file)
        factories_mpass[i].read()

    for i, file in enumerate(files) :
        factories[i] = Measure_Factory(file)
        factories[i].read()

    meta_mpass: List[Meta_Measure] = []
    meta: List[Meta_Measure] = []

    for i in range(len(factories_mpass)):
        meta_mpass.append(factories_mpass[i].get_meta())
    for i in range(len(factories)):
        meta.append(factories[i].get_meta())


    key_resolver = lambda meta_measure: (meta_measure.elem.value
                                        + meta_measure.thread.value/1000)
    meta_mpass.sort(key=key_resolver)
    meta.sort(key=key_resolver)

    # https://stackoverflow.com/a/17496530/4960953
    # make list of dicts to create pd.Dataframe at the end
    pd_raw_rows = []

    for mpass_elem, elem in zip(meta_mpass, meta):
        pd_raw_row = dict()
        if mpass_elem.elem.value == elem.elem.value:
            pd_raw_row["elements"] = elem.elem.value
        else:
            raise RuntimeError("Not sorted!")

        if mpass_elem.thread.value == elem.thread.value:
            pd_raw_row["threads"] = elem.thread.value
        else:
            raise RuntimeError("Not sorted!")

        pd_raw_row["BW"] = elem.Bandwidth.value
        pd_raw_row["BW-Multipass"] = mpass_elem.Bandwidth.value

        pd_raw_rows.append(pd_raw_row)

    df = pd.DataFrame(pd_raw_rows)
    # print(df.head())

    df_new = df.pivot(index="elements", columns="threads")
    # print(df_new.head())

    # df_new["BW-Multipass"].plot()
    df_new.plot()
    plt.ylabel("Bandwidth in GB/s")
    plt.xlabel("Elements")
    plt.xscale('log', base=2)

    # plt.title("")
    plt.legend()
    plt.savefig("5_2.pdf", bbox_inches='tight')