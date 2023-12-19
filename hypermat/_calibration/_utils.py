
####################### - بــسم الله الرحمــان الرحيــم - #####################

"""
    HyperMAT
    Created August 2023
    Copyright (C) Mohamed ZAARAOUI

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 2
    of the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def init_plot():
    """Initialize plot"""
    matplotlib.rcParams['font.family'] = ['Times New Roman']
    plt.minorticks_on()
    plt.gca().grid(which='major', color='#808080')
    plt.gca().grid(which='minor', color='#C0C0C0')

def read_file(file_path, **kwds):
    """Read csv, txt files"""
    out = np.genfromtxt(file_path, **kwds)
    ncols = out.shape[1]
    return out[~np.isnan(out)].reshape(-1,ncols)

def to_dict(data, keys):
    """Create a dictionary with the given data and keys."""
    out = dict()
    for i, key in enumerate(keys):
        out[key] = data[...,i]
    return out
