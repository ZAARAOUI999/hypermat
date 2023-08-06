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

####################### - بــسم الله الرحمــان الرحيــم - #####################

import numpy as np

__all__ = ('trace', 'det', 'like', 'transpose', 'identity', 'dot', 'mul', 'inv',
           'dya', 'zeros')

def trace(_a):
    """Provides tensor trace"""
    return np.einsum('...ii', _a)

def det(_a):
    """Provides tensor determinant"""
    return np.linalg.det(_a)

def inv(_a):
    """Provides tensor inverse"""
    return np.linalg.pinv(_a)

def like(_b, _a):
    """Broadcasts tensor to a given shape"""
    return np.broadcast_to(_b, _a.shape)

def transpose(_a):
    """Provides tensor transpose"""
    return np.einsum('...ij->...ji', _a, optimize=True)

def identity(_a):
    """Provides tensor identity"""
    return like(np.eye(3), _a)

def zeros(shape):
    """Null tensor"""
    return np.zeros(shape)

def dot(_a, _b):
    """Calculates tensor dot product"""
    return np.einsum('...ij,...jk->...ik', _a, _b, optimize=True)

def mul(_a, _b, mode=1):
    """Calculates tensor product"""
    if mode:
        _c = np.einsum('...ij,...ij->...ij', _a, _b, optimize=True)
    else:
        _c = np.einsum('ij...,ij...->ij...', _a, _b, optimize=True)
    return _c

def dya(_a, _b, seq):
    """Calculates tensor dyadic product"""
    i = seq.index(',')
    path = f'...{seq[:i]},...{seq[i+1:]}->...ijkl'
    _c = np.einsum(path, _a, _b, optimize=True)
    return _c
