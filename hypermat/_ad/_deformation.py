
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
from typing import Union, Iterable

import tensortrax as tr
import tensortrax.math as tm

class Deformation():
    """Deformation Gradient class"""
    def __init__(self, f: Union[Iterable, int, float], **kwargs):
        """Initialise deformation gradient values

        Args:
            f (Union[Iterable, int, float]): Initial values of deformation gradient.
        """
        self._f = tr.Tensor(f, ntrax=2, **kwargs)
        self._f.init(gradient=True, hessian=True, δx=True, Δx=True)

    @property
    def invariants(self):
        """Calculate Cauchy Green Strain invariants."""
        _f = self._f
        _c = _f.T @ _f
        _i1 = tm.trace(_c)
        _i2 = 0.5 * (_i1**2.0 - tm.trace(_c@_c))
        _i3 = tm.linalg.det(_c)
        _j1 = _i3**(-1.0/3.0) * _i1
        _j2 = _i3**(-2.0/3.0) * _i2
        _j3 = _i3**(1.0/2.0)
        return (_c, _j1, _j2, _j3)
    @property
    def stretches(self):
        """Calculate principal stretches."""
        _f = self._f
        _c = _f.T @ _f
        _lmbda_i = tm.linalg.det(_c)**(-1.0/6.0) * tm.linalg.eigvalsh(_c)**0.5
        return _lmbda_i
