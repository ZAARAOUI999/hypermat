
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
import tensortrax as tr

from .._ad._deformation import Deformation
from ._utils import volumetric


class StrainEnergy():
    """Strain energy class"""
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.iso_func = func
        self.vol_func = volumetric
        self.args = args
        self.kwargs = kwargs
        self._bulk = 0
        if 'K' in self.kwargs:
            self._bulk = self.kwargs.pop('K')
    def jacobian(self, _x):
        "Calculate the first Piola-Kirchoff stress tensor: P = dW/dF"
        _x = Deformation(_x)
        _dwdf =  tr.Δ(self.iso_func(_x, **self.kwargs))
        if self._bulk:
            kwargs = {'K':self._bulk}
            _dwvdf =  tr.Δ(self.vol_func(_x, **kwargs))
            _dwdf += _dwvdf
        _dwdf[np.abs(_dwdf)<1.0e-13]=0.0
        return _dwdf[0,0]
    def hessian(self, _x):
        """Calculate the fourth-order elasticity tensor A = d²W/dF²"""
        _x = Deformation(_x)
        _ddwdfdf =  tr.Δδ(self.iso_func(_x, **self.kwargs))
        if self._bulk:
            kwargs = {'K':self._bulk}
            _ddwvdfdf =  tr.Δδ(self.vol_func(_x, **kwargs))
            _ddwdfdf += _ddwvdfdf
        _ddwdfdf[np.abs(_ddwdfdf)<1.0e-13]=0.0
        return _ddwdfdf

