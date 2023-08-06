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

from ._utils import volumetric
from .._ad import Variable

class StrainEnergy():
    """Strain energy class"""
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.iso_func = func
        self.vol_func = volumetric
        self.args = args
        self.kwargs = kwargs
    def jacobian(self, _grad):
        "Calculate the second Piola-Kirchoff stress tensor: S = 2 dW/dC"
        _grad = Variable(_grad, 1)
        _shape = _grad.shape[:-2]
        _bulk = 0
        if 'K' in self.kwargs.keys():
            _bulk = self.kwargs.pop('K')
        _dwdc = _grad.jacobian(self.iso_func, **self.kwargs).reshape((*_shape,3,3))
        if _bulk:
            kwargs = {'K':_bulk}
            _dwvdc = _grad.jacobian(self.vol_func, **kwargs).reshape((*_shape,3,3))
            _dwdc += _dwvdc
        return 2.0 * _dwdc
    def hessian(self, _grad):
        "Calculate the material tangent tensor: M = 4 d²W/dC²"
        _grad = Variable(_grad, 1)
        _shape = _grad.shape[:-2]
        _bulk = 0
        if 'K' in self.kwargs.keys():
            _bulk = self.kwargs.pop('K')
        _ddwdcdc = _grad.hessian(self.iso_func, **self.kwargs).reshape((*_shape,3,3,3,3))
        if _bulk:
            kwargs = {'K':_bulk}
            _ddwvdcdc = _grad.hessian(self.vol_func, **kwargs).reshape((*_shape,3,3,3,3))
            _ddwdcdc += _ddwvdcdc
        return 4.0 * _ddwdcdc
