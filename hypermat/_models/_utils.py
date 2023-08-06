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

def isochoric(func, grad, **kwargs):
    """Isochoric part of strain energy"""
    w_iso = func(grad, **kwargs)
    return w_iso

def volumetric(grad, **kwargs):
    """volumetric part of strain energy"""
    det = grad.invariants[-1]
    bulk = kwargs.pop('K')
    w_vol = 0.5 * bulk * (det - 1.0)**2.0
    return w_vol
