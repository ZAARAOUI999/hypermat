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
from typing import Union, Iterable
from .._math import (trace, det, like, transpose, identity, dot, mul, inv, dya,
                     zeros)

class Variable():
    """Hyper-Dual variable class"""
    def __init__(self, f: Union[Iterable, int, float],
                 d: Union[Iterable, int, float] = 0.0,
                 dd: Union[Iterable, int, float] = 0.0):
        self.f = np.array(f)
        self.shape = f.shape
        self.n = self.shape[:-2]
        if isinstance(d, (int, float)):
            v = d
            d = np.zeros_like(f)
            d.fill(v)
        if isinstance(d, (int, float)):
            v = dd
            n = self.n
            m = len(self.f)
            dd = np.zeros((m,*n,*n))
            dd.fill(v)    
        self.d = d
        self.dd = dd
        
    def __repr__(self):
        return f"Variable \n f = {self.f}\n d = {self.d}"
    
    def __setitem__(self, indices, values):
        if isinstance(values, Variable):
            self.f[indices] = values.f
            self.d[indices] = values.d
        else:
            self.f[indices] = values
    
    def __getitem__(self, indices):
        return self.f[indices]
    
    def __neg__(self):
        return Variable(-self.f, -self.d, -self.dd)
    
    def __add__(self, other):
        if isinstance(other, (Iterable, int, float)):
            return Variable(self.f + other, self.d, self.dd)
        return Variable(self.f + other.f, self.d + other.d, self.dd + other.dd)
    __iadd__ = __radd__ = __add__
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return - self + other
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Variable(other * self.f, other * self.d, other * self.dd)
        return Variable(mul(self.f, other.f, 0),
                        mul(other.f, self.d, 0) + mul(other.d, self.f, 0),
                        mul(other.dd, self.f, 0) + dya(self.d, other.d,'ij,kl')\
                            + dya(other.d, self.d,'ij,kl') + mul(other.f, self.dd, 0))
    
    __rmul__ = __mul__
    
    def __truediv__(self, other): # self / other #TODO not fixet yet!
        return Variable(self.f / other.f,
                        (self.d * other.f - self.f * other.d) / other.f**2)

    def __rtruediv__(self, other): # other / self #TODO not fixet yet!
        return Variable(other.f / self.f,
                        (other.d * self.f - other.f * self.d) / self.f**2)
    
    def __matmul__(self,other):
        if isinstance(other, (int, float)):
            return Variable(other * self.f, other * self.d, other * self.dd)
        return Variable(dot(self.f, other.f),
                        dot(self.d, other.f) + dot(self.f, other.d),
                        mul(other.dd, self.f, 0) + dya(self.d, other.d,'ij,kl')\
                            + dya(other.d, self.d,'ij,kl') + mul(other.f, self.dd, 0))
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        Der1 = other * np.power(self.f, other-1.0)
        Der2 = other * (other-1.0) * (self.f**(other-2.0))
        return Variable(np.power(self.f, other),
                        mul(Der1, self.d, 0),
                        mul(self.dd, Der1, 0) + mul(dya(self.d,self.d,'ij,kl'), Der2, 0))
    
    @property
    def T(self):
        return transpose(self.f)
    
    def ravel(self):
        return self.f.ravel()
    
    def reshape(self, shape):
        return self.f.reshape(shape)
    
    def I1(self, C):
        Fun = trace(C) 
        Der = identity(C)
        Hes = zeros((*self.n,3,3,3,3))
        return Variable(Fun, Der, Hes)
    
    def I2(self, C):
        I = identity(C)
        TrC = trace(C)
        Fun = 0.5 * (TrC**2.0 - trace(dot(C,C)))
        Der = mul(TrC, I, 0) - transpose(C)
        dTrCIdC = np.einsum('...ij,...kl->...ijkl',I, I)
        dTrCdC = np.einsum('...ij,...kl->...iklj',I, I)
        Hes = dTrCIdC - dTrCdC
        return Variable(Fun, Der, Hes)
    
    def I3(self, C):
        invC = inv(C)
        Fun = det(C)
        Der = mul(Fun, transpose(invC), 0)
        dTrCinvdC = - np.einsum('...ij,...kl->...iklj',invC, invC)
        invCinvC = np.einsum('...ij,...kl->...ijkl',invC, invC)
        Hes = mul(Fun, invCinvC, 0) + mul(Fun, dTrCinvdC, 0)
        return Variable(Fun, Der, Hes)
    
    @property
    def invariants(self):
        F = self.f
        C = dot(transpose(F), F)
        I1 = self.I1(C)  
        I2 = self.I2(C)  
        I3 = self.I3(C)
        J1 = I3**(-1.0/3.0) * I1 
        J2 = I3**(-2.0/3.0) * I2 
        J3 = I3**(1.0/2.0)
        return C, J1, J2, J3
    
    def jacobian(self, func, **kwargs):
        _x = self
        _fx = func(_x, **kwargs)
        return  _fx.d.reshape((*self.n,3,3))
    
    def hessian(self, func, **kwargs):
        _x = self
        _fx = func(_x, **kwargs)
        return  _fx.dd.reshape((*self.n,3,3,3,3))
