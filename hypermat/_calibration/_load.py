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


import lmfit as lm
import numpy as np
from scipy.optimize import minimize, fsolve, zeros

import matplotlib.pyplot as plt
import matplotlib

from ._utils import to_dict
from .._math import *

matplotlib.rcParams['font.family'] = ['Times New Roman']
plt.minorticks_on()
plt.gca().grid(which='major', color='#808080')
plt.gca().grid(which='minor', color='#C0C0C0')

SN = {'E':r'Engineering Strain $\epsilon_e$', 'T':'True Strain $\epsilon_t$'}
SS = {'E':'Engineering Stress $\sigma_e$', 'T':'True Stress $\sigma_t$'}

class Test():
    def __init__(self, umat, data, ss_type: str = 'E'):
        """ss_type: E for Engineering and T for True"""
        self._umat = umat
        self.data = data
        self._grad = None
        self._ss_type = ss_type
        self._label = 'Experimental data'

    def stress(self):
        """Computes stress tensor"""
        if self._ss_type=='E':
            _stress = self._grad @ self._umat.jacobian(self._grad)
        elif self._ss_type=='T':
            _stress = np.einsum('...,...ij->...ij', det(self._grad),
                                self._grad @ self._umat.jacobian(self._grad) @ self._grad)
        return _stress

    def fit_data(self, min_values: list[int|float] = [],
                 max_values: list[int|float] = [],
                 variables: list[bool] = []):
        """Fits model parameters with the experimental data"""
        params_dict = self._umat.kwargs.copy()
        params_dict['K'] = self._umat._bulk
        params = lm.Parameters()
        for i, item in enumerate(params_dict.items()):
            params.add(item[0], vary=variables[i], value=item[1],
                       min=min_values[i], max=max_values[i])
        minner = lm.Minimizer(self._objective, params)
        result = minner.minimize(method='least_squares') #'least_squares', 'nelder', 'leastsq'

    def plot_model(self, **kwargs):
        """Plots model strain-stress curve"""
        _stress = self.stress()[...,0,0].ravel()
        _strain = self.data['strain']
        plt.gca().plot(_strain, _stress, label=self._label, **kwargs)
        plt.gca().legend()

    def plot(self, **kwargs):
        """Plots experimental stress-strain curve"""
        _stress = self.data['stress']
        _strain = self.data['strain']
        plt.gca().plot(_strain, _stress, label='Experimental data', **kwargs)
        plt.gca().set_xlabel(SN[self._ss_type])
        plt.gca().set_ylabel(SS[self._ss_type])
        plt.gca().legend()

class Uniaxial(Test):
    """Compressible uniaxial loading."""
    def __init__(self, umat, data, **kwds):
        super().__init__(umat, data, **kwds)
        self._grad = self.update_grad()
        self._label = 'Uniaxial'

    def _function(self, x):
        """Computes stress for a given stretch"""
        F = self._grad
        F[0,:,1,1] = F[0,:,2,2] = x
        stress = self.stress()
        return stress

    def _objective(self, params):
        """Optimization function"""
        e = self.data['strain']
        lamda = 1 + e if self._ss_type=='E' else np.exp(e)
        for param in params:
            if params[param].name!='K':
                self._umat.kwargs[params[param].name] = params[param].value
            else:
                self._umat._bulk = params[param].value
        def calcS22Abs(x):
            return abs(self._function(x)[...,1,1].ravel())
        # search for transverse stretch that gives S22=0
        lam2 = fsolve(calcS22Abs, x0=1.0/np.sqrt(lamda))
        stress = self._function(lam2)[...,0,0].ravel()
        return abs(self.data['stress'] - stress)

    def update_grad(self):
        """Update deformation gradient"""
        e = self.data['strain']
        lamda = 1 + e if self._ss_type=='E' else np.exp(e)
        n = e.shape[0]
        F = np.zeros((1,n,3,3))
        F[0,:,0,0] = lamda
        F[0,:,1,1] = F[0,:,2,2] = 1.0/np.sqrt(lamda)
        return F

class EquiBiaxial(Test):
    """Compressible biaxial loading"""
    def __init__(self, umat, data, **kwds):
        super().__init__(umat, data, **kwds)
        self._grad = self.update_grad()
        self._label = 'Equibiaxial'

    def _function(self, x):
        """Computes stress for a given stretch"""
        F = self._grad
        F[0,:,2,2] = x
        stress = self.stress()
        return stress

    def _objective(self, params):
        """Optimization function"""
        e = self.data['strain']
        lamda = 1 + e if self._ss_type=='E' else np.exp(e)
        for param in params:
            if params[param].name!='K':
                self._umat.kwargs[params[param].name] = params[param].value
            else:
                self._umat._bulk = params[param].value
        def calcS22Abs(x):
            return abs(self._function(x)[...,2,2].ravel())
        # search for transverse stretch that gives S33=0
        lam2 = fsolve(calcS22Abs, x0=1.0/np.square(lamda))
        stress = self._function(lam2)[...,0,0].ravel()
        return abs(self.data['stress'] - stress)

    def update_grad(self):
        """Update deformation gradient"""
        e = self.data['strain']
        lamda = 1 + e if self._ss_type=='E' else np.exp(e)
        n = e.shape[0]
        F = np.zeros((1,n,3,3))
        F[0,:,0,0] = F[0,:,1,1] = lamda
        F[0,:,2,2] = 1.0/np.square(lamda)
        return F

class PureShear(Test):
    """Compressible planar loading."""
    def __init__(self, umat, data, **kwds):
        super().__init__(umat, data, **kwds)
        self._grad = self.update_grad()
        self._label = 'Pure shear'

    def _function(self, x):
        """Computes stress for a given stretch"""
        F = self._grad
        F[0,:,2,2] = x
        stress = self.stress()
        return stress

    def _objective(self, params):
        """Optimization function"""
        e = self.data['strain']
        lamda = 1 + e if self._ss_type=='E' else np.exp(e)
        for param in params:
            if params[param].name!='K':
                self._umat.kwargs[params[param].name] = params[param].value
            else:
                self._umat._bulk = params[param].value
        def calcS22Abs(x):
            return abs(self._function(x)[...,2,2].ravel())
        # search for transverse stretch that gives S33=0
        lam2 = fsolve(calcS22Abs, x0=1.0/lamda)
        stress = self._function(lam2)[...,0,0].ravel()
        return abs(self.data['stress'] - stress)

    def update_grad(self):
        """Update deformation gradient"""
        e = self.data['strain']
        lamda = 1 + e if self._ss_type=='E' else np.exp(e)
        n = e.shape[0]
        F = np.zeros((1,n,3,3))
        F[0,:,0,0] = lamda
        F[0,:,1,1] = 1.0
        F[0,:,2,2] = 1.0/lamda
        return F
