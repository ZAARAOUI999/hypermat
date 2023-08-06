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

from numpy import (ndarray as NDA, array as T, sqrt as SQRT, log as LN,
                   exp as EXP)
from .._ad import Variable as SN
###############################################################################
##################### - HYPERELASTIC MATERIAL MODELS - ########################
############################################################################### 

###############################################################################
######################## - PHENOMENOLOGICAL MODELS -###########################
###############################################################################


###############################################################################
############################# - W = (I1, I2) - ################################
###############################################################################

######################### - SERIES FUNCTION MODELS - ##########################

def mooney_rivlin(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _C10, _C01 = list(_params.values())
    _W = _C10 * (_J1 - 3.0) + _C01 * (_J2 - 3.0)
    return _W

def neo_hooke(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _C10 = list(_params.values())[0]
    _W = _C10 * (_J1 - 3.0)
    return _W

def isihara(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _C10, _C20, _C01 = list(_params.values())
    _W = _C10 * (_J1 - 3.0) + _C20 * (_J1 - 3.0)**2.0 * _C01 * (_J2 - 3.0) \
       
    return _W

def biderman(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _C10, _C20, _C30, _C01 = list(_params.values())
    _W = _C10 * (_J1 - 3.0) + _C01 * (_J2 - 3.0) + _C20 * (_J1 - 3.0)**2.0 +\
        _C30 * (_J1 - 3.0)**3.0
    return _W

def james_green_simpson(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _C10, _C20, _C30, _C01, _C11 = list(_params.values())
    _W = _C10 * (_J1 - 3.0) + _C01 * (_J2 - 3.0) +  _C11 * (_J1 - 3.0) \
        * (_J2 - 3.0) + _C20 * (_J1 - 3.0)**2.0 + _C30 * (_J1 - 3.0)**3.0
    return _W

def haines_wilson(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _C10, _C20, _C30, _C01, _C02, _C11 = list(_params.values())
    _W = _C10 * (_J1 - 3.0) + _C01 * (_J2 - 3.0) +  _C11 * (_J1 - 3.0) \
        * (_J2 - 3.0) + _C20 * (_J1 - 3.0)**2.0 + _C30 * (_J1 - 3.0)**3.0 +\
           + _C02 * (_J2 - 3.0)**2.0
    return _W

def yeoh(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _C = list(_params.values())[:-1]
    _W = 0.0
    for _i, _c in enumerate(_C):
        _W += _c * (_J1 - 3.0)**(_i+1)
    return _W

def lion(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _C10, _C01, _C50 = list(_params.values())
    _W = _C10 * (_J1 - 3.0) + _C01 * (_J2 - 3.0)**2.0 * _C50 * (_J1 - 3.0)**5.0
    return _W

def haupt_sedlan(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _C10, _C30, _C01, _C02, _C11 = list(_params.values())
    _W = _C10 * (_J1 - 3.0) + _C01 * (_J2 - 3.0) +  _C11 * (_J1 - 3.0) \
        * (_J2 - 3.0) + _C30 * (_J1 - 3.0)**3.0 + _C02 * (_J2 - 3.0)**2.0
    return _W

def hartmann_neff(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _alpha = list(_params.values())[0]
    _Ci0 = list(_params.values())[1:-1:2]
    _C0j = list(_params.values())[2:-1:2]
    _W = _alpha * (_J1**3.0 - 9.0)
    for _i, _ in enumerate(_Ci0):
        _W += _Ci0[_i] * (_J1 - 3.0)**(_i+1) + _C0j[_i] * (_J2**1.5 - 3.0**1.5)**(_i+1)
    return _W

def carroll(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _A, _B, _C = list(_params.values())
    _W = _A * _J1 + _B * _J1**4.0 + _C * _J2**0.5
    return _W

def nunes(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _C1, _C2 = list(_params.values())
    _W = _C1 * (_J1 - 3.0) + (4.0 / 3.0) * _C2 * (_J2 - 3.0)**(3.0 / 4.0)
    return _W

def bahreman_darijani(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _A2, _B2, _A4, _A6 = list(_params.values())
    _W = _A2 * (_J1 - 3.0) + _B2 * (_J2 - 3.0) + _A4 * (_J1**2.0 - 2.0 * _J2 - 3.0) +\
        _A6 * (_J1**3.0 - 3.0 * _J1 * _J2)
    return _W

def zhao(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _C1, _C2, _C3, _C4 = list(_params.values())
    _W = _C1 * (_J1 - 3.0) + _C2 * (_J2 - 3.0) + _C3 * (_J1**2.0 - 2.0 * _J2 - 3.0) +\
        _C4 * (_J1**2.0 - 2.0 * _J2 - 3.0)**2.0
    return _W

################## - LIMITING CHAIN EXTENSIBILITY MODELS - ####################

def warner(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _mu, _Im = list(_params.values())
    _A = LN(1.0 - (_J1 - 3.0) / (_Im - 3.0))
    _W = - 0.5 * _mu * _Im * _A
    return _W

def kilian(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _mu, _JL = list(_params.values())
    _A = SQRT((_J1 - 3.0) / _JL)
    _W = - _mu * _JL * (LN(1.0 - _A) + _A)
    return _W

def van_der_waals(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _mu, _lambda_m, _beta, _a,  _K = list(_params.values())
    _A = -(_lambda_m**2.0 - 3.0) * (LN(1.0 - _beta) + _beta) - (2.0 / 3.0) * _a *\
        (0.5 * (_J1 - 3.0))**1.5
    _W = _mu * _A
    return _W

def gent(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _E, _Im = list(_params.values())
    _A = LN(1.0 - (_J1 - 3.0) / (_Im - 3.0))
    _W = - (_E / 6.0) * (_Im - 3.0) * _A
    return _W

def takamizawa_hayashi(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _c, _Jm = list(_params.values())
    _A = LN(1.0 - ((_J1 - 2.0) / _Jm)**2.0)
    _W = _c * _A
    return _W

def yeoh_fleming(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _A, _B, _Im, _C10 = list(_params.values())
    _C = (_Im - 3.0) * (1.0 - EXP(-_B * (_J1 - 3.0) / (_Im - 3.0)))
    _D = (_Im - 3.0) * LN(1.0 - (_J1 - 3.0) / (_Im - 3.0))
    _W = (_A / _B) * _C - _C10 * _D
    return _W

def gent_3ps(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _mu, _alpha, _Im,  _K = list(_params.values())
    _A = - _alpha * (_Im - 3.0) * LN(1.0 - (_J1 - 3.0) / (_Im - 3.0))
    _B = (1.0 - _alpha) * (_J2 - 3.0)
    _W = 0.5 * _mu * (_A + _B)
    return _W

def pucci_saccomandi(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _mu, _k, _Jm = list(_params.values())
    _A = LN(1.0 - (_J1 - 2.0) / _Jm)
    _W = _k * LN(_J2 / 3.0) - 0.5 * _mu * _Jm * _A
    return _W

def horgan_saccomandi(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _mu, _J = list(_params.values())
    _A = LN((_J**2.0 * (1.0 - _J1) + _J * _J2 - 1.0) / (_J - 1.0)**3.0)
    _W = - 0.5 * _mu * _J * _A
    return _W

def beatty(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _c, _Im = list(_params.values())
    _A = - _c * _Im * (_Im - 3.0) / (2.0 * _Im - 3.0)
    _B = LN((1.0 - (_J1 - 3.0) / (_Im - 3.0)) / (1.0 + (_J1 - 3.0) / _Im))
    _W = _A * _B
    return _W

# def horgan_murphy(_F: SN, **_params) -> SN: #TODO
#     _C, _J1, _J2, _J3 = _F.invariants
#     return

########## - POWER LAW, EXPONENTIAL OR LOGARITHMIC FUNCTION MODELS - ##########

def knowles(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _mu, _b, _n = list(_params.values())
    _a1 = _mu / (2.0 * _b)
    _a2 = (1.0 + ((_b / _n) * (_J1 - 3.0)))**_n
    _W = _a1 * (_a2 - 1.0)
    return _W

def swanson(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _AB = list(_params.values())[:-1]
    _n = len(_AB)//2
    _Ai, _alphai, _Bi, _betai = list(_params.values())[0:_n:2], _params[1:_n:2], _params[_n:-1:2], _params[_n+1:-1:2]
    _A, _B = 0, 0
    for _i, _ in enumerate(_Ai):
        _A += (_Ai[_i] / (1.0 + _alphai[_i])) * (_J1 / 3.0)**(1.0 + _alphai[_i])
        _B += (_Bi[_i] / (1.0 + _betai[_i])) * (_J2 / 3.0)**(1.0 + _betai[_i])
    _W = 1.5 * _A + 1.5 * _B
    return _W

def yamashita_kawabata(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _C1, _C2, _C3, _N = list(_params.values())
    _W = _C1 * (_J1 - 3.0) + _C2 * (_J2 - 3.0) + _C3 / (_N + 1.0) *\
        (_J1 - 3.0)**(_N + 1.0)
    return _W

def davis_de_thomas(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _A, _n, _C, _k = list(_params.values())
    _W = 0.5 * _A / (1.0 - 0.5 * _n) * (_J1 - 3.0 + _C**2.0)**(1.0 - 0.5 * _n) +\
        + _k * (_J1 - 3.0)**2.0
    return _W

def gregory(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _A, _B, _C, _m, _n = list(_params.values())
    _W = 0.5 * _A / (1.0 - 0.5 * _n) * (_J1 - 3.0 + _C**2.0)**(1.0 - 0.5 * _n) +\
        0.5 * _B / (1.0 + 0.5 * _m) * (_J1 - 3.0 + _C**2.0)**(1.0 + 0.5 * _m) 
    return _W

def modified_gregory(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _A, _alpha, _M, _B, _beta, _N = list(_params.values())
    _alpha1 = 1.0 + _alpha
    _beta1 = 1.0 + _beta
    _W = _A / _alpha1 * (_J1 - 3.0 + _M**2.0)**_alpha1 +\
        _B / _beta1 * (_J1 - 3.0 + _N**2.0)**_beta1
    return _W

def aimn(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _mu = list(_params.values())
    _W = 0.5 * _mu * (_J1/_J2 + 2.0 * _J3**0.5)
    return _W

def beda(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _mu = list(_params.values())
    _W = 0.5 * _mu * (_J1/_J2 + 2.0 * _J3**0.5)
    return _W

def lopez_pamies(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _mu = list(_params.values())
    _W = 0.5 * _mu * (_J1/_J2 + 2.0 * _J3**0.5)
    return _W

def gen_yeoh(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _mu = list(_params.values())
    _W = 0.5 * _mu * (_J1/_J2 + 2.0 * _J3**0.5)
    return _W

def hart_smith(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _mu = list(_params.values())
    _W = 0.5 * _mu * (_J1/_J2 + 2.0 * _J3**0.5)
    return _W

def veronda_westmann(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _mu = list(_params.values())
    _W = 0.5 * _mu * (_J1/_J2 + 2.0 * _J3**0.5)
    return _W

def fung_demiray(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _mu = list(_params.values())
    _W = 0.5 * _mu * (_J1/_J2 + 2.0 * _J3**0.5)
    return _W

def blatz_ko(_F: SN, **_params) -> SN:
    _, _J1, _J2, _J3 = _F.invariants
    _mu = list(_params.values())
    _W = 0.5 * _mu * (_J1/_J2 + 2.0 * _J3**0.5)
    return _W
