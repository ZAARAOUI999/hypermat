
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

from ._models import (neo_hooke, mooney_rivlin, isihara, biderman,
                                   yeoh, james_green_simpson, ogden)
from ._energy import StrainEnergy


###############################################################################
######################## - PHENOMENOLOGICAL MODELS -###########################
###############################################################################


###############################################################################
############################# - W = (I1, I2) - ################################
###############################################################################

######################### - SERIES FUNCTION MODELS - ##########################


class NeoHooke(StrainEnergy):
    """Neo-Hooke hyperelastic model"""
    def __init__(self, **kwargs):
        self._label = 'Neo-Hooke'
        super().__init__(neo_hooke, **kwargs)

class MooneyRivlin(StrainEnergy):
    """Mooney-Rivlin hyperelastic model"""
    def __init__(self, **kwargs):
        self._label = 'Mooney-Rivlin'
        super().__init__(mooney_rivlin, **kwargs)

class Isihara(StrainEnergy):
    """Isihara hyperelastic model"""
    def __init__(self, **kwargs):
        self._label = 'Isihara'
        super().__init__(isihara, **kwargs)

class Biderman(StrainEnergy):
    """Biderman hyperelastic model"""
    def __init__(self, **kwargs):
        self._label = 'Biderman'
        super().__init__(biderman, **kwargs)

class Yeoh(StrainEnergy):
    """Yeoh hyperelastic model"""
    def __init__(self, **kwargs):
        self._label = 'Yeoh'
        super().__init__(yeoh, **kwargs)

class JamesGreenSimpson(StrainEnergy):
    """James-Green-Simpson hyperelastic model"""
    def __init__(self, **kwargs):
        self._label = 'JamesGreenSimpson'
        super().__init__(james_green_simpson, **kwargs)

###############################################################################
############################# - W = (λ1, λ2, λ3) - ################################
###############################################################################

class Ogden(StrainEnergy):
    """Ogden hyperelastic model"""
    def __init__(self, **kwargs):
        self._label = 'Ogden'
        super().__init__(ogden, **kwargs)
##############################################################################
