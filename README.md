<p align="center">
 <img width="300" height="300" src="https://github.com/ZAARAOUI999/hypermat/assets/115699524/c378b967-6457-48d7-afd3-db25973f2325">
 </p>



<h1 style="color:purple;" align="center" font-family= 'Varela Round'>HyperMAT <br>هايبرمات</h1>
<p align="center">
 Hyperelastic formulations using an algorithmic differentiation with hyper-dual numbers in Python.
</p>

[![Generic badge](https://img.shields.io/badge/pypi-v0.1.1-<COLOR>.svg)](https://pypi.org/project/hypermat/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8285247.svg)](https://doi.org/10.5281/zenodo.8285247) [![Downloads](https://static.pepy.tech/badge/hypermat/week)](https://pepy.tech/project/hypermat)


HyperMAT is based on the definitions of the first Piola-Kirchhoff stress $\large P$ and the tangent modulus $\large \hat{A}$ given below:

<p align="center">
 $\large P = \frac{\partial{W}}{\partial{F}}$ <br> $\large \hat{A} = \frac{\partial^2{W}}{\partial{F}^2}$
</p>

<h2>How to use</h2>

This is a quick example of how to use:

```python
import numpy as np
import hypermat as hm

# Initialise material
umat = hm.NeoHooke(C10=0.5, K=0)

# Initialise deformation gradient values
np.random.seed(125161)
F = (np.eye(3) + np.random.rand(50, 8, 3, 3) / 10).T

# Get stress tensor
P = umat.jacobian(F)

# Get tangent modulus tensor
A = umat.hessian(F)
```
 
Sometimes a lucky engineer will have some tension or compression stress-strain test data, or simple shear test data. Processing and applying these data is a critical step to analyze the hyperelastic models. HyperMAT has a calibration module that can help to get the best fitted model parameters. Let's take a look on how are things going on:

```python
import os
from hypermat import NeoHooke, Yeoh, read_file, to_dict, Uniaxial


#Prepare material models
umat1 = NeoHooke(C10=1.5,K=2000)
umat2 = Yeoh(C10=0.5,C20=-0.01,C30=0.2, K=2000)

#Prepare experimental data
dir_path = os.path.dirname(os.path.realpath(__file__))
dataset = read_file(dir_path+'//_data//_data_2.csv', delimiter=',', dtype=float)
data = to_dict(dataset[1:,:], ['time', 'strain', 'stress'])
strain = data['strain']
stress = data['stress']

#Choose loading type (Uniaxial, Biaxial or Shear)
test1 = Uniaxial(umat1, data)
test2 = Uniaxial(umat2, data)

#Plot experimental data
test1.plot()

#Fit parameters
test1.fit_data([0,0],[20,2000],[True, False])
test2.fit_data([0,-20,-20,0],[20,20,20,2000],[True,True,True, False])

#Plot results
test1.plot_model(c='r')
test2.plot_model(c='g')
```
You should get something like that:

<p align="center">
 <img src="https://github.com/ZAARAOUI999/hypermat/assets/115699524/c38d9db0-9497-40d4-a80b-e832cde3f4dc">
</p>

```
HyperMAT fitted parameters
{'C10': 0.6624343754510106}
{'C10': 0.5903745146776757, 'C20': -0.09056730756209555, 'C30': 0.3065185192428228}
MCalibration fitted parameters
{'C10': 0.623489155}
{'C10': 0.585555703, 'C20': -0.0846386036, 'C30': 0.304613717}
```

A special thank you goes to [Dutzler](https://github.com/adtzlr), for providing us many powerful tools such as [TensorTrax](https://github.com/adtzlr/tensortrax) and [hyperelastic](https://github.com/adtzlr/hyperelastic).
<h2>License</h2>

HyperMAT- Hyperelastic formulations using an algorithmic differentiation with hyper-dual numbers in Python, (C) 2023 Mohamed ZAARAOUI, Tunisia.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
