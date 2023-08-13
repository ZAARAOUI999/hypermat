<p align="center">
 <img width="300" height="300" src="https://github.com/ZAARAOUI999/hypermat/assets/115699524/f45bb772-92ca-4fdb-bc49-08d3bdda6786">
 </p>
<h1 align="center">HyperMAT <br>هايبرمات</h1>
<p align="center">
 Hyperelastic formulations using an algorithmic differentiation with hyper-dual numbers in Python.
</p>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8224136.svg)](https://doi.org/10.5281/zenodo.8224136) [![Generic badge](https://img.shields.io/badge/pypi-v0.0.4-<COLOR>.svg)](https://pypi.org/project/hypermat/)


HyperMAT is based on the definitions of the second Piola-Kirchhoff stress $\large S$ and the material tangent modulus $\large \hat{C}$ given below:

<p align="center">
 $\large S = 2 \frac{\partial{W}}{\partial{C}}$ <br> $\large \hat{C} = 4 \frac{\partial^2{W}}{\partial{C}^2}$
</p>

<h2>How to use</h2>

This is a quick example of how to use:

```python
import numpy as np
import hypermat as hm

# Initialise material
umat = hm.NeoHooke(C10=0.5, K=0)

# Initialise deformation gradient values
F = np.array(hm.like(np.eye(3), np.zeros((100,100,3,3))))
F[...,0,0] = 0.25
F[...,0,2] = 0.5
F[...,2,1] = 0.3

# Get stress tensor
S = umat.jacobian(F)

# Get material tangent modulus tensor
C = umat.hessian(F)
```

> <picture>
>   <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Mqxx/GitHub-Markdown/main/blockquotes/badge/light-theme/info.svg">
>   <img alt="Info" src="https://raw.githubusercontent.com/Mqxx/GitHub-Markdown/main/blockquotes/badge/dark-theme/info.svg">
> </picture><br>
>
> Only models in the form of series function based on invariants are supported.
 
Sometimes a lucky engineer will have some tension or compression stress-strain test data, or simple shear test data. Processing and applying these data is a critical step to analyze the hyperelastic models. HyperMAT has a calibration module that can help to get the best fitted model parameters. Let's take a look on how things are going on:

```python
import os
from hypermat import NeoHooke, Yeoh, read_file, to_dict, Uniaxial


#Prepare material models
umat1 = NeoHooke(C10=1.5,K=2000)
umat2 = Yeoh(C10=0.5,C20=-0.01,C30=0.2, K=2000)

#Prepare experimental data
cdir = os.getcwd()
dataset = read_file(cdir+'//_hypermat//_calibration//_data//_data_2.csv', delimiter=',', dtype=np.float64)
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
 <img src="https://github.com/ZAARAOUI999/hypermat/assets/115699524/5513f0ef-733f-40c6-ad99-369814ae97ee">
</p>

```
HyperMAT fitted parameters
{'C10': 0.6624343754510106}
{'C10': 0.5903745146776757, 'C20': -0.09056730756209555, 'C30': 0.3065185192428228}
MCalibration fitted parameters
{'C10': 0.623489155}
{'C10': 0.585555703, 'C20': -0.0846386036, 'C30': 0.304613717}
```
<h2>License</h2>

HyperMAT- Hyperelastic formulations using an algorithmic differentiation with hyper-dual numbers in Python, (C) 2023 Mohamed ZAARAOUI, Tunisia.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
