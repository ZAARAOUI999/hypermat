<p align="center">
 <img width="300" height="300" src="https://github.com/ZAARAOUI999/hypermat/assets/115699524/f45bb772-92ca-4fdb-bc49-08d3bdda6786">
 </p>
<h1 align="center">HyperMAT <br>هايبرمات</h1>
<p align="center">
 Hyperelastic formulations using an algorithmic differentiation with hyper-dual numbers in Python.
</p>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8224136.svg)](https://doi.org/10.5281/zenodo.8224136)


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

<h2>License</h2>

HyperMAT- Hyperelastic formulations using an algorithmic differentiation with hyper-dual numbers in Python, (C) 2023 Mohamed ZAARAOUI, Tunisia.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
