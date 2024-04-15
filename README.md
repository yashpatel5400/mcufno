# Winograd Neural Operators
Recent interest has honed in on leveraging such neural operators for robust control in nonlinear dynamical systems, where hand-crafted models of dynamics tend to lack insufficient fidelity to produce high-quality control. However, the deployment of such neural operator-based control to edge devices is infeasible due to the significant computational burden arising from computing the kernel integral operator, both in older graph-based operator methods and more recent Fourier Neural Operators. For this reason, we propose a new method building upon Winograd convolutions to efficiently compute the kernel integral transform on edge devices in settings where regular mesh discretizations are present. In such cases, we demonstrate that our proposed Winograd Neural Operator achieves a speedup of roughly 2x over other operator alternatives in PDE benchmark tasks.

To reproduce results of the associated manuscript, first download the following files to the `experiments/` folder. Models can be searched and downloaded from https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2987:

###### 2D Diffusion Reaction Eqn

```
# data: 2D_diff-react_NA_NA.h5
https://darus.uni-stuttgart.de/api/access/datafile/133017

# model: 2D_diff-react_NA_NA_FNO.pt
```

-------------

###### 2D Darcy Flow Eqn

```
# data: 2D_DarcyFlow_beta1.0_Train.hdf5
https://darus.uni-stuttgart.de/api/access/datafile/133219

# model: 2D_DarcyFlow_beta10.0_Train_FNO.pt
```

------------------

###### 2D Shallow Water Eqn

```
# data: 2D_rdb_NA_NA.h5
https://darus.uni-stuttgart.de/api/access/datafile/133021

# model: 2D_rdb_NA_NA_FNO.pt
```

To reproduce the experiments, simply run `./run_exp.sh`. Experiments were run on a Nvidia RTX 2080 Ti GPU with Python 3.11.8 and PyTorch version 2.2.2+cu121 (installed via pip).