SA-GPR
======

This repository contains a Python code for carrying out Symmetry-Adapted Gaussian Process Regression (SA-GPR) for the machine-learning of tensors. For more information, see:

1. Andrea Grisafi, David M. Wilkins, Gabor Csányi, Michele Ceriotti, "Symmetry-Adapted Machine Learning for Tensorial Properties of Atomistic Systems", Phys. Rev. Lett. 120, 036002 (2018)

2. David M. Wilkins, Andrea Grisafi, Yang Yang, Ka Un Lao, Robert A. DiStasio, Michele Ceriotti, "Accurate Molecular Polarizabilities with Coupled-Cluster Theory and Machine Learning", Proc. Natl. Acad. Sci. 116, 3401 (2019)

3. Andrea Grisafi, David M. Wilkins, Benjamin A. R. Meyer, Alberto Fabrizio, Clemence Corminboeuf, Michele Ceriotti, "A Transferable Machine-Learning Model of the Electron Density", ACS Cent. Sci. 5, 57 (2019)

4. Andrea Grisafi, Michele Ceriotti, "Incorporating long-range physics in atomic-scale machine learning", J. Chem. Phys. 151, 204105 (2019)

5. Félix Musil, Michael J. Willatt, Mikhail A. Langovoy, Michele Ceriotti, "Fast and Accurate Uncertainty Prediction in Chemical Machine Learning", J. Chem. Theory Comput. 15, 906 (2019)

Versions
========

The current version of SOAPFAST (v3.0.1) is written in python 3. It has the same functionality as the previous release (v2.3), which is written in python 2.

Requirements
============

The python packages :code:`ase`, :code:`scipy`, :code:`sympy` and :code:`cython` are required to run this code. 

Installation
============

This program is installed using a makefile, found in the :code:`soapfast` subdirectory. To install the python packages needed, the command :code:`make python` is used. To compile the cython parts of this code, :code:`make cython` is used, and to compile the fortran code needed for long-range descriptors, :code:`make LODE`. The commands :code:`make` or :code:`make all` will compile both the cython and fortran codes. To remove object files run :code:`make clean`.

Note that the makefile is set up to install python packages only for the current user. If you have the requisite permissions and want instead to install them for all users, the :code:`PIPOPTS` variable in the makefile should be set to a blank string.

Workflow
========

There are two steps to applying SA-GPR to a physical problem:

1. Calculation of the similarities (kernels) between systems.
2. Minimization of the prediction error and generation of the weights for prediction.

The first step is applied by running :code:`sagpr_get_PS` and :code:`sagpr_get_kernel`, and the second by running :code:`sagpr_train`. The following examples aim to make this workflow clearer.

Examples
========

The example/ directory contains four sub-directories, with data for the response properties of water monomers, water dimers, Zundel cations and boxes of 32 water molecules. We illustrate three examples of how to use SA-GPR for these directories.

Before starting, source the environment settings file :code:`env.sh` using :code:`$ source env.sh`.

1. Water Monomer - Cartesian Tensor Learning
--------------------------------------------

We start by learning the energy of a water monomer. The energy only has a scalar (L=0) component, so that the kernel used is the standard SOAP kernel. We begin by computing the L=0 kernels between the 1000 molecules. The first step is to find the power spectra of these configurations:

::

  $ cd example/water_monomer
  $ sagpr_get_PS -n 8 -l 6 -rc 4.0 -sg 0.3 -c 'O' -lm 0 -f coords_1000.xyz -o PS0

Since most of these options are the default, we could equally well use the command:

::

 $ sagpr_get_PS -c 'O' -lm 0 -f coords_1000.xyz -o PS0

Next, we combine the power spectrum with itself to obtain the kernel matrix:

::

  $ sagpr_get_kernel -z 2 -ps PS0.npy -s PS0_natoms.npy -o kernel0

We create a nonlinear (zeta=2) L=0 kernel matrix, :code:`kernel0.npy`, using the coordinates in coords_1000.xyz, with Gaussian width 0.3 Angstrom, an angular cutoff of l=6, 8 radial functions, a radial cutoff of 4 Angstrom, central atom weighting of 1.0 and centering of the environment on oxygen atoms. This kernel can now be used to perform the regression:

::

  $ sagpr_train -r 0 -reg 1e-8 -f coords_1000.xyz -k kernel0.npy -p potential -rdm 200 -pr

The regression is performed for a rank-0 tensor, using the kernel file we produced, with a training set containing 200 randomly selected configurations. The file :code:`coords_1000.xyz` contains the energies of the 1000 molecules under the heading "potential", and we use a regularization parameter of 1e-8. By varying the value of the :code:`ftr` variable from 0 to 1, it is possible to create a learning curve which spans a range of training examples from 0 to the full data set.

We will next learn the polarizability of the monomers: since the symmetric polarizability tensor has an L=0 and an L=2 component, we must also compute the L=2 kernel:

::

  $ sagpr_get_PS -c 'O' -lm 2 -f coords_1000.xyz -o PS2
  $ sagpr_get_kernel -z 2 -ps PS2.npy -ps0 PS0.npy -s PS2_natoms.npy -o kernel2

The regression can be performed in the same way as before:

::

  $ sagpr_train -r 2 -reg 1e-8 1e-8 -f coords_1000.xyz -k kernel0.npy kernel2.npy -p alpha -rdm 200 -pr

Here we must specify both the L=0 and L=2 kernels to be used, as well as the regularizations in both cases. The user is encouraged to optimize these regularizations.

2. Zundel Cation - Cartesian Tensor Learning
--------------------------------------------

Next, we learn the dipole moment of the zundel cation. This has one spherical component, transforming as L=1, so we must first build the respective kernel. Rather than calculating the power spectrum for the entire set of coordinates in one go, we begin by splitting the coordinates into smaller blocks, for each of which we compute the power spectra. To split the coordinates we use the :code:`split_dataset` script:

::

  $ cd example/water_zundel
  $ split_dataset.py -f coords_1000.xyz -n 20 -o zundel

This will create 20 data files containing the coordinates of 50 Zundel cations each. For each of these files we then compute the power spectrum:

::

  $ for i in {0..19}
  $  do
  $  sagpr_get_PS -lm 1 -f zundel_${i}.xyz -o PS1_${i} > /dev/null &
  $ done

The power spectra thus created must then be combined together to find the power spectrum for the entire dataset:

::

  $ rebuild_power_spectrum.py -lm 1 -c coords_1000.xyz -nb 20 -f PS1

This creates :code:`PS1.npy`, which contains the full power spectrum. The next step, as usual, is to build the kernel. For this we also build the L=0 power spectrum in order to find the nonlinear L=1 kernel.

::

  $ sagpr_get_PS -lm 0 -f coords_1000.xyz -o PS0
  $ sagpr_get_kernel -z 2 -ps PS1.npy -ps0 PS0.npy -s PS0_natoms.npy -o kernel1

We now use the kernel built to perform regression. Rather than do the regression and prediction in one go, we instead demonstrate the generation of an SA-GPR model using :code:`sagpr_train` and the prediction of the dipole moments using :code:`sagpr_prediction`. Firstly we train an SA-GPR model:

::

  $ sagpr_train -r 1 -reg 1e-8 -f coords_1000.xyz -k kernel1.npy -p mu -rdm 200

Because we have not used the :code:`-pr` flag here, this code does not give any predictions, it only prints out the details of the model generated (note that these will be printed regardless of whether you use this flag; the :code:`-w` flag allows you to control the name of these output files). Now, using this model we predict the dipole moments for this system. In addition to the weights generated, we need to know the kernel between the testing and training sets. For this, we can use the following code:

::

  $ python
  $ >>> import numpy as np
  $ >>> wt = np.load("weights_1.npy",allow_pickle=True)
  $ >>> kr = np.load("kernel1.npy")
  $ >>> trr = wt[3]
  $ >>> ter = np.setdiff1d(range(1000),trr)
  $ >>> ktest = np.zeros((800,200,3,3),dtype=float)
  $ >>> for i in range(800):
  $ ...     for j in range(200):
  $ ...             ktest[i,j] = kr[ter[i],trr[j]]
  $ ...
  $ >>> np.save("ker_test.npy",ktest)

Because this is quite a contrived example (in this case, it is of course easier not to use :code:`sagpr_prediction` and just to do the predictions with the regression code), this snippet is not given as a separate script. However, it is important to note that a list of the configurations used in training the model is stored in the third record of the weights.

Next, we use this kernel to carry out the prediction:

::

  $ sagpr_prediction -w weights -r 1 -k ker_test.npy -o prediction

Using the model generated in the previous step, we predict the dipole moments. These are printed in both :code:`prediction_cartesian.txt` and :code:`prediction_L1.txt` (note that the latter is not the same as the former, and the order of elements differs due to the definition of the L=1 component). To test the quality of this prediction we must compare these results with the true answers. Although these are tabulated in the output files, we could also use a method such as the following:

::

  $ python
  $ >>> from ase.io import read
  $ >>> import numpy as np
  $ >>> xyz = read("coords_1000.xyz",':')
  $ >>> wt = np.load("weights_1.npy",allow_pickle=True)
  $ >>> trr = wt[3]
  $ >>> ter = np.setdiff1d(range(1000),trr)
  $ >>> corrfile = open("compare_cartesian.out","w")
  $ >>> for i in range(len(ter)):
  $ ...     dipole = xyz[ter[i]].info["mu"]
  $ ...     print (' '.join(str(e) for e in list(dipole)),file=corrfile)
  $ ...

The file :code:`compare_cartesian.out` contains the correct values of the dipole moments for comparison. Carrying out this comparison with a randomly chosen training set:

::

  $ paste compare_cartesian.out prediction_cartesian.txt | awk 'BEGIN{err=0.0;n=0}{n++;err += ($1 - $4)**2 + ($2 - $5)**2 + ($3 - $6)**2}END{print (err/n)**0.5}'

we find a root mean squared error of 0.003 a.u., which can be compared to the root mean square dipole moment of 0.675 a.u., for an intrinsic error of about 0.5%.

3. Bulk Water - Polarizability and Sparsification
-------------------------------------------------

We now learn the polarizabilities of bulk water systems. This is a more challenging problem because the large systems mean that we could end up with quite memory-intensive calculations. The solution is to sparsify the power spectra. This means that some small subset of the spherical harmonic components is kept.

In order to sparsify our power spectra, we first need some sparsification parameters. To generate sparsified power spectra for the L=0 component:

::

  $ cd example/water_bulk
  $ sagpr_get_PS -f coords_1000.xyz -lm 0 -p -nc 200 -o PS0

Here we take 200 spherical harmonic components (which is about a ninth as many as the number, 1792, that would be present in the unsparsified power spectrum). It should be noted that no effort has been made here to check on the optimum number of components to be kept, and the user is encouraged to perform this check themselves. Particular attention should be paid to the output of the code, which gives the smallest eigenvalue of the A-matrix used in sparsification. This matrix should be positive definite, so a sparsification that leads to negative eigenvalues -- particularly large negative eigenvalues -- should be treated with suspicion. The list of power spectrum columns retained and the A matrix, which are required for further sparsifying power spectra, are also printed.

We now combine this sparsified power spectrum as usual to give a kernel:

::

  $ sagpr_get_kernel -ps PS0.npy -z 2 -s PS0_natoms.npy -o kernel0

In order to learn the polarizability, we will also need an L=2 kernel. Because sparsification can take some time, and this part has the potential to be very memory-intensive, instead of using the entire set of coordinates to sparsify we can use some subset of it instead. To use a randomly chosen 500 frames to generate the sparsification details we can use the command:

::

  $ sagpr_get_PS -f coords_1000.xyz -lm 2 -p -nc 100 -ns 500 -sm 'random' -o PS2

Here we are decreasing the number of spherical components from 6656 to 100, which will considerably speed up the combination to give a kernel. The details thus generated are then used to sparsify the entire set of coordinates:

::

  $ sagpr_get_PS -f coords_1000.xyz -lm 2 -p -sf PS2 -o PS2_sparse

Note that we could also, if desired, split this calculation into smaller parts as in the previous example. Now, we build the kernel as before:

::

  $ sagpr_get_kernel -z 2 -ps PS2_sparse.npy -ps0 PS0.npy -s PS2_sparse_natoms.npy -o kernel2

Having obtained these kernels, we will build a SA-GPR model to predict the polarizability.

::

  $ sagpr_train -r 2 -reg 1e-8 1e-5 -f coords_1000.xyz -k kernel0.npy kernel2.npy -p alpha -rdm 500 -pr -t 1.0

The errors in doing this prediction are quite high, but we could decrease them by retaining more spherical components when sparsifying. Note that the :code:`-t 1.0` flag ensures we do not learn the apparent L=1 component of this tensor. We set the threshold for discounting a component at 1.0 atomic units, meaning that we learn the L=0 and L=2, but not the L=1. This threshold should be set according to the error in calculation of the alpha tensor. Note that if we would like to learn this component (i.e. if it it physical), this can be done by computing an L=1 kernel and including this in the arguments, without the threshold flag.

4. Water Monomer - Spherical Tensor Learning
--------------------------------------------

Rather than learning the full polarizability of the water monomers, as in example 1, we could instead learn just the L=2 component. For this, we will rebuild the L=2 kernel centering both on O and on H atoms (unlike in example 1, where we centered only on O atoms):

::

  $ cd example/water_monomer
  $ sagpr_get_PS -lm 0 -f coords_1000.xyz
  $ sagpr_get_PS -lm 2 -f coords_1000.xyz
  $ sagpr_get_kernel -z 2 -ps PS2.npy -ps0 PS0.npy -s PS2_natoms.npy -o kernel2

Because we have not specified any centres, the code will take all of the atoms present as centres (i.e., H and O). Note that in this case, we have rebuilt the L=0 power spectrum as well, for creation of the nonlinear kernel. We don't actually need this power spectrum, as we could use our old power spectra centered only on O -- so this can be used instead if the user prefers.

Having rebuilt :code:`kernel2.npy`, we will now use it to learn the L=2 component of the polarizability. Firstly we must isolate this component:

::

  $ sagpr_cart_to_sphr -f coords_1000.xyz -p alpha -r 2 -o processed_coords_1000.xyz

This command splits the alpha property in :code:`coords_1000.xyz` into spherical components, and creates :code:`processed_coords_1000.xyz` containing alpha_L0 and alpha_L2. Next, we can run the regression code with the :code:`-sp` flag to learn the L=2 spherical component:

::

  $ sagpr_train -reg 1e-8 -f processed_coords_1000.xyz -p alpha_L2 -r 2 -k kernel2.npy -rdm 200 -pr -sp

The L=2 error here should be compared to that obtained in example 1.

5. Bulk Water - Environment Sparsification and Ice Tensor Prediction
--------------------------------------------------------------------

When training a model to predict the dielectric tensors in bulk water, there is likely to be a fair amount of redundancy: using 1000 configurations with 32 water molecules each, we have 96,000 environments used for training. In addition to sparsification on the spherical components, we can further sparsify on the training environments: this has the potential to save memory both when building the kernels between training and testing systems and when doing the regression.

We start with the power spectra generated in example 3, and take 500 environments from each using furthest-point sampling. To generate a list of environments to be retained, we first have to convert the original power spectrum into an environmental power spectrum:

::

  $ cd example/water_bulk
  $ mv PS2_sparse.npy PS2.npy;mv PS2_sparse_natoms.npy PS2_natoms.npy
  $ get_atomic_power_spectrum.py -lm 0 -p PS0.npy -f coords_1000.xyz -o PS0_sparse_atomic
  $ get_atomic_power_spectrum.py -lm 2 -p PS2.npy -f coords_1000.xyz -o PS2_sparse_atomic

Rather than having a row for each molecule, these power spectra have a row for each environment. The next step is to get a list of the 500 furthest-point environments for each power spectrum. Firstly, we have to produce a power spectrum file that has each environment as a separate entry, after which the FPS details can be found:

::

  $ sagpr_do_env_fps -p PS0_sparse_atomic.npy -n 500 -o FPS_0
  $ sagpr_do_env_fps -p PS2_sparse_atomic.npy -n 500 -o FPS_2

The next step is to apply these FPS details to get a sparsified power spectrum:

::

  $ sagpr_apply_env_fps -p PS0_sparse_atomic.npy -sf FPS_0_rows -o PS0_full_sparsified_atomic
  $ sagpr_apply_env_fps -p PS2_sparse_atomic.npy -sf FPS_2_rows -o PS2_full_sparsified_atomic

These steps take only the furthest-point sampled rows of these two power spectra and produce two outputs which have been sparsified both on the spherical components and on the environments. In order to build a model, we now need to find a number of kernels: namely, the kernels between the sparsified power spectra and themselves, and between the sparsified power spectra and the power spectra that have not been sparsified on environments (this will allow us to build a model where we reduce from the situation with all environments to a situation with fewer environments).

Now, we build kernels to be used in regression:

::

  $ sagpr_get_kernel -ps PS0.npy PS0_full_sparsified_atomic.npy -s PS0_natoms.npy NONE -z 2 -o KERNEL_L0_NM
  $ sagpr_get_kernel -ps PS0_full_sparsified_atomic.npy -s NONE -z 2 -o KERNEL_L0_MM
  $ sagpr_get_kernel -ps PS2.npy PS2_full_sparsified_atomic.npy -ps0 PS0.npy PS0_full_sparsified_atomic.npy -s PS2_natoms.npy NONE -z 2 -o KERNEL_L2_NM
  $ sagpr_get_kernel -ps PS2_full_sparsified_atomic.npy -ps0 PS0_full_sparsified_atomic.npy -s NONE -z 2 -o KERNEL_L2_MM

The regression is then performed to give weights:

::

  $ sagpr_train -r 2 -reg 1e-8 1e-5 -ftr 1.0 -f coords_1000.xyz -sf KERNEL_L0_NM.npy KERNEL_L0_MM.npy KERNEL_L2_NM.npy KERNEL_L2_MM.npy -p alpha -sel 0 500 -m 'pinv' -t 1.0

Then, we could use these weights, for example, to predict the properties of the training set:

::

  $ sagpr_prediction -r 2 -k KERNEL_L0_NM.npy KERNEL_L2_NM.npy -o prediction

Proceeding as before, we find the errors to be about 10% of the intrinsic variation of the dataset (a little worse than the unsparsified case, but as before this can be modified by retaining a different number of environments) More interesting, however, is to use this model for extrapolation: that is, to predict the properties of systems that are outside of the training set. To do this, we can use the five ice configurations in :code:`ice.xyz`. Firstly, we must build the power spectra and the kernels between the training and testing set:

::

  $ sagpr_get_PS -f ice.xyz -lm 0 -p -sf PS0 -o PS0_ice
  $ sagpr_get_PS -f ice.xyz -lm 2 -p -sf PS2 -o PS2_ice
  $ sagpr_get_kernel -z 2 -ps PS0_ice.npy PS0_full_sparsified_atomic.npy -s PS0_ice_natoms.npy NONE -o KERNEL_L0_ice_train
  $ sagpr_get_kernel -z 2 -ps PS2_ice.npy PS2_full_sparsified_atomic.npy -ps0 PS0_ice.npy PS0_full_sparsified_atomic.npy -s PS0_ice_natoms.npy NONE -o KERNEL_L2_ice_train

We can then use these kernels to do the prediction:

::

  $ sagpr_prediction -r 2 -k KERNEL_L0_ice_train.npy KERNEL_L2_ice_train.npy -o prediction_ice

6. Water Dimer - Environment Sparsification All-In-One
------------------------------------------------------

The script :code:`src/scripts/train_predict_env_sparse.py` is an all-in-one script that takes in a set of power spectra, builds and tests an environmentally-sparsified model for a property. Although this script involves quite a large number of command-line arguments, by putting together a significant part of the workflow in the regression task we should be able to save some time.

The only ingredients we need are pre-generated power spectra, which have been sparsified only on features. We will begin by generating these power spectra for the water dimers:

::

  $ cd example/water_dimer
  $ for lm in 0 1 2 3
  $ do
  $ sagpr_get_PS -f coords_1000.xyz -lm ${lm} -o PS${lm} &
  $ done

With these power spectra we can learn any of the properties in which we might be interested. Here, we will learn them all. Firstly, the energy:

::

  $ train_predict_env_sparse.py -p PS0.npy -fr coords_1000.xyz -s PS0_natoms.npy -sm rdm -n 800 -e 500 -z 2 -k 0 -pr potential -reg 1e-7

This command builds a sparse model for the potential energy of water dimers. :code:`-sm rdm -n 800` means that it will take 800 dimers at random as the training set; :code:`-e 500` means that of the 2400 environments present we will take 500 of them, and :code:`-k 0` means that the kernel coming from the power spectrum in position 0 of the :code:`-p PS0.npy` argument (i.e., :code:`PS0.npy`) will be used for prediction. Similarly, we can build models for the dipole moment, polarizability and hyperpolarizability:

::

  $ train_predict_env_sparse.py -p PS0.npy PS1.npy -fr coords_1000.xyz -s PS1_natoms.npy -sm rdm -n 800 -e 500 -z 2 -k 1 -pr mu -reg 1e-6
  $ train_predict_env_sparse.py -p PS0.npy PS2.npy -fr coords_1000.xyz -s PS2_natoms.npy -sm rdm -n 800 -e 500 -z 2 -k 0 1 -pr alpha -reg 1e-8 1e-5
  $ train_predict_env_sparse.py -p PS0.npy PS1.npy PS3.npy -fr coords_1000.xyz -s PS3_natoms.npy -sm rdm -n 800 -e 500 -z 2 -k 1 2 -pr beta -reg 1e-7 1e-5

Note that the L=0 power spectrum is always specified as the first argument, and that the :code:`-k` argument denotes which of the power spectra give rise to kernels that will actually be used to build the SA-GPR model (whereas in some cases the L=0 is only used to build a nonlinear kernel).

7. Learning Atomic Properties
-----------------------------

SA-GPR can be used to learn the properties of individual atoms. A dataset in which the water monomers are dressed with the absolute value of the quantum-mechanical force acting on them is given, and we show here how to learn this property for both O and H atoms.

::

  $ cd example/water_atomic_forces

We firstly need to find the atomic power spectra for both types of atom individually:

::

  $ sagpr_get_PS -f coords_800.xyz -lm 0 -o PS0 -a

This produces the files :code:`PS0_atomic_O.npy` and :code:`PS0_atomic_H.npy`, each of which can be used to build atomic kernels:

::

  $ for atom in O H;do sagpr_get_kernel -ps PS0_atomic_${atom}.npy -s NONE -z 2 -o KER0_atomic_${atom} & done

We now have the two kernels :code:`KER0_atomic_O.npy` and :code:`KER0_atomic_H.npy`. These are all we need to do atomic regression. We now choose a training set using farthest-point sampling:

::

  $ sagpr_do_env_fps -p PS0_atomic_O.npy -n 800 -o fps_O
  $ sagpr_do_env_fps -p PS0_atomic_H.npy -n 1600 -o fps_H

These routines give us FPS ordering of the entire set, so we will want to choose some fraction as a training set. Taking 500 atoms for O and 1000 for H, we obtain training set selections:

::

  $ python
  $ >>> import numpy as np
  $ >>> fpO = np.load("fps_O_rows.npy")
  $ >>> fpH = np.load("fps_H_rows.npy")
  $ >>> np.save("selection_O.npy",fpO[:500])
  $ >>> np.save("selection_H.npy",fpO[:1000])

Finally, we do the regression:

::

  $ sagpr_train -f coords_800.xyz -r 0 -reg 1e-11 -p force -sel selection_O.npy -pr -k KER0_atomic_O.npy -c 'O'
  $ sagpr_train -f coords_800.xyz -r 0 -reg 1e-11 -p force -sel selection_H.npy -pr -k KER0_atomic_H.npy -c 'H'

Using the FPS details generated when making this example, the atomic regression for oxygen gives an RMSE of 0.202945510808 a.u. and the atomic regression for hydrogen gives 0.27160254007 a.u.; these can be compared to the intrinsic deviations within the dataset of 7.20049 a.u. and 5.37668 a.u. respectively (that is, relative errors of 2.8 and 5.1%).

8. Learning Asymmetric Tensors
------------------------------

The prediction of asymmetric properties is also possible with this code. To showcase this feature, a dataset has been included that contains a single molecule to which random rotations have been applied both to its coordinates and to its polarizability, and the same molecule to which an asymmetric part has been added to the polarizability before randomly rotating it.

::

  $ cd example/asymmetry

We can observe the difference between the two polarizabilities using the command :code:`list_spherical_components.py`:

::

  $ list_spherical_components.py -f symmetric.xyz -p alpha -r 2
  $ list_spherical_components.py -f asymmetric.xyz -p alpha -r 2

We see in the first case that the symmetric polarizability tensor has the familiar L=0 and L=2 spherical components, but that the asymmetric case has an additional L=1 component, described as being imaginary (because the L=1 part of a rank-2 tensor transforms as the imaginary unit times a spherical harmonic). In order to predict the polarizability in the asymmetric case, we will thus have to build L=0, L=1 and L=2 kernels:

::

  $ for lm in 0 1 2;do sagpr_get_PS -lm ${lm} -o PS${lm} -f asymmetric.xyz;done
  $ for lm in 0 1 2;do sagpr_get_kernel -z 2 -ps PS${lm}.npy -ps0 PS0.npy -s PS0_natoms.npy -o KER${lm};done 

Having built these kernels we can carry out the regression straightforwardly as before:

::

  $ sagpr_train -r 2 -reg 1e-8 1e-8 1e-8 -f asymmetric.xyz -p alpha -k KER0.npy KER1.npy KER2.npy -sel 0 80 -pr

Note that the relative error in learning the L=0 component is very large (around 100%); this is simply because these coordinates were produced by applying random rigid-body rotations to the molecule. For the same reason, the L=1 and L=2 components are learned with 0% error. Rather than comparing these numbers, we can check on the quality of the prediction by using the :code:`prediction_cartesian.txt` output file:

::

  $ cat prediction_cartesian.txt | awk 'BEGIN{n=0}{n++;for (i=1;i<=9;i++){x[i] += $i}}END{for (i=1;i<=9;i++){printf "%f ",x[i]/n};printf "\n"}' > avg.out;cat avg.out prediction_cartesian.txt | awk 'BEGIN{n=0;std=err=0.0}{if (n==0){n=1;for (i=1;i<=9;i++){x[i]=$i}}else{for (i=1;i<=9;i++){std += ($i - x[i])**2;err += ($i - $(i+9))**2}}}END{print (err/std)**0.5}';rm avg.out

We obtain an error of 5e-7% in predicting the asymmetric polarizability tensor. It should be noted that this feature has not yet been tested using data that was *not* produced by a rigid rotation.

9. Second Hyperpolarizability Learning
--------------------------------------

We next take the learning of the second hyperpolarizability tensor (gamma) of water monomers. The previous incarnation of SA-GPR code was limited to learning tensor orders up to third, so we show here how to deal with a general tensor order. The file :code:`water_gamma.xyz` is provided with these tensors (computed using a smaller cc-pVDZ basis set than all of the other response properties).

::

  $ cd example/water_monomer

The first step is to find which spherical kernels we must produce in order to learn this property.

::

  $ list_spherical_components.py -f coords_gamma.xyz -p gamma -r 4

We see that we need to build kernels of order 0, 2 and 4. This can be done with a simple for loop:

::

  $ for lm in 0 2 4;do sagpr_get_PS -lm ${lm} -f coords_gamma.xyz -o PS${lm} & done
  $ for lm in 0 2 4;do sagpr_get_kernel -z 2 -ps PS${lm}.npy -ps0 PS0.npy -s PS${lm}_natoms.npy -o KER${lm} & done

Having built these kernels we can now carry out the regression to predict the gamma tensors:

::

  $ sagpr_train -r 4 -reg 1e-9 1e-9 1e-9 -f coords_gamma.xyz -p gamma -k KER0.npy KER2.npy KER4.npy -sel 0 800 -pr 

The overall error in learning these tensors is 0.457 a.u. (which is 0.25% of the intrinsic deviation of the data).

10. Different Methods for Environmental Sparsification
------------------------------------------------------

To highlight the different methods for including environmental sparsification in the regression, we will learn the scalar component of the polarizability of the QM7b set (see ref. [2]).

::

  $ cd example/qm7b

Since we are provided with the full polarizability tensor, we first need to take the trace. Having done so, we will then split the set up into a training set comprising 5400 molecules and a test set containing 1811 molecules.

::

  $ cartesian_to_spherical.py -f coords.xyz -o coords_trace.xyz -p alpha -r 2
  $ python
  $ >>> from ase.io import read,write
  $ >>> frames = read("coords_trace.xyz",':')
  $ >>> write("train.xyz",frames[:5400])
  $ >>> write("test.xyz",frames[5400:])

Next, we get the scalar power spectrum for the training set, sparsified on spherical components.

::

  $ sagpr_get_PS -f train.xyz -c H C N O S Cl -s H C N O S Cl -lm 0 -nc 600 -o PS0_train

Using the sparsification details for this set, we get the power spectrum for the testing set.

::

  $ sagpr_get_PS -f test.xyz -c H C N O S Cl -s H C N O S Cl -lm 0 -sf PS0_train -o PS0_test

To get an idea of how good our sparsified models are, we will begin by building an unsparsified model. Firstly, we build the kernels and do the regression as usual, then predict on the training set.

::

  $ sagpr_get_kernel -z 2 -ps PS0_train.npy -s PS0_train_natoms.npy -o K0
  $ sagpr_get_kernel -z 2 -ps PS0_test.npy PS0_train.npy -s PS0_test_natoms.npy PS0_train_natoms.npy -o K0_TT
  $ sagpr_train -r 0 -reg 1e-9 -f train.xyz -p alpha_L0 -k K0.npy -sel 0 5400 -w weights_all_env -perat
  $ sagpr_prediction -r 0 -w weights_all_env -k K0_TT.npy -o prediction_all_env

The peratom scalar polarizability components are given by :code:`test_peratom.txt`, and the prediction error from an unsparsified set can be found as:

::

  $ python
  $ from ase.io import read
  $ frames = read("test.xyz",':')
  $ >>> fl = open('test_peratom.txt','w')
  $ >>> for i in xrange(len(frames)):
  $ ...     print >> fl, frames[i].info['alpha_L0'] / len(frames[i].get_chemical_symbols())
  $ ...
  $ paste prediction_all_env_L0.txt test_peratom.txt | awk 'BEGIN{m=n=0}{m+=($1-$2)**2;n++}END{print (m/n)**0.5}'

An error of 0.033 a.u./atom was found in testing this (the actual value obtained may differ very slightly).

Next, we build sparsified models. Firstly, we must find atomic power spectra and choose a number of environments. Here we try 1000, 2000 and 5000 environments.

::

  $ get_atomic_power_spectrum.py -lm 0 -p PS0_train.npy -f train.xyz -o PS0_train_atomic
  $ for env in 1000 2000 5000;do do_fps.py -p PS0_train_atomic.npy -n ${env} -o fps_${env};done
  $ for env in 1000 2000 5000;do apply_fps.py -p PS0_train_atomic.npy -sf fps_${env}_rows -o PS0_train_atomic_${env};done

Having created the sparsified power spectra, we now build the appropriate kernels.

::

  $ for env in 1000 2000 5000;do sagpr_get_kernel -z 2 -ps PS0_train.npy PS0_train_atomic_${env}.npy -s PS0_train_natoms.npy NONE -o K0_NM_${env};done
  $ for env in 1000 2000 5000;do sagpr_get_kernel -z 2 -ps PS0_train_atomic_${env}.npy -s NONE -o K0_MM_${env};done
  $ for env in 1000 2000 5000;do sagpr_get_kernel -z 2 -ps PS0_test.npy PS0_train_atomic_${env}.npy -s PS0_test_natoms.npy NONE -o K0_TT_${env};done

With these kernels, we now perform the regression. There are three possibilities, presented in order; in each case, the regression will be carried out, followed by prediction and finally the error on the testing set will be printed as a function of the number of environments.

We begin by using the :code:`solve` function to perform the regression.

::

  $ for env in 1000 2000 5000;do sagpr_train -r 0 -reg 1e-8 -f train.xyz -p alpha_L0 -sf K0_NM_${env}.npy K0_MM_${env}.npy -perat -w weights_sparse_solve_${env} -m solve;done
  $ for env in 1000 2000 5000;do sagpr_prediction -r 0 -w weights_sparse_solve_${env} -k K0_TT_${env}.npy -o prediction_sparse_solve_${env};done
  $ for env in 1000 2000 5000;do paste prediction_sparse_solve_${env}_L0.txt test_peratom.txt | awk 'BEGIN{m=n=0}{m+=($1-$2)**2;n++}END{print (m/n)**0.5}';done

It should be noted that for the 5000-environment case, sagpr_train gives a warning that the matrix to be inverted is ill-conditioned. This is reflected in the three prediction errors, 0.059 a.u./atom, 0.051 a.u./atom, 0.096 a.u./atom, the latter being much higher than expected. One way to fix this is to tune the regularization: using a value of 1e-5 rather than 1e-8 (the optimum for an unsparsified model) gives errors of 0.058 a.u./atom, 0.047 a.u./atom, 0.036 a.u./atom, with the latter being very close to the unsparsified model's prediction error.

Alternatively, rather than using the :code:`solve` function we could try using the :code:`pinv` (pseudoinverse) function:

::

  $ for env in 1000 2000 5000;do sagpr_train -r 0 -reg 1e-5 -f train.xyz -p alpha_L0 -sf K0_NM_${env}.npy K0_MM_${env}.npy -perat -w weights_sparse_pinv_${env} -m pinv;done
  $ for env in 1000 2000 5000;do sagpr_prediction -r 0 -w weights_sparse_pinv_${env} -k K0_TT_${env}.npy -o prediction_sparse_pinv_${env};done
  $ for env in 1000 2000 5000;do paste prediction_sparse_pinv_${env}_L0.txt test_peratom.txt | awk 'BEGIN{m=n=0}{m+=($1-$2)**2;n++}END{print (m/n)**0.5}';done

The :code:`pinv` function avoids ill-conditioned matrices, but it should be noted that once again the optimum regularization is different from that in the unsparsified model (once again, the errors are 0.058 a.u./atom, 0.047 a.u./atom, 0.036 a.u./atom). However, while this function is more "forgiving", and preferable to using :code:`solve` in sparsification problems, it can be much more expensive.

An alternative is to apply a "jitter" term, using the :code:`solve` function but with a diagonal matrix with small magnitude added to the matrix to be inverted, so that it is full-rank:

::

  $ for env in 1000 2000 5000;do sagpr_train -r 0 -reg 1e-5 -f train.xyz -p alpha_L0 -sf K0_NM_${env}.npy K0_MM_${env}.npy -perat -w weights_sparse_jitter_${env} -m solve -j CHOOSE;done
  $ for env in 1000 2000 5000;do sagpr_prediction -r 0 -w weights_sparse_jitter_${env} -k K0_TT_${env}.npy -o prediction_sparse_jitter_${env};done
  $ for env in 1000 2000 5000;do paste prediction_sparse_jitter_${env}_L0.txt test_peratom.txt | awk 'BEGIN{m=n=0}{m+=($1-$2)**2;n++}END{print (m/n)**0.5}';done

The option :code:`CHOOSE` means that the code will choose a magnitude for this matrix that is as small as possible while still making the resulting matrix full-rank. Alternatively, one can enter their chosen value. The :code:`CHOOSE` option can make this step quite expensive in its current form, so should be used with care. However, this method may be useful in cases where :code:`pinv` is very expensive. In this case, we obtain 0.058 a.u./atom, 0.047 a.u./atom, 0.044 a.u./atom. This latter case is a problem largely because in this case the jitter isn't really necessary. This should be treated as an experimental feature that may in future become useful.

11. Uncertainty Estimation
--------------------------

This example uses three data sets, which will be made available on publication of the relevant paper. Once they are available, they wil be found in `example/amino_acid`. We begin by calculating the power spectra for the training set, choosing an active set of environments and finding the kernels between the training set and the active set.

::

  $ cd example/amino_acid
  $ for lm in 0 1;do sagpr_parallel_get_PS -nrun 28 -lm ${lm} -f train.xyz -o PS${lm} -c H C N O S -s H C N O S -n 4 -l 2 -sg 0.23726178 -rs 1 2.91603113 6.08786224 -sm random -nc 400 -ns 1000 -rc 5.0;done
  $ for lm in 0 1;do sagpr_parallel_get_PS -nrun 28 -lm ${lm} -f train.xyz -o PS${lm}_train -c H C N O S -s H C N O S -n 4 -l 2 -sg 0.23726178 -rs 1 2.91603113 6.08786224 -sf PS${lm} -rc 5.0;done
  $ for lm in 0 1;do get_atomic_power_spectrum.py -p PS${lm}_train.npy -f train.xyz -o PS${lm}_train_atomic;done
  $ do_fps.py -p PS1_train_atomic.npy -n 8000 -v | tee fps.out
  $ for lm in 0 1;do apply_fps.py -p PS${lm}_train_atomic.npy -o PS${lm}_train_sparse;done
  $ sagpr_get_kernel -z 2 -ps PS1_train.npy PS1_train_sparse.npy -ps0 PS0_train.npy PS0_train_sparse.npy -s PS1_train_natoms.npy NONE -o K1_NM
  $ sagpr_get_kernel -z 2 -ps PS1_train_sparse.npy -ps0 PS0_train_sparse.npy -s NONE -o K1_MM

Next, we subsample (taking 5000 points per sample and 8 samples):

::

  $ mkdir np_5000
  $ cd np_5000
  $ python subsample.py -k ../K1_NM.npy -f ../train.xyz -np $(pwd | sed "s/\_/ /g" | awk '{print $NF}') -ns 8
  $ multi_train.sh 1 9.694108361689872e-06 ../K1_MM.npy

This produces 8 weight files, one for each model. The next step is to calibrate the uncertainty estimate, which we do using a validation set.

::

  $ cd ../
  $ for fl in validation test;do
  $         sagpr_parallel_get_PS -nrun 28 -lm 0 -f ${fl}.xyz -o PS0_${fl} -c H C N O S -s H C N O S -n 6 -l 4 -sg 0.23088253 -rs 1 4.15454532 8.24538508 -sf PS0 -rc 5.0
  $         sagpr_parallel_get_PS -nrun 28 -lm 2 -f ${fl}.xyz -o PS2_${fl} -c H C N O S -s H C N O S -n 4 -l 2 -sg 0.23088253 -rs 1 4.15454532 8.24538508 -sf PS2 -rc 5.0
  $         sagpr_get_kernel -z 2 -ps PS2_${fl}.npy PS2_train_sparse.npy -ps0 PS0_${fl}.npy PS0_train_sparse.npy -s PS2_${fl}_natoms.npy NONE -o K2_${fl}
  $ done
  $ cd np_5000
  $ nsample=$(ls | grep -c WEIGHTS)

Note that the first part also creates the kernels needed for the test set.

::

  $ for i in $(seq 1 ${nsample});do
  $         sagpr_prediction -r 1 -w WEIGHTS.${i} -k ../K1_validation.npy -o PREDICTION.${i} -sp
  $         cat ../validation.xyz | sed "s/\(=\|\"\)/ /g" | awk '{if (NF==1){nat=$1}}/Properties/{for (i=1;i<=NF;i++){if ($i=="mu_L1"){printf "%.16f %.16f %.16f\n", $(i+1)/nat,$(i+2)/nat,$(i+3)/nat}}}' > CALC.${i}_L1.txt
  $         paste PREDICTION.${i}_L1.txt CALC.${i}_L1.txt | awk '{print $1,$4;print $2,$5;print $3,$6}' > RESIDUAL.${i}_L1.txt
  $         echo "Predicted model number "${i}
  $ done

This is a version of the script `multi_predict.sh`, but is written out explicitly here. Having made these predictions, we now calibrate the uncertainty estimation factor alpha:

::

  $ get_alpha.sh

This creates a file, `alpha.txt`, the last line of which is the square of the factor by which the predictions of each individual model must be moved away from the average value.

Finally, we use the test set to see how good our predictions are, not only of the dipole moment but also of the uncertainty.

::

  $ cd ../;mkdir test_predictions;cd test_predictions
  $ nmodel=$(ls ../np_5000 | grep -c WEIGHTS)
  $ for i in $(seq 1 ${nmodel});do
  $         sagpr_prediction -r 1 -w ../np_5000/WEIGHTS.${i} -k ../K1_test.npy -o PREDICTION.${1}.${i} -sp
  $ done
  $ for i in PREDICTION.np_5000.*_L1.txt;do cat ${i} | awk '{print ($1**2 + $2**2 + $3**2)**0.5}' > $(echo ${i} | sed "s/L1/NORM/");done
  $ paste PREDICTION.np_5000.*_NORM.txt | awk '{n=m=0;for (i=1;i<=NF;i++){n++;m+=$i};printf "%.16f\n",m/n}' > PREDICTION_MEAN.np_5000_NORM.txt
  $ export alpha=$(tail -n 1 ../np_5000/alpha.txt)
  $ paste PREDICTION_MEAN.np_5000_NORM.txt PREDICTION.np_5000.*_NORM.txt | awk 'BEGIN{al=ENVIRON["alpha"]**0.5}{printf "%.16f ",$1;for (i=2;i<=NF;i++){dd=$1 + ($i-$1)*al;printf "%.16f ",dd};printf "\n"}' | awk '{l=m=n=0;for (i=1;i<=NF;i++){n++;m+=$i};yavg=(m/n);for (i=2;i<=NF;i++){l+=($i-yavg)**2};printf "%.16f %.16f\n",$1,(l/(n-1))**0.5}' > PREDICTION_COMMITTEE.np_5000_NORM.txt
  $ cat ../test.xyz | sed "s/\(=\|\"\)/ /g" | awk '{if (NF==1){nat=$1}}/Properties/{for (i=1;i<=NF;i++){if ($i=="mu_L1"){printf "%.16f %.16f %.16f\n", $(i+1)/nat,$(i+2)/nat,$(i+3)/nat}}}' | awk '{print ($1**2 + $2**2 + $3**2)**0.5}' > CALC_NORM.txt 
  $ paste CALC_NORM.txt PREDICTION_COMMITTEE.np_5000_NORM.txt > calc_pred_uncertainty.txt

As suggested by the name, `calc_pred_uncertainty.txt` has three columns: the calculated dipole moment norm, the dipole moment norm predicted by the eight models, and the estimated uncertainty from this committee. A good test of whether the model is accurately gauging its uncertainty is to compare the norm of the difference between the first two columns (i.e., the residual) with the uncertainty estimate. If the estimated uncertainty does not match the residual (it likely will not), then it should at least be larger than the residual in the majority of cases, meaning that the model is properly "cautious" in its estimates.

Contact
=======

d.wilkins@qub.ac.uk

andrea.grisafi@epfl.ch

Contributors
============

David Wilkins, Andrea Grisafi, Andrea Anelli, Guillaume Fraux, Jigyasa Nigam, Edoardo Baldi, Linnea Folkmann, Michele Ceriotti
