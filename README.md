## Free-breathing-liver-Kooshball-Reconstruction
Source code and representative sample data for respiratory-resolved 4D reconstruction of free-breathing 3D multi-echo GSI radial liver MRI for PDFF, R2* and QSM mapping.
This repository provides the source code and representative sample data for free-breathing, isotropic-resolution liver PDFF, R2*, and QSM mapping using 3D golden-step interleaved (GSI) multi-echo radial MRI with respiratory-resolved 4D reconstruction in MATLAB.
The repository is intended to support methodological transparency and reproducibility by providing the core reconstruction workflow, documentation, and a representative example dataset for testing and demonstration.

The sample liver multi-echo kooshball data can be downloaded from: 

## Overview

The shared demo reconstructs three echoes from a free-breathing 3D kooshball / GSI radial acquisition:
- Echo 1: center-out half-echo readout
- Echo 2: full-echo readout
- Echo 3: full-echo readout

The code performs:
1. loading of representative raw data and reconstruction parameters,
2. coil sensitivity estimation using BART ESPIRiT,
3. respiratory-resolved 4D data sorting,
4. initial NUFFT reconstruction,
5. low-rank / XD-GRASP-style iterative reconstruction with temporal and spatial regularization,
6. display of reconstructed images for all three echoes.

This repository is for research use only. It is not intended for clinical use or diagnostic decision-making.

---

## Repository contents

### Main script
- `Mian_Reconstruction.m`  
  Entry-point demo script. Loads the sample data, estimates coil sensitivity maps using BART, runs the reconstruction for all three echoes, and displays representative reconstructed slices.

### Core reconstruction functions
- `LiverRecon_4D.m`  
  Main reconstruction function. It performs:
  - SI-based respiratory sorting,
  - additional cardiac sorting if cardiac signal is provided,
  - NUFFT operator construction,
  - adjoint NUFFT reconstruction,
  - low-rank temporal basis estimation,
  - iterative low-rank 4D reconstruction with temporal/spatial regularization.

- `CSL1NlCg_LowRank4D.m`  
  Iterative nonlinear conjugate-gradient solver used for low-rank 4D reconstruction. The objective contains:
  - data consistency term,
  - respiratory temporal regularization,
  - optional cardiac temporal regularization,
  - spatial total variation regularization.

- `GeneDygpu_NUFFTOperator4D.m`  
  Builds the gpuNUFFT operators for each respiratory/cardiac state.

- `ForNUFFT_GPU5D.m`  
  Forward NUFFT operator wrapper for 5D data.

- `AdjNUFFT_GPU5D.m`  
  Adjoint NUFFT operator wrapper for 5D data.

### Data
Representative sample data are expected in a `Data/` directory:

- `Data/ReconParam.mat`  
  Reconstruction parameters and auxiliary variables used by the workflow, including respiratory signal and other reconstruction-related settings.

- `Data/FirstEcho.mat`  
  First-echo multi-coil k-space data, trajectory, and density compensation weights.

- `Data/Second&ThirdEcho.mat`  
  Second- and third-echo multi-coil k-space data, trajectories, and density compensation weights.

---

## Sample data description

The representative sample data were acquired from a healthy volunteer on a 3T MRI system during free breathing using a 3D GSI non-Cartesian radial acquisition.

The released sample dataset includes:
- trajectory-corrected spokes,
- gradient-delay-corrected spokes,
- 3D trajectories,
- density compensation weights,
- and reconstruction-related parameters required to run the demo.

The current sample dataset is intended to reproduce the demonstration workflow in the manuscript rather than to serve as a complete population dataset.

---

## Requirements

### Software
- MATLAB
- CUDA-compatible GPU
- gpuNUFFT
- BART (Berkeley Advanced Reconstruction Toolbox)
