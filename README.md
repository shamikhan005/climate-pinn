# climatePINN
pytorch implementation of a physics-informed neural network (PINN) designed to solve fundamental partial differential equations (PDEs) that govern climate dynamics, such as heat flow and advection.

## core concept

a traditional neural network learns by minimizing its error on a large dataset. a PINN, however, learns from two sources:

1. data loss ($L_{data}$): a standard MSE loss based on a small number of known data points (e.g., initial and boundary conditions)

2. physics loss ($L_{physics}$): an MSE loss based on how well the model's output obeys the governing PDE. this loss is calculated on thousands of unlabeled "collocation" points inside the domain, forcing the model to learn underlying physics.

the total loss is: $L_{total}$ = $L_{data}$ + $L_{physics}$

## key results

### 1. 1D heat (diffusion) equation
the model successfully learned to solve the 1D heat equation, $\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$

the PINN's prediction is visually identical to the ground truth

<img width="1539" height="500" alt="1d_heat_pinn" src="https://github.com/user-attachments/assets/373cb428-bfeb-49e9-9de8-d84840aa89ac" />

### 2. 1D advection-diffusion equation
the model successfully solved the 1D advection-diffusion equation, $\frac{\partial u}{\partial t} + c \frac{\partial u}{\partial x} = \alpha \frac{\partial^2 u}{\partial x^2}$

this is more complex problem that models both the transport (advection) and spreading (diffusion) of a property, which is a core mechanism of climate and weather.

<img width="1539" height="500" alt="advection_pinn_result" src="https://github.com/user-attachments/assets/75bac84b-3f0c-4901-860b-3e03e4d1664a" />

## key findings: PINN vs traditional NN

to prove the value of PINN, conducted an experiment 

<img width="1539" height="810" alt="compare_models" src="https://github.com/user-attachments/assets/d75f4bce-b67f-42c9-ae79-2bff5e059655" />

- model 1 (PINN): trained on 150 sparse data points + 5000 physics collocation points.
- model 2 (data-only NN): trained on only the 150 sparse data points.

result: the traditional NN failed completely, only learning the boundary values. the PINN was able to reconstruct the entire continuous field with high fidelity.

**the PINN was ~94x more accurate (3.48e-04 MAE) than the traditional network (3.30e-02 MAE).**

## setup & installation

this project uses PyTorch with CUDA for GPU acceleration.

1. clone the repo
   ```
   git clone https://github.com/shamikhan005/climate-pinn.git
   cd climate-pinn
   ```
2. create conda environment
   ```
   conda create -n climate-pinn python=3.12
   conda activate climate-pinn
   ```
3. install dependencies
   option a - using ``` environment.yml``` (recommended):
   ```
   conda env update --file environment.yml --prune
   ```

   option b - manual install
   ```
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   conda install numpy matplotlib
   ```
4. verify installation
   check python & pytorch:
   ```
   python --version
   python -c ""import torch; print('CUDA available:', torch.cuda.is_available())"
   ```
5. run
   ```
   # 1d heat equation
   python heat_pinn.py

   # PINN vs data-only NN comparison
   python compare_models.py

   # 1d advection-diffusion
   python advection_pinn.py
   ```
