import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

# --- environment & physics setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# define physical constants
ALPHA = 0.1   # thermal diffusivity
L = 1.0       # length of the rod
T = 0.1       # final time

# define analytical solution (ground truth)
def analytical_solution(x, t, alpha=ALPHA):
  """calucalte the exact solution for the 1D heat equation"""
  return np.sin(np.pi * x) * np.exp(-alpha * (np.pi**2) * t)

# --- generating training data ---
print("generating training data...")

# boundary and initial condition points
N_boundary = 100

# ic points
t_ic = torch.zeros((N_boundary, 1), device=device)
x_ic = torch.linspace(0, L, N_boundary, device=device).view(-1, 1)

# get true temperatures at these ic points
u_ic = torch.tensor(analytical_solution(x_ic.cpu().numpy(), t_ic.cpu().numpy()), dtype=torch.float32, device=device)

# bc1 points
t_bc1 = torch.linspace(0, T, N_boundary, device=device).view(-1, 1)
x_bc1 = torch.zeros((N_boundary, 1), device=device)

# get true temperatures of bc1
u_bc1 = torch.zeros((N_boundary, 1), device=device)

# bc2 points
t_bc2 = torch.linspace(0, T, N_boundary, device=device).view(-1, 1)
x_bc2 = torch.full((N_boundary, 1), L, device=device)

# get true temperatures of bc2
u_bc2 = torch.zeros((N_boundary, 1), device=device)

# group boundary and ic data
x_boundary = torch.cat([x_ic, x_bc1, x_bc2])
t_boundary = torch.cat([t_ic, t_bc1, t_bc2])
u_boundary = torch.cat([u_ic, u_bc1, u_bc2])

# collections points (for checking PDE)
N_collocation = 5000

# random points inside the domain (t > 0, 0 < x < L)
t_collocation = torch.rand(N_collocation, 1, device=device) * T
x_collocation = torch.rand(N_collocation, 1, device=device) * L
t_collocation.requires_grad = True
x_collocation.requires_grad = True

print(f"generated {u_boundary.shape[0]} boundary/ic points")
print(f"generated {x_collocation.shape[0]} collocation points for physics")

# --- neural network ---
class PINN(nn.Module):
  """PINN architecture"""
  def __init__(self):
    super(PINN, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(2, 32),    # two inputs (x,t)
      nn.Tanh(),
      nn.Linear(32, 32),
      nn.Tanh(),
      nn.Linear(32, 32),
      nn.Tanh(),
      nn.Linear(32, 1)     # one output (u)
    )
  
  def forward(self, x_input, t_input):
    """concatenates x and t before passing through network"""
    model_input = torch.cat([x_input, t_input], dim=1)
    return self.net(model_input)

# initialize model
pinn_model = PINN().to(device)
print(pinn_model)

# --- loss function and training loop ---

# loss function (MSE)
data_loss_fn = nn.MSELoss()

# initialize optimizer
optimizer = torch.optim.Adam(pinn_model.parameters(), lr=1e-3)

# training loop
epochs = 20000
print(f"starting training for {epochs} epochs...")
start_time = time.time()

for epoch in range(epochs):
  pinn_model.train()

  # data loss calculation
  u_boundary_pred = pinn_model(x_boundary, t_boundary)
  loss_data = data_loss_fn(u_boundary_pred, u_boundary)

  # physics loss calculation
  u_collocation_pred = pinn_model(x_collocation, t_collocation)

  # calculate derivatives
  du_dt = torch.autograd.grad(
    u_collocation_pred, t_collocation,
    grad_outputs=torch.ones_like(u_collocation_pred),
    create_graph=True
  )[0]

  du_dx = torch.autograd.grad(
    u_collocation_pred, x_collocation,
    grad_outputs=torch.ones_like(u_collocation_pred),
    create_graph=True
  )[0]

  d2u_dx2 = torch.autograd.grad(
    du_dx, x_collocation,
    grad_outputs=torch.ones_like(du_dx),
    create_graph=True
  )[0]

  # calculate residual of PDE
  residual = du_dt - ALPHA * d2u_dx2

  # calcualte physics loss
  loss_physics = data_loss_fn(residual, torch.zeros_like(residual))

  # total loss and backpropagation
  loss = loss_data + loss_physics

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if (epoch + 1) % 1000 == 0:
    print(f"epoch [{epoch+1}/{epochs}], total loss: {loss.item():.4e}, data loss: {loss_data.item():.4e}, physics loss: {loss_physics.item():.4e}")

end_time = time.time()
print(f"training finished in {end_time - start_time:.2f} seconds")

# --- evaluation and visualization ---
print("evaluating model and plotting results...")

pinn_model.eval()

# create test grid
t_test_vals = np.linspace(0, T, 100)
x_test_vals = np.linspace(0, L, 100)
t_grid, x_grid = np.meshgrid(t_test_vals, x_test_vals)

# flatten grid and convert to tensors
x_test = torch.tensor(x_grid.flatten(), dtype=torch.float32, device=device).view(-1, 1)
t_test = torch.tensor(t_grid.flatten(), dtype=torch.float32, device=device).view(-1, 1)

# get model predictions
with torch.no_grad():
  u_pinn_pred = pinn_model(x_test, t_test).cpu().numpy().reshape(x_grid.shape)

# get analytical solution
u_analytical = analytical_solution(x_grid, t_grid)

# calculate error
error = np.abs(u_pinn_pred - u_analytical)
print(f"mean absolute error on test grid: {error.mean():.4e}")

# visualization
plt.figure(figsize=(18, 5))

# PINN prediction
plt.subplot(1, 3, 1)
plt.pcolormesh(t_grid, x_grid, u_pinn_pred, shading='auto', cmap='Reds')
plt.colorbar(label='Temperature u(x,t)')
plt.xlabel('Time (t)')
plt.ylabel('Position (x)')
plt.title('PINN Prediction')

# analytical (ground truth)
plt.subplot(1, 3, 2)
plt.pcolormesh(t_grid, x_grid, u_analytical, shading='auto', cmap='Reds')
plt.colorbar(label='Temperature u(x,t)')
plt.xlabel('Time (t)')
plt.ylabel('Position (x)')
plt.title('Analytical Solution (Ground Truth)')

# absolute error
plt.subplot(1, 3, 3)
plt.pcolormesh(t_grid, x_grid, error, shading='auto', cmap='bwr', vmin=-error.max(), vmax=error.max())
plt.colorbar(label='Absolute Error')
plt.xlabel('Time (t)')
plt.ylabel('Position (x)')
plt.title('Absolute Error |Prediction - Truth|')

plt.tight_layout()
plt.show()