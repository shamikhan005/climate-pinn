import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

# --- environment setup and physics setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

torch.manual_seed(42)
np.random.seed(42)

# define physical constants
ALPHA = 0.01  # thermal diffusivity (lowered to see advection better)
L = 1.0       # length of the rod
T = 0.5       # final time (increased to see the wave move)
C = 1.0       # advection velocity

# define analytical solution
def analytical_solution(x, t, alpha=ALPHA, c=C, L=L):
  """calculates an exact solution for 1D advection-diffusion."""
  term1 = np.exp(-(alpha * (np.pi**2) * t) / (L**2) - (c * (x - c * t)) / (2 * alpha))
  term2 = np.sinh((c * L) / (2 * alpha)) / np.sinh((c * x) / (2 * alpha) + (c * L * (L - x + c * t)) / (2 * alpha * L))
  k = 2 * np.pi / L
  return np.exp(-alpha * (k**2) * t) * np.sin(k * (x - c * t))

# --- generate training data ---
print("generating training data...")

N_boundary = 100

# ic points 
t_ic = torch.zeros((N_boundary, 1), device=device)
x_ic = torch.linspace(0, L, N_boundary, device=device).view(-1, 1)
u_ic = torch.tensor(analytical_solution(x_ic.cpu().numpy(), t_ic.cpu().numpy()), dtype=torch.float32, device=device)

# bc points
N_bc = 100
t_bc = torch.linspace(0, T, N_bc, device=device).view(-1, 1)
x_bc_left = torch.zeros((N_bc, 1), device=device)
x_bc_right = torch.full((N_bc, 1), L, device=device)

t_bc.requires_grad = True
x_bc_left.requires_grad = True
x_bc_right.requires_grad = True

# collocation points
N_collocation = 5000
t_collocation = torch.rand(N_collocation, 1, device=device) * T
x_collocation = torch.rand(N_collocation, 1, device=device) * L
t_collocation.requires_grad = True
x_collocation.requires_grad = True

print(f"generated {x_ic.shape[0]} ic points")
print(f"generated {t_bc.shape[0]} bc points (for periodic)")
print(f"generated {x_collocation.shape[0]} collocation points")

# --- neural network ---
class PINN(nn.Module):
  def __init__(self):
    super(PINN, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(2, 32), nn.Tanh(),
      nn.Linear(32, 32), nn.Tanh(),
      nn.Linear(32, 32), nn.Tanh(),
      nn.Linear(32, 1)
    )
  def forward(self, x_input, t_input):
    model_input = torch.cat([x_input, t_input], dim=1)
    return self.net(model_input)

pinn_model = PINN().to(device)

# --- loss function and training loop ---
data_loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(pinn_model.parameters(), lr=1e-3)

# training loop
epochs = 25000
print(f"starting training for {epochs} epochs...")
start_time = time.time()

for epoch in range(epochs):
  pinn_model.train()

  # data loss calculation
  u_ic_pred = pinn_model(x_ic, t_ic)
  loss_ic = data_loss_fn(u_ic_pred, u_ic)

  # boundary loss
  u_bc_left_pred = pinn_model(x_bc_left, t_bc)
  u_bc_right_pred = pinn_model(x_bc_right, t_bc)
  loss_bc_val = data_loss_fn(u_bc_left_pred, u_bc_right_pred)

  du_dx_left = torch.autograd.grad(u_bc_left_pred, x_bc_left, torch.ones_like(u_bc_left_pred), create_graph=True)[0]
  du_dx_right = torch.autograd.grad(u_bc_right_pred, x_bc_right, torch.ones_like(u_bc_right_pred), create_graph=True)[0]
  loss_bc_grad = data_loss_fn(du_dx_left, du_dx_right)

  loss_data = loss_ic + loss_bc_val + loss_bc_grad

  # physics loss calculation
  u_collocation_pred = pinn_model(x_collocation, t_collocation)

  du_dt = torch.autograd.grad(u_collocation_pred, t_collocation, torch.ones_like(u_collocation_pred), create_graph=True)[0]
  du_dx = torch.autograd.grad(u_collocation_pred, x_collocation, torch.ones_like(u_collocation_pred), create_graph=True)[0]
  d2u_dx2 = torch.autograd.grad(du_dx, x_collocation, torch.ones_like(du_dx), create_graph=True)[0]

  residual = du_dt + C * du_dx - ALPHA * d2u_dx2

  loss_physics = data_loss_fn(residual, torch.zeros_like(residual))

  # total loss and backpropagation
  loss = loss_data + loss_physics

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if (epoch + 1) % 1000 == 0:
    print(f"epoch [{epoch+1}/{epochs}], total loss: {loss.item():.4e}, data: {loss_data.item():.4e}, physics: {loss_physics.item():.4e}")

end_time = time.time()
print(f"training finished in {end_time - start_time:.2f} seconds")

# --- evaluation and visualization ---
print("evaluating model and plotting results...")
pinn_model.eval()

t_test_vals = np.linspace(0, T, 100)
x_test_vals = np.linspace(0, L, 100)
t_grid, x_grid = np.meshgrid(t_test_vals, x_test_vals)

x_test = torch.tensor(x_grid.flatten(), dtype=torch.float32, device=device).view(-1, 1)
t_test = torch.tensor(t_grid.flatten(), dtype=torch.float32, device=device).view(-1, 1)

with torch.no_grad():
  u_pinn_pred = pinn_model(x_test, t_test).cpu().numpy().reshape(x_grid.shape)

# get analytical solution
u_analytical = analytical_solution(x_grid, t_grid)

# calculate error
error = np.abs(u_pinn_pred - u_analytical)
print(f"mean absolute error on test grid: {error.mean():.4e}")

# --- visualization ---
plt.figure(figsize=(18, 5))
v_min = u_analytical.min()
v_max = u_analytical.max()

plt.subplot(1, 3, 1)
plt.pcolormesh(t_grid, x_grid, u_pinn_pred, shading='auto', cmap='seismic', vmin=v_min, vmax=v_max)
plt.colorbar(label='Value u(x,t)')
plt.xlabel('Time (t)')
plt.ylabel('Position (x)')
plt.title('PINN Prediction (Advection-Diffusion)')

plt.subplot(1, 3, 2)
plt.pcolormesh(t_grid, x_grid, u_analytical, shading='auto', cmap='seismic', vmin=v_min, vmax=v_max)
plt.colorbar(label='Value u(x,t)')
plt.xlabel('Time (t)')
plt.ylabel('Position (x)')
plt.title('Analytical Solution (Ground Truth)')

plt.subplot(1, 3, 3)
plt.pcolormesh(t_grid, x_grid, error, shading='auto', cmap='bwr')
plt.colorbar(label='Absolute Error')
plt.xlabel('Time (t)')
plt.ylabel('Position (x)')
plt.title('Absolute Error |Prediction - Truth|')

plt.tight_layout()
plt.show()
