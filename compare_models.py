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
ALPHA = 0.1  # thermal diffusivity
L = 1.0  # length of the rod
T = 0.1  # final time
EPOCHS = 20000


# analytical solution (ground truth)
def analytical_solution(x, t, alpha=ALPHA):
    return np.sin(np.pi * x) * np.exp(-alpha * (np.pi**2) * t)


# --- neural network architecture ---
# same architecture for both models for a fair comparison
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x_input, t_input):
        model_input = torch.cat([x_input, t_input], dim=1)
        return self.net(model_input)


# --- generate data ---
# two sets of data:
# 1. sparse set of "real" data (for both models)
# 2. large set of collocation points (for the PINN only)

# --- sparse "real" data (for data Loss) ---
N_SPARSE = 50
print(f"generating {N_SPARSE * 3} total sparse data points...")

# sparse ic points (t=0)
t_ic_sparse = torch.zeros((N_SPARSE, 1), device=device)
x_ic_sparse = torch.linspace(0, L, N_SPARSE, device=device).view(-1, 1)
u_ic_sparse = torch.tensor(
    analytical_solution(x_ic_sparse.cpu().numpy(), t_ic_sparse.cpu().numpy()),
    dtype=torch.float32,
    device=device,
)

# sparse bc1 points (x=0)
t_bc1_sparse = torch.linspace(0, T, N_SPARSE, device=device).view(-1, 1)
x_bc1_sparse = torch.zeros((N_SPARSE, 1), device=device)
u_bc1_sparse = torch.zeros((N_SPARSE, 1), device=device)

# sparse bc2 points (x=L)
t_bc2_sparse = torch.linspace(0, T, N_SPARSE, device=device).view(-1, 1)
x_bc2_sparse = torch.full((N_SPARSE, 1), L, device=device)
u_bc2_sparse = torch.zeros((N_SPARSE, 1), device=device)

# group all sparse data
x_sparse_data = torch.cat([x_ic_sparse, x_bc1_sparse, x_bc2_sparse])
t_sparse_data = torch.cat([t_ic_sparse, t_bc1_sparse, t_bc2_sparse])
u_sparse_data = torch.cat([u_ic_sparse, u_bc1_sparse, u_bc2_sparse])

# --- collocation points (for physics loss, PINN only) ---
N_collocation = 5000
print(f"generating {N_collocation} collocation points for PINN...")

t_collocation = torch.rand(N_collocation, 1, device=device) * T
x_collocation = torch.rand(N_collocation, 1, device=device) * L
t_collocation.requires_grad = True
x_collocation.requires_grad = True

# --- loss function (used by both) ---
data_loss_fn = nn.MSELoss()

# pinn model training
print("\n--- training model 1: PINN ---")
pinn_model = Net().to(device)
optimizer_pinn = torch.optim.Adam(pinn_model.parameters(), lr=1e-3)
start_time_pinn = time.time()

for epoch in range(EPOCHS):
    pinn_model.train()

    # data Loss (on sparse data)
    u_sparse_pred = pinn_model(x_sparse_data, t_sparse_data)
    loss_data = data_loss_fn(u_sparse_pred, u_sparse_data)

    # physics Loss (on collocation points)
    u_collocation_pred = pinn_model(x_collocation, t_collocation)

    du_dt = torch.autograd.grad(
        u_collocation_pred,
        t_collocation,
        torch.ones_like(u_collocation_pred),
        create_graph=True,
    )[0]
    du_dx = torch.autograd.grad(
        u_collocation_pred,
        x_collocation,
        torch.ones_like(u_collocation_pred),
        create_graph=True,
    )[0]
    d2u_dx2 = torch.autograd.grad(
        du_dx, x_collocation, torch.ones_like(du_dx), create_graph=True
    )[0]

    residual = du_dt - ALPHA * d2u_dx2
    loss_physics = data_loss_fn(residual, torch.zeros_like(residual))

    # total Loss
    loss = loss_data + loss_physics

    optimizer_pinn.zero_grad()
    loss.backward()
    optimizer_pinn.step()

    if (epoch + 1) % 2000 == 0:
        print(
            f"PINN epoch [{epoch + 1}/{EPOCHS}], total loss: {loss.item():.4e}, data: {
                loss_data.item():.4e}, physics: {loss_physics.item():.4e}"
        )

end_time_pinn = time.time()
print(f"PINN training finished in {
      end_time_pinn - start_time_pinn:.2f} seconds")

print("\n--- training model 2: data-only NN ---")
nn_model = Net().to(device)  # fresh model, same architecture
optimizer_nn = torch.optim.Adam(nn_model.parameters(), lr=1e-3)
start_time_nn = time.time()

for epoch in range(EPOCHS):
    nn_model.train()

    # data Loss (on sparse data)
    u_sparse_pred = nn_model(x_sparse_data, t_sparse_data)
    loss = data_loss_fn(u_sparse_pred, u_sparse_data)

    # no physics loss

    optimizer_nn.zero_grad()
    loss.backward()
    optimizer_nn.step()

    if (epoch + 1) % 2000 == 0:
        print(f"NN epoch [{epoch + 1}/{EPOCHS}], data Loss: {loss.item():.4e}")

end_time_nn = time.time()
print(
    f"data-only NN training finished in {end_time_nn - start_time_nn:.2f} seconds")


# evaluation and comparison
print("\n--- Evaluating Models ---")

# create a full test grid
t_test_vals = np.linspace(0, T, 100)
x_test_vals = np.linspace(0, L, 100)
t_grid, x_grid = np.meshgrid(t_test_vals, x_test_vals)

x_test = torch.tensor(x_grid.flatten(), dtype=torch.float32,
                      device=device).view(-1, 1)
t_test = torch.tensor(t_grid.flatten(), dtype=torch.float32,
                      device=device).view(-1, 1)

# get predictions from both models
pinn_model.eval()
nn_model.eval()
with torch.no_grad():
    u_pinn_pred = pinn_model(
        x_test, t_test).cpu().numpy().reshape(x_grid.shape)
    u_nn_pred = nn_model(x_test, t_test).cpu().numpy().reshape(x_grid.shape)

# get analytical solution
u_analytical = analytical_solution(x_grid, t_grid)

# calculate errors
error_pinn = np.abs(u_pinn_pred - u_analytical)
error_nn = np.abs(u_nn_pred - u_analytical)

print(f"PINN mean absolute error on test grid:     {error_pinn.mean():.4e}")
print(f"data-only NN mean absolute error on test grid: {error_nn.mean():.4e}")

# --- visualization ---
plt.figure(figsize=(18, 10))

# --- PINN ---
plt.subplot(2, 3, 1)
plt.pcolormesh(t_grid, x_grid, u_pinn_pred,
               shading="auto", cmap="Reds", vmin=0, vmax=1)
plt.colorbar(label="Temp u(x,t)")
plt.xlabel("Time (t)")
plt.ylabel("Position (x)")
plt.title("PINN Prediction (from sparse data)")

plt.subplot(2, 3, 2)
plt.pcolormesh(
    t_grid, x_grid, u_analytical, shading="auto", cmap="Reds", vmin=0, vmax=1
)
plt.colorbar(label="Temp u(x,t)")
plt.xlabel("Time (t)")
plt.ylabel("Position (x)")
plt.title("Analytical Solution (Ground Truth)")

plt.subplot(2, 3, 3)
plt.pcolormesh(
    t_grid, x_grid, error_pinn, shading="auto", cmap="bwr", vmin=-0.1, vmax=0.1
)
plt.colorbar(label="Absolute Error")
plt.xlabel("Time (t)")
plt.ylabel("Position (x)")
plt.title("PINN Absolute Error")

# --- data-only NN ---
plt.subplot(2, 3, 4)
plt.pcolormesh(t_grid, x_grid, u_nn_pred, shading="auto",
               cmap="Reds", vmin=0, vmax=1)
plt.colorbar(label="Temp u(x,t)")
plt.xlabel("Time (t)")
plt.ylabel("Position (x)")
plt.title("Data-Only NN Prediction (from sparse data)")

# plot the sparse data points on top
plt.scatter(
    t_sparse_data.cpu().numpy(),
    x_sparse_data.cpu().numpy(),
    s=5,
    c="blue",
    marker="x",
    label=f"{N_SPARSE * 3} data points",
)
plt.legend()

plt.subplot(2, 3, 5)
plt.pcolormesh(
    t_grid, x_grid, u_analytical, shading="auto", cmap="Reds", vmin=0, vmax=1
)
plt.colorbar(label="Temp u(x,t)")
plt.xlabel("Time (t)")
plt.ylabel("Position (x)")
plt.title("Analytical Solution (Ground Truth)")

plt.subplot(2, 3, 6)
plt.pcolormesh(
    t_grid, x_grid, error_nn, shading="auto", cmap="bwr", vmin=-0.1, vmax=0.1
)
plt.colorbar(label="Absolute Error")
plt.xlabel("Time (t)")
plt.ylabel("Position (x)")
plt.title("Data-Only NN Absolute Error")

plt.tight_layout()
plt.show()

