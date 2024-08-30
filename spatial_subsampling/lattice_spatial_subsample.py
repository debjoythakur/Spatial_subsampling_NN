#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import torch
from scipy.spatial import distance_matrix
from scipy.linalg import cholesky

def simulate_lattice_and_generate_data(lambda_n, d, R_0, sigma2, phi, nu, f, p):
    """
    Simulates lattice locations, generates spatially correlated covariates using Matern variogram,
    and computes the response variable Y based on the provided function f.
    
    Parameters:
    - lambda_n: Scaling factor for the region.
    - d: Dimension of the space.
    - R_0: Initial region array (e.g., [-0.5, 0.5] * d).
    - sigma2: Variance parameter of the Matern variogram.
    - phi: Range parameter of the Matern variogram.
    - nu: Smoothness parameter of the Matern variogram.
    - f: Function to compute the response variable Y from covariates X.
    - p: Number of covariates.

    Returns:
    - X: Generated covariates with spatial correlation.
    - Y: Response variable calculated using function f.
    - lattice_locations: Simulated lattice locations.
    """
    
    # Step 1: Simulate Lattice Locations
    R_n = lambda_n * R_0               # Scale the region
    eta_n = 1 / (lambda_n + d)         # Grid spacing
    
    # Generate grid points
    grid_points = np.meshgrid(*[np.arange(R_n[i, 0], R_n[i, 1], eta_n) for i in range(d)])
    lattice_locations = np.vstack(map(np.ravel, grid_points)).T
    
    # Step 2: Generate Covariate Data
    def matern_covariance(h, sigma2, phi, nu):
        from scipy.special import kv, gamma
        if h == 0:
            return sigma2
        else:
            factor = (2 ** (1 - nu)) / gamma(nu)
            arg = np.sqrt(2 * nu) * (h / phi)
            return sigma2 * factor * (arg ** nu) * kv(nu, arg)
    
    N = lattice_locations.shape[0]
    dist_matrix = distance_matrix(lattice_locations, lattice_locations)
    
    # Generate the covariance matrix using the Matern variogram
    covariance_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            covariance_matrix[i, j] = matern_covariance(dist_matrix[i, j], sigma2, phi, nu)
    
    # Cholesky decomposition for generating correlated random variables
    L = cholesky(covariance_matrix + 1e-6 * np.eye(N), lower=True)
    
    # Generate p spatially correlated fields
    X = np.dot(L, np.random.randn(N, p))
    
    # Step 3: Calculate Response Variable Y using f(X)
    Y = f(X)
    
    return X, Y, lattice_locations

# Example usage
lambda_n = 5  # Example value
d = 2         # 2D space
R_0 = np.array([[-0.5, 0.5]] * d)
sigma2 = 1.0  # Variance parameter of the Matern variogram
phi = 0.2     # Range parameter of the Matern variogram
nu = 0.5      # Smoothness parameter of the Matern variogram
p = 10        # Number of covariates

# Define the function f(X) = sum of weighted covariates
f = lambda X: np.sum(X * np.arange(1, p + 1), axis=1)

X, Y, lattice_locations = simulate_lattice_and_generate_data(lambda_n, d, R_0, sigma2, phi, nu, f, p)


# In[4]:


import time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
from scipy.spatial import distance_matrix

def lattice_spatial_nn(lattice_locations, delta, psi, Y, X, p, epoch_num=5, act_fun=torch.tanh, beta=0.3):
    """
    Function to analyze spatial data using a GNN with specific parameters.
    
    Inputs:
    - lattice_locations: The locations of the lattice points.
    - delta: Parameter to define the radii.
    - psi: Neighborhood size scaling parameter.
    - Y: Response variable.
    - X: Covariates matrix.
    - p: Number of covariates.
    - epoch_num: Number of epochs for training the GNN (default=5).
    - act_fun: Activation function (default=torch.tanh).
    - beta: Parameter to define hidden channels size scaling (default=0.3).

    Outputs:
    - N_n: Number of lattice points.
    - runtime: Time taken to execute the function.
    - optimum_radius: Radius with the minimum MSE.
    - average_predictions: Average predictions up to the stopping radius.
    - mse_subsample: Mean squared error for the subsample up to the stopping radius.
    """
    
    class GAT(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, heads=4, activation=F.tanh):
            super(GAT, self).__init__()
            self.activation = activation
            self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.3)
            self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=0.3)
            self.conv3 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        
        def forward(self, x, edge_index):
            x = self.activation(self.conv1(x, edge_index))
            x = self.activation(self.conv2(x, edge_index))
            x = self.conv3(x, edge_index)
            return x.mean(dim=0)  # Assuming we're predicting a single value per graph

    def define_neighborhoods_with_multiple_radii(lattice_locations, radii, psi):
        dist_matrix = distance_matrix(lattice_locations, lattice_locations)
        N = lattice_locations.shape[0]
        neighborhood_sizes = [int(N**psi) for _ in radii]  # Apply n^{\psi} constraint

        all_neighborhoods = []
        for idx, radius in enumerate(radii):
            neighborhoods = []
            for i in range(dist_matrix.shape[0]):
                neighbors = np.where(dist_matrix[i] <= radius)[0]
                # Ensure neighborhood size is n^{\psi}
                if len(neighbors) > neighborhood_sizes[idx]:
                    neighbors = np.random.choice(neighbors, neighborhood_sizes[idx], replace=False)
                neighborhoods.append(neighbors)
            all_neighborhoods.append(neighborhoods)

        return all_neighborhoods

    def create_subgraphs_for_multiple_radii(X, Y, neighborhoods):
        graphs = []
        for i, neighbors in enumerate(neighborhoods):
            subgraph_indices = np.array(neighbors)
            if len(subgraph_indices) == 0:
                continue  # Skip if there are no neighbors

            # Ensure the central node is included in the subgraph
            if i not in subgraph_indices:
                subgraph_indices = np.append(subgraph_indices, i)

            subgraph_indices = np.sort(subgraph_indices)  # Sort for consistency
            edge_index = np.array([[i, j] for j in subgraph_indices if i != j]).T

            # Re-index the edge_index to be relative to the subgraph
            edge_index = torch.tensor([np.searchsorted(subgraph_indices, edge_index[0]),
                                       np.searchsorted(subgraph_indices, edge_index[1])], dtype=torch.long)

            # Create node features and labels
            x = torch.tensor(np.hstack([Y[neighbors].reshape(-1, 1), X[neighbors]]), dtype=torch.float)
            y = torch.tensor([Y[i]], dtype=torch.float)

            graphs.append(Data(x=x, edge_index=edge_index, y=y))
        return graphs

    def train_gat(graphs, beta, epoch_num):
        N = len(graphs)
        hidden_channels = int(N**beta)  # Apply n^{\beta} constraint

        model = GAT(in_channels=(p + 1), hidden_channels=hidden_channels, out_channels=1, heads=4, activation=act_fun)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = torch.nn.MSELoss()

        full_loader = DataLoader(graphs, batch_size=1, shuffle=False)

        best_val_loss = float('inf')
        best_model = None
        no_improvement_epochs = 0

        for epoch in range(epoch_num):
            model.train()
            for data in full_loader:
                optimizer.zero_grad()
                output = model(data.x, data.edge_index).mean()
                loss = criterion(output, data.y)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0
            predictions = []
            with torch.no_grad():
                for data in full_loader:
                    output = model(data.x, data.edge_index).mean()
                    val_loss += criterion(output, data.y).item()
                    predictions.append(output.item())
            val_loss /= len(full_loader)

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
                no_improvement_epochs = 0  # Reset counter
            else:
                no_improvement_epochs += 1

            if no_improvement_epochs >= 2:  # Early stopping if no improvement after 2 epochs
                break

        model.load_state_dict(best_model)

        return model, best_val_loss, predictions

    radii = [1 / (lambda_n + d) + 1 / ((delta - i) * (lambda_n + d)) for i in range(delta)]

    initial_avg_neighborhood_size = None
    optimum_radius = None
    minimum_mse = float('inf')
    stopping_radius = None

    mse_results = {}
    all_predictions = {}
    average_neighborhood_sizes = {}
    neighborhood_size_mse = {}

    start_time = time.time()

    for radius in radii:
        neighborhoods = define_neighborhoods_with_multiple_radii(lattice_locations, [radius], psi)[0]
        subgraphs = create_subgraphs_for_multiple_radii(X, Y, neighborhoods)

        avg_neighborhood_size = np.mean([len(neighbors) for neighbors in neighborhoods])

        if initial_avg_neighborhood_size is None:
            initial_avg_neighborhood_size = avg_neighborhood_size

        if avg_neighborhood_size > 3 * initial_avg_neighborhood_size:
            stopping_radius = radius
            break

        trained_model, overall_mse, predictions = train_gat(subgraphs, beta, epoch_num)

        if avg_neighborhood_size not in neighborhood_size_mse:
            neighborhood_size_mse[avg_neighborhood_size] = []

        neighborhood_size_mse[avg_neighborhood_size].append(overall_mse)

        if overall_mse < minimum_mse:
            minimum_mse = overall_mse
            optimum_radius = radius

        mse_results[radius] = overall_mse
        all_predictions[radius] = predictions
        average_neighborhood_sizes[radius] = avg_neighborhood_size

    runtime = time.time() - start_time

    if stopping_radius is None:
        stopping_radius = radii[-1]  # In case stopping condition is never met

    valid_radii = [r for r in radii if r in all_predictions]

    if valid_radii:
        predictions_array = np.array([all_predictions[r] for r in valid_radii if r <= stopping_radius])

        average_predictions = np.mean(predictions_array, axis=0)

        mse_subsample = np.mean((average_predictions - Y) ** 2)

    N_n = lattice_locations.shape[0]

    return N_n, runtime, optimum_radius, average_predictions, mse_subsample


# In[5]:


N_n, runtime, optimum_radius, average_predictions, mse_subsample = lattice_spatial_nn(lattice_locations, delta = 5
                                                                                       , psi = 0.7, Y = Y, X = X, p=10)


# In[3]:


import time
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
def spatial_lattice_iid_neural_network(df, delta_n, activation_function='tanh'):
    start_time = time.time()  # Start the timer
    
    df = df.iloc[:, 1:]  # Assuming the first column needs to be excluded

    neighbor_feature = []
    response = []
    models = []
    predictions = []
    mses = []
    
    for i in range(len(df)):
        temp = pd.DataFrame(columns=df.columns)
        
        for j in range(len(df)):
            if abs(df.iloc[j, 0] - df.iloc[i, 0]) < delta_n and abs(df.iloc[j, 1] - df.iloc[i, 1]) < delta_n:
                temp = pd.concat([temp, df.iloc[[j], :]], ignore_index=True)
        
        if not temp.empty:
            neighbor_feat = temp.iloc[:, 2:].values  # Excluding the first two columns
            resp = df.iloc[i, 2:].values  # Response variable
            
            # Ensure the neighbor features and response variables have the correct shapes
            if neighbor_feat.shape[0] != resp.shape[0]:
                neighbor_feat = neighbor_feat.T
            
            neighbor_feature.append(neighbor_feat)
            response.append(resp)
            
            # Splitting data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(neighbor_feat, resp, test_size=0.2, random_state=42)
            
            # Ensuring the response variable has the correct shape for a single output node
            y_train = np.array(y_train).reshape(-1, 1)
            y_test = np.array(y_test).reshape(-1, 1)
            
            # Defining and fitting a neural network model
            model = Sequential()
            model.add(Dense(15, input_dim=X_train.shape[1], activation=activation_function))  # Use specified activation function
            model.add(Dense(15, activation=activation_function))  # Use specified activation function
            model.add(Dense(1))  # Output layer with 1 node

            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=100, verbose=0)  # Fit the model

            models.append(model)
            
            # Making predictions on the test set
            pred = model.predict(neighbor_feat)
            predictions.append(pred)
            
            # Calculating MSE
            mse = mean_squared_error(resp, pred)
            mses.append(mse)
    
    average_mse = np.mean(mses)
    runtime = time.time() - start_time  # Calculate runtime
    
    return average_mse, predictions, runtime

# Example usage:
import pandas as pd
df = pd.read_csv("~/R/nearest_neughbor_sieve/df20.csv")
delta_n = 0.8
mse, preds, run_time = spatial_lattice_iid_neural_network(df, delta_n, activation_function='tanh')
print(f"Average MSE: {mse}")
print(f"Runtime: {run_time} seconds")

