import numpy as np


class GeometryChangeMetric:

    def __init__(self):
        super(GeometryChangeMetric, self).__init__()

    def store_baseline_grid(self, density_grid):
        self.baseline_grid = density_grid.clone().cpu().detach().numpy().squeeze()

    def compute(self, density_grid):
        density_grid = density_grid.clone().cpu().detach().numpy().squeeze()
        return ((density_grid - self.baseline_grid)**2).mean()
        #return torch.nn.functional.mse_loss(density_grid, self.baseline_grid)
        #return np.dot(self.baseline_grid, density_grid) / (
        #            np.linalg.norm(self.baseline_grid) * np.linalg.norm(density_grid))