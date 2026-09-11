import torch
import torch.nn as nn


class ImageFeatureExtractor(nn.Module):
    """Encode an RGB image and concatenate it with robot-state features."""

    def __init__(self, image_shape, robot_feature_dim: int, output_dim: int):
        super().__init__()
        channels = image_shape[0]

        self.CNN_feature_extractor = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, *image_shape)
            flattened_dim = self.CNN_feature_extractor(sample).shape[1]

        self.image_linear = nn.Sequential(
            nn.Linear(flattened_dim, output_dim), nn.LayerNorm(output_dim), nn.Tanh()
        )
        self.output_dim = output_dim + robot_feature_dim

    def forward(self, obs):
        image = obs["image"]
        if image.dtype == torch.uint8:
            image = image.float().div(255.0)

        image_feature = self.image_linear(self.CNN_feature_extractor(image))
        return torch.cat((image_feature, obs["feature"]), dim=1)


class TwinQNetwork(nn.Module):
    """Estimate two Q-values to limit overestimation."""

    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        input_dim = feature_dim + action_dim

        def make_q_network():
            return nn.Sequential(
                nn.Linear(input_dim, 400),
                nn.ReLU(),
                nn.Linear(400, 300),
                nn.ReLU(),
                nn.Linear(300, 1),
            )

        self.Q1 = make_q_network()
        self.Q2 = make_q_network()

    def forward(self, features, actions):
        x = torch.cat((features, actions), dim=1)
        return self.Q1(x), self.Q2(x)


class Actor(nn.Module):
    """Map encoded features to normalized actions."""

    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh(),
        )

    def forward(self, features):
        return self.net(features)
