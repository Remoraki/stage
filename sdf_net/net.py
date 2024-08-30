import torch
import torch.nn as nn


# The structure proposed in DeepSDF paper
class SDFNet(nn.Module):
    def __init__(self, dropout_prob=0.2, inDim=2):
        super(SDFNet, self).__init__()
        self.fc_stack_1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(inDim, 512)),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.utils.weight_norm(nn.Linear(512, 512 - inDim)),  # 510 = 512 - 2
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        self.fc_stack_2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.utils.weight_norm(nn.Linear(512, 1))
        )
        self.th = nn.Tanh()

    def forward(self, x):
        skip_out = self.fc_stack_1(x)
        skip_in = torch.cat([skip_out, x], 1)
        y = self.fc_stack_2(skip_in)
        out = self.th(y)
        return out

    def gradient(self, points: torch.Tensor) -> torch.Tensor:
        """
        SDF gradient
        :param points: points tensor, size (n, 2)
        :return: gradient tensor, size (n, 2)
        """
        points.requires_grad_(True)
        y = self.forward(points)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        grad = torch.autograd.grad(outputs=y, inputs=points, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True)[0]
        return grad.unsqueeze(1)
