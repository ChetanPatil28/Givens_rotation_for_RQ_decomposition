import torch
torch.set_printoptions(precision=6)
from typing import Tuple


# Input shape should be (B, 3, 4)
def decompose_projection_mat(P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    epsilon = torch.tensor([0.0], device=P.device, dtype=P.dtype)

    submat_3x3 = P[:, 0:3, 0:3]
    last_column = P[:, 0:3, 3].unsqueeze(-1)
    t = torch.matmul(-torch.inverse(submat_3x3), last_column).squeeze(-1)

    identity = torch.eye(3).unsqueeze(0)
    Q = identity.repeat(P.shape[0], 1, 1)
    K = submat_3x3.clone()

    Qx = torch.eye(3).unsqueeze(0).repeat(P.shape[0], 1, 1)
    Qy = torch.eye(3).unsqueeze(0).repeat(P.shape[0], 1, 1)
    Qz = torch.eye(3).unsqueeze(0).repeat(P.shape[0], 1, 1)

    # set K[2, 1]  to 0.
    cond1 = (submat_3x3[:, 2, 1] == 0)
    c_x = -K[:, 2, 2].clone()
    s_x = K[:, 2, 1].clone()
    l_x = torch.sqrt(torch.pow(c_x, 2) + torch.pow(s_x, 2) + epsilon)

    c_x /= l_x
    s_x /= l_x
    c_x[cond1] = 1
    s_x[cond1] = 0

    Qx[:, 1, 1] = c_x
    Qx[:, 1, 2] = -s_x
    Qx[:, 2, 1] = s_x
    Qx[:, 2, 2] = c_x

    K = torch.bmm(K, Qx)
    Q = torch.bmm(Qx.transpose(1, 2), Q)


    # set K[2, 0]  to 0.
    cond2 = K[:, 2, 0] == 0
    c_y = K[:, 2, 2].clone()
    s_y = K[:, 2, 0].clone()
    l_y = torch.sqrt(torch.pow(c_y, 2) + torch.pow(s_y, 2) + epsilon)

    c_y /= l_y
    s_y /= l_y
    c_y[cond2] = 1
    s_y[cond2] = 0

    Qy[:, 0, 0] = c_y
    Qy[:, 0, 2] = s_y
    Qy[:, 2, 0] = -s_y
    Qy[:, 2, 2] = c_y

    K = torch.bmm(K, Qy)
    Q = torch.bmm(Qy.transpose(1, 2), Q)

    cond3 = K[:, 1, 0] == 0
    c_z = -K[:, 1, 1].clone()
    s_z = K[:, 1, 0].clone()
    l_z = torch.sqrt(torch.pow(c_z, 2) + torch.pow(s_z, 2) + epsilon)

    c_z /= l_z
    s_z /= l_z
    c_z[cond3] = 1 
    s_z[cond3] = 0

    Qz[:, 0, 0] = c_z
    Qz[:, 0, 1] = -s_z
    Qz[:, 1, 0] = s_z
    Qz[:, 1, 1] = c_z

    K = torch.bmm(K, Qz)
    Q = torch.bmm(Qz.transpose(1, 2), Q)
    R = Q.clone()

    # Ensure that the diagonals are postive.
    cond4 = (K[:, 2, 2] < 0)
    K[cond4] *= -1
    R[cond4] *= -1


    S1 = torch.tensor([[
        [1., 0., 0.],
        [0., 1., 0. ],
        [0., 0., 1.]
        ]], device=P.device,dtype=P.dtype)

    cond5 = torch.where((K[:, 1, 1] <0 ), torch.tensor([-1.]), torch.tensor([1.]))
    S1[:,1, 1] = cond5

    K = torch.bmm(K, S1)
    R = torch.bmm(S1, R)

    S2 = torch.tensor([[
        [1., 0., 0.],
        [0., 1., 0. ],
        [0., 0., 1.]
        ]], device=P.device,dtype=P.dtype)

    cond6 = torch.where((K[:, 0, 0] < 0), torch.tensor([-1.]), torch.tensor([1.]))

    S2[:, 0, 0] = cond6 

    K = torch.bmm(K, S2)
    R = torch.bmm(S2, R)

    return K, R, t


