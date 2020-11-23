import torch, spconv
import torch.nn.functional as F
import torch.nn as nn


class NoopLayer(nn.Module):
    def __init__(self):
        super(NoopLayer, self).__init__()

    def forward(self, x):
        return x


class ChannelNorm(nn.Module):
    def __init__(self, in_channels):
        super(ChannelNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, in_channels), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1, in_channels), requires_grad=True)

    def forward(self, x):
        is_4d_tensor = x.ndim == 4
        if is_4d_tensor:
            n, c, h, w = x.shape
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, keepdim=True)
            x = (x-mean)/(var+1e-5).sqrt()*self.weight.view(1, c, 1, 1) + self.bias.view(1, c, 1, 1)
        else:
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True)
            x = (x-mean)/(var+1e-5).sqrt()*self.weight + self.bias
        return x

# class ChannelNorm(nn.Module):
#     def __init__(self, in_channels):
#         super(ChannelNorm, self).__init__()
#
#         self.weight = nn.Parameter(torch.ones(1, in_channels), requires_grad=True)
#         self.bias = nn.Parameter(torch.zeros(1, in_channels), requires_grad=True)
#
#     def forward(self, x):
#         is_4d_tensor = x.ndim == 4
#
#         if is_4d_tensor:
#             n, c, h, w = x.shape
#             x = x.permute(0, 2, 3, 1).reshape(-1, c)
#
#         mean = x.mean(dim=-1, keepdim=True)
#         var = x.var(dim=-1, keepdim=True)
#         x = (x-mean)/(var+1e-5).sqrt()*self.weight + self.bias
#
#         if is_4d_tensor:
#             x = x.view(n, h, w, c).permute(0, 3, 1, 2).contiguous()
#
#         return x


def compute_start_points(shapes, margin):
    """
    tiling features in horizontal direction with a given margin
    shapes: list, in NCHW format
    margin: int, margin to tiling features of different levels
    """
    N, C, H0, W0 = shapes[0]
    x0 = 0
    start_points = []
    ranges = []
    for shape in shapes:
        _, _, H, W = shape
        y0 = H0 - H
        start_points.append((y0, x0))
        ranges.append((x0, y0, x0 + W, y0 + H))
        x0 += W + margin
    return start_points, ranges, (H0, x0)


def process_per_level(dense_features, mask, level, stride):
    """
    extract sparse features and indices, compute locations in image of them as well

    Arguments:
        dense_features: N*C*H*W
        mask: N*H*W*1

    Returns:
        sparse_features: m*3
        indices: m*C
        locations: m*2
    """

    # swap channels
    dense_features = dense_features.permute(0, 2, 3, 1).contiguous()

    mask = mask > 0

    indices = torch.nonzero(mask.squeeze(3))  # [[batch_idx, y_idx, x_idx]], m*3
    sparse_features = torch.masked_select(dense_features, mask).view(-1, dense_features.size(-1))
    locations = indices[:, [2, 1]].float() * stride + stride // 2

    return sparse_features, locations, indices


def dense_to_sparse(all_dense_features, masks, fpn_strides, margin):
    """
    Arguments:
        all_dense_features: level first, list of N*C*H*W tensors
        masks: list of N*H*W*1 tensors
    Returns:
        all_sp_features: sparse features for cls/loc
        all_stage1_predictions: masked features corresponding to all_sp_features
        pack_info: a dict contains:
            locations: locations in original image
            levels: level info
            ranges:
    """

    batch_size = all_dense_features[0].size(0)

    all_sp_features = []
    all_indices = []
    all_locations = []
    all_levels = []
    for level, (df, mask) in enumerate(zip(all_dense_features, masks)):
        sp_feats, loc, idx = process_per_level(
            df, mask, level, fpn_strides[level]
        )
        all_sp_features.append(sp_feats)
        all_locations.append(loc)
        all_indices.append(idx)
        all_levels.append(idx.new_ones(len(idx)) * level)

    # compute starting points and container shape when tiling all features
    start_points, ranges, container_shape = compute_start_points(
        [x.shape for x in all_dense_features], margin
    )

    # merge information from all levels
    for indices, start_point in zip(all_indices, start_points):
        y0, x0 = start_point
        indices[:, 1] += y0
        indices[:, 2] += x0
    all_indices = torch.cat(all_indices, dim=0).int()
    all_sp_features = torch.cat(all_sp_features, dim=0)
    all_locations = torch.cat(all_locations, dim=0)
    all_levels = torch.cat(all_levels, dim=0)

    all_sp_features = spconv.SparseConvTensor(
        all_sp_features, all_indices, container_shape, batch_size
    )

    pack_info = {
        'locations': all_locations,
        'levels': all_levels,
        'ranges': ranges
    }

    return all_sp_features, pack_info


def expand_mask(mask, kernel_size=3, permute=True):
    """
    mask: n*h*w*1 or n*1*h*w
    """
    if permute:
        mask = mask.permute(0, 3, 1, 2).contiguous()  # to nchw
    mask = mask.float()

    if kernel_size <= 1:
        pass
    else:
        weight = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device)/(kernel_size*kernel_size)
        mask = F.conv2d(mask, weight, padding=kernel_size//2)
    # else:
    #     weight = torch.ones((1, 1, 1, kernel_size), device=mask.device) / kernel_size
    #     mask = F.conv2d(mask, weight, padding=(0, kernel_size // 2))
    #     weight = torch.ones((1, 1, kernel_size, 1), device=mask.device) / kernel_size
    #     mask = F.conv2d(mask, weight, padding=(kernel_size // 2, 0))

    mask = mask.permute(0, 2, 3, 1) > 0.0  # to nhwc
    return mask