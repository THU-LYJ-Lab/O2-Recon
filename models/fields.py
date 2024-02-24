import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder, positional_encoding_c2f

class Cascaded_SDFNetwork(nn.Module):
    def __init__(self,
                 stage,
                 d_in_1,
                 d_out_1,
                 d_hidden_1,
                 n_layers_1,
                 skip_in_1,
                 multires_1,
                 d_in_2,
                 d_out_2,
                 d_hidden_2,
                 n_layers_2,
                 skip_in_2,
                 multires_2,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 ):
        super(Cascaded_SDFNetwork, self).__init__()
        self.stage = stage

        self.coarse_sdf = SDFNetwork(d_in_1, d_out_1, d_hidden_1, n_layers_1, skip_in_1,
                                     multires_1, bias, scale, geometric_init, weight_norm)
        
        self.fine_sdf = Refine_SDFNetwork(d_in_2, d_out_2, d_out_1 - 1, d_hidden_2, n_layers_2, skip_in_2,
                                        multires_2, bias, scale, weight_norm)
        
        self.weigth_emb_c2f = self.coarse_sdf.weigth_emb_c2f
        
        
    def forward(self, inputs):
        coarse_output = self.coarse_sdf(inputs)
        if self.stage == 1:
            return coarse_output
        
        sdf_coarse = coarse_output[:, :1]
        feat_coarse = coarse_output[:, 1:]

        fine_output = self.fine_sdf(inputs, feat_coarse)
        sdf_fine = fine_output[:, :1]
        feat_fine = fine_output[:, 1:]
        sdf = sdf_fine + sdf_coarse
        feat = feat_fine + feat_coarse

        return torch.cat([sdf, feat], dim=-1)  
    
    def freeze_param(self):
        for param in self.coarse_sdf.parameters():
            param.requires_grad = False
        
    def sdf(self, x):
        return self.forward(x)[:, :1]

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

class Refine_SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_feat,
                 d_hidden,
                 n_layers,
                 skip_in,
                 multires=0,
                 bias=0.5,
                 scale=1,
                 weight_norm=True):
        super(Refine_SDFNetwork, self).__init__()

        dims = [d_feat] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.embed_fn_fine = None
        self.skip_in = skip_in

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] += input_ch

        self.num_layers = len(dims)
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - input_ch
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)
            if l == self.num_layers - 2:
                nn.init.constant_(lin.bias, 0)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        
        self.activation = nn.Softplus(beta=100)
    
    def forward(self, inputs, feat):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = torch.cat([inputs, feat], 1)
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)


class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 activation='softplus',
                 reverse_geoinit = False,
                 use_emb_c2f = False,
                 emb_c2f_start = 0.1,
                 emb_c2f_end = 0.5):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        # import pdb; pdb.set_trace()
        self.embed_fn_fine = None

        self.multires = multires
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in, normalize=False)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch
        logging.info(f'SDF input dimension: {dims[0]}')

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale
        self.use_emb_c2f = use_emb_c2f
        if self.use_emb_c2f:
            self.emb_c2f_start = emb_c2f_start
            self.emb_c2f_end = emb_c2f_end
            logging.info(f"Use coarse-to-fine embedding (Level: {self.multires}): [{self.emb_c2f_start}, {self.emb_c2f_end}]")

        self.alpha_ratio = 0.0

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if reverse_geoinit:
                        logging.info(f"Geometry init: Indoor scene (reverse geometric init).")
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                    else:
                        logging.info(f"Geometry init: DTU scene (not reverse geometric init).")
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        else:
            assert activation == 'relu'
            self.activation = nn.ReLU()

        self.weigth_emb_c2f = None
        self.iter_step = 0
        self.end_iter = 3e5

    def forward(self, inputs):
        inputs = inputs * self.scale

        if self.use_emb_c2f and self.multires > 0:
            inputs, weigth_emb_c2f = positional_encoding_c2f(inputs, self.multires, emb_c2f=[self.emb_c2f_start, self.emb_c2f_end], alpha_ratio = (self.iter_step / self.end_iter))
            self.weigth_emb_c2f = weigth_emb_c2f
        elif self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)
        else:
            NotImplementedError

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

class FixVarianceNetwork(nn.Module):
    def __init__(self, base):
        super(FixVarianceNetwork, self).__init__()
        self.base = base
        self.iter_step = 0

    def set_iter_step(self, iter_step):
        self.iter_step = iter_step

    def forward(self, x):
        return torch.ones([len(x), 1]) * np.exp(-self.iter_step / self.base)

class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val=1.0, use_fixed_variance = False):
        super(SingleVarianceNetwork, self).__init__()
        if use_fixed_variance:
            logging.info(f'Use fixed variance: {init_val}')
            self.variance = torch.tensor([init_val])
        else:
            self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)

class RenderingNetwork(nn.Module):
    def __init__(
            self,
            d_feature,
            mode,
            d_in,
            d_out,
            d_hidden,
            n_layers,
            weight_norm=True,
            multires_view=0,
            squeeze_out=True,
            stage=2
    ):
        super().__init__()
        
        self.stage = stage
        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embed_ch = (input_ch - 3) // 2
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)
            if self.stage == 1:
                view_dirs[-self.embed_ch:] = 0

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


# Code from nerf-pytorch
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, d_in=3, d_in_view=3, multires=0, multires_view=0, output_ch=4, skips=[4], use_viewdirs=False):
        """
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in, normalize=False)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view, normalize=False)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha + 1.0, rgb
        else:
            assert False
