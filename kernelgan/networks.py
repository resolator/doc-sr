import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import scipy.io as sio

import kernelgan.loss as loss
from kernelgan.util import swap_axis, zeroize_negligible_val, kernel_shift


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        struct = [7, 5, 3, 1, 1, 1]
        channels = 64
        
        input_crop_size = 64
        scale_factor = 0.5
        
        # First layer - Converting RGB image to latent space
        self.first_layer = nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=struct[0], bias=False)

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            feature_block.append(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=struct[layer], bias=False))
        self.feature_block = nn.Sequential(*feature_block)

        # Final layer - Down-sampling and converting back to image
        stride = int(1 / scale_factor)
        self.final_layer = nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=struct[-1], stride=stride, bias=False)

        # Calculate number of pixels shaved in the forward pass
        self.output_size = self.forward(torch.FloatTensor(torch.ones([1, 1, input_crop_size, input_crop_size]))).shape[-1]
        self.forward_shave = int(input_crop_size * scale_factor) - self.output_size

    def forward(self, input_tensor):
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        input_tensor = swap_axis(input_tensor)
        downscaled = self.first_layer(input_tensor)
        features = self.feature_block(downscaled)
        output = self.final_layer(features)
        return swap_axis(output)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        channels = 64
        kernel_size = 7
        n_layers = 7
        
        input_crop_size = 64

        # First layer - Convolution (with no ReLU)
        self.first_layer = nn.utils.spectral_norm(nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, bias=True))
        feature_block = []  # Stacking layers with 1x1 kernels
        for _ in range(1, n_layers - 1):
            feature_block += [
                nn.utils.spectral_norm(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=True)),
                nn.BatchNorm2d(channels),
                nn.ReLU(True)
            ]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1, bias=True)),
            nn.Sigmoid()
        )

        # Calculate number of pixels shaved in the forward pass
        self.forward_shave = input_crop_size - self.forward(torch.FloatTensor(torch.ones([1, 3, input_crop_size, input_crop_size]))).shape[-1]

    def forward(self, input_tensor):
        receptive_extraction = self.first_layer(input_tensor)
        features = self.feature_block(receptive_extraction)
        return self.final_layer(features)


def weights_init_D(m):
    """ initialize weights of the discriminator """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
            
def weights_init_G(m):
    """ initialize weights of the generator """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
            
            
class KernelGAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.last_g_loss = None
        self.last_d_loss = None
        input_crop_size = 64
        self.G_kernel_size = 13
        self.D_kernel_size = 7
        
        self.scale_factor = 0.5
        self.n_filtering = 40
        
        # Optimizer params
        self.lr = 2e-4
        self.beta1 = 0.5
        self.step_size = 750
        
        # Variables for constraints
        self.bic_loss_counter = 0
        self.similar_to_bicubic = False  # Flag indicating when the bicubic similarity is achieved
        self.insert_constraints = True  # Flag is switched to false once constraints are added to the loss
        self.bic_loss_to_start_change = 0.4
        self.lambda_bicubic_decay_rate = 100.
        self.lambda_sparse_end = 5
        self.lambda_centralized_end = 1
        self.lambda_bicubic_min = 5e-6
        
        # Define the GAN
        self.G = Generator()
        self.D = Discriminator()

        # Calculate D's input & output shape according to the shaving done by the networks
        self.d_input_shape = self.G.output_size
        self.d_output_shape = self.d_input_shape - self.D.forward_shave

        # Input tensors
        self.g_input = torch.FloatTensor(1, 3, input_crop_size, input_crop_size).cuda()
        self.d_input = torch.FloatTensor(1, 3, self.d_input_shape, self.d_input_shape).cuda()

        # The kernel G is imitating
        self.kernel = torch.FloatTensor(self.G_kernel_size, self.G_kernel_size).cuda()
        
        self.lambda_sum2one = 0.5
        self.lambda_boundaries = 0.5
        self.lambda_sparse = 5
        self.lambda_center = 1
        
        # Losses
        self.GAN_loss_layer = loss.GANLoss(d_last_layer_size=self.d_output_shape)
        self.bicubic_loss = loss.DownScaleLoss(scale_factor=self.scale_factor)
        self.loss_bicubic = 0

        # Define loss function
        self.criterionGAN = self.GAN_loss_layer.forward

        # Initialize networks weights
        self.G.apply(weights_init_G)
        self.D.apply(weights_init_D)

    # noinspection PyUnboundLocalVariable
    def calc_kernel(self):
        """given a generator network, the function calculates the kernel it is imitating"""
        delta = torch.Tensor([1.]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
        for ind, w in enumerate(self.G.parameters()):
            kernel = F.conv2d(delta, w, padding=self.G_kernel_size - 1) if ind == 0 else F.conv2d(kernel, w)
        return kernel.squeeze().flip([0, 1])
        
    def training_step(self, batch, batch_idx, optimizer_idx):
        g_input, d_input = batch
        g_input, d_input = g_input[0].contiguous(), d_input[0].contiguous()

        # train generator
        loss = None
        self.step = batch_idx
        if optimizer_idx == 0:
            g_pred = self.G.forward(g_input)  # Generator forward pass
            d_pred_fake = self.D.forward(g_pred)  # Pass Generators output through Discriminator
            loss_g = self.criterionGAN(d_last_layer=d_pred_fake, is_d_input_real=True)  # Calculate generator loss
            loss = loss_g + self.calc_constraints(g_pred)  # Sum all losses
            self.log('train/generator_loss', loss)
            self.last_g_loss = loss.item()

        # train discriminator
        if optimizer_idx == 1:
            d_pred_real = self.D.forward(d_input)  # Discriminator forward pass over real example
            # Discriminator forward pass over fake example (generated by generator)
            # Note that generator result is detached so that gradients are not propagating back through generator
            g_output = self.G.forward(g_input)
            d_pred_fake = self.D.forward((g_output + torch.randn_like(g_output) / 255.).detach())
            
            # Calculate discriminator loss
            loss_d_fake = self.criterionGAN(d_pred_fake, is_d_input_real=False)
            loss_d_real = self.criterionGAN(d_pred_real, is_d_input_real=True)
            loss = (loss_d_fake + loss_d_real) * 0.5
            self.log('train/discriminator_loss', loss)
            self.last_d_loss = loss.item()

        if self.last_d_loss is not None and self.last_g_loss is not None:
            self.log('train_loss', (self.last_g_loss + self.last_d_loss) / 2)
            self.last_g_loss = None
            self.last_d_loss = None

        return loss
    
    def on_train_batch_end(self, epoch_output, batch_end_outputs, batch, batch_idx):
        if batch_idx == 0:
            return None

        # Until similar to bicubic is satisfied, don't update any other lambdas
        if not self.similar_to_bicubic:
            if self.loss_bicubic < self.bic_loss_to_start_change:
                if self.bic_loss_counter >= 2:
                    self.similar_to_bicubic = True
                else:
                    self.bic_loss_counter += 1
            else:
                self.bic_loss_counter = 0
        
        # Once similar to bicubic is satisfied, consider inserting other constraints
        elif batch_idx % self.lambda_update_freq == 0 and self.lambda_bicubic > self.lambda_bicubic_min:
            self.lambda_bicubic = max(self.lambda_bicubic / self.lambda_bicubic_decay_rate, self.lambda_bicubic_min)
            if self.insert_constraints and self.lambda_bicubic < 5e-3:
                self.lambda_centralized = self.lambda_centralized_end
                self.lambda_sparse = self.lambda_sparse_end
                
                self.insert_constraints = False

    def calc_constraints(self, g_pred):
        self.kernel = self.calc_kernel()  # Calculate K which is equivalent to G
        
        loss_sum_to_one = loss.loss_sum_to_one(self.kernel)
        loss_boundaries = loss.loss_boundaries(self.kernel)
        loss_center = loss.loss_center(self.kernel, self.scale_factor)
        loss_sparse = loss.loss_sparse(self.kernel)
        
        self.loss_bicubic = self.bicubic_loss.forward(g_input=self.g_input, g_output=g_pred)  # Calculate constraints
        
        # Apply constraints co-efficients
        return loss_sum_to_one * self.lambda_sum2one + \
               loss_boundaries * self.lambda_boundaries + \
               loss_center * self.lambda_center + \
               loss_sparse * self.lambda_sparse
    
    def post_process_kernel(self):
        k = self.kernel.detach().cpu().float().numpy()
        significant_k = zeroize_negligible_val(k, self.n_filtering)  # Zeroize negligible values
        centralized_k = kernel_shift(significant_k, sf=int(1 / self.scale_factor))  # Force centralization on the kernel
        # return shave_a2b(centralized_k, k)
        self.kernel = centralized_k
    
    def save_kernel(self, kernel_path):
        sio.savemat(kernel_path.absolute(), {'Kernel': self.kernel})
        
    def configure_optimizers(self):
        opt_G = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        opt_D = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        
        # sched_G = torch.optim.lr_scheduler.StepLR(opt_G, self.step_size, gamma=0.1)
        # sched_D = torch.optim.lr_scheduler.StepLR(opt_D, self.step_size, gamma=0.1)
        # return [opt_G, opt_D], [sched_G, sched_D]
        return [opt_G, opt_D]
