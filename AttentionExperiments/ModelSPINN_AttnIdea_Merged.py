import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# --------------------- Helpers --------------------- #
def get_logger(log_path):
    import logging
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
        fh = logging.FileHandler(log_path, mode='a')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    return logger


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0


def eval_metrix(pred_label, true_label):
    mae = np.mean(np.abs(pred_label - true_label))
    mape = np.mean(np.abs((true_label - pred_label) / (true_label + 1e-9))) * 100
    mse = np.mean((pred_label - true_label) ** 2)
    rmse = np.sqrt(mse)
    return [mae, mape, mse, rmse]


# --------------------- Basic Modules --------------------- #
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class Subnet(nn.Module):
    """
    Basic feedforward subnet used for each individual feature.
    """

    def __init__(self, output_dim, layers_num, hidden_dim, dropout, activation):
        super(Subnet, self).__init__()
        input_dim = 1
        layers = []
        for i in range(layers_num):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
                if activation == "sin":
                    layers.append(Sin())
                else:
                    layers.append(nn.LeakyReLU(0.01))
            elif i == layers_num - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if activation == "sin":
                    layers.append(Sin())
                else:
                    layers.append(nn.LeakyReLU(0.01))
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self._init_weights()
        self.output_dim = output_dim

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


# --------------------- Solution_u Module (Global Self-Attention) --------------------- #
class Solution_u_spinn_Attn(nn.Module):
    """
    Processes each input feature with a dedicated subnet, then uses global multi-head self-attention
    to combine the tokens.
    """

    def __init__(self, input_dim, subnet_args, attn_embed_dim=16, n_heads=1):
        super(Solution_u_spinn_Attn, self).__init__()
        self.input_dim = input_dim
        # Remove extra keys that are not for Subnet
        subnet_args = dict(subnet_args)
        subnet_args.pop("attn_embed_dim", None)
        subnet_args.pop("attn_heads", None)
        self.subnets = nn.ModuleList([Subnet(**subnet_args) for _ in range(input_dim)])
        out_dim = self.subnets[0].output_dim
        self.proj = nn.Linear(out_dim, attn_embed_dim) if out_dim != attn_embed_dim else None
        self.self_attn = nn.MultiheadAttention(embed_dim=attn_embed_dim, num_heads=n_heads, batch_first=True)
        self.final_fc = nn.Linear(attn_embed_dim, 1)
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, x):
        B = x.size(0)
        tokens = []
        for i, net in enumerate(self.subnets):
            token = net(x[:, i].unsqueeze(1))  # shape (B, out_dim)
            tokens.append(token)
        tokens = torch.stack(tokens, dim=1)  # (B, input_dim, out_dim)
        if self.proj is not None:
            tokens = self.proj(tokens)  # (B, input_dim, attn_embed_dim)
        attn_out, _ = self.self_attn(tokens, tokens, tokens)  # (B, input_dim, attn_embed_dim)
        pooled = attn_out.mean(dim=1)  # (B, attn_embed_dim)
        out = self.final_fc(pooled)  # (B,1)
        return self.activation(out)


# --------------------- Merged Dynamical_F Module --------------------- #
class Dynamical_F_Attn_Cross_Merged(nn.Module):
    """
    Processes the PDE input (e.g. [xt, u, u_x, u_t]) via per-dimension subnets.
    Then splits the resulting tokens into two groups (group A and group B).
    Within each group a scaled dot-product self-attention is applied,
    and additionally cross-attention is computed between the two groups.
    The outputs are combined (here, averaged within each group and then concatenated)
    and passed through a final linear layer.

    Parameters:
      - total_dim: total number of tokens (should equal the input feature dimension)
      - groupA_size: number of tokens in group A (if None, defaults to total_dim//2)
      - subnet_args: parameters for each subnet (do not include attn keys)
      - attn_embed_dim: embedding dimension for attention (all tokens are projected to this)
      - n_heads: number of attention heads
    """

    def __init__(self, total_dim, subnet_args, attn_embed_dim=16, n_heads=1, groupA_size=None):
        super(Dynamical_F_Attn_Cross_Merged, self).__init__()
        self.total_dim = total_dim  # e.g., x_dim*2 + 1
        if groupA_size is None:
            self.groupA_size = total_dim // 2
        else:
            self.groupA_size = groupA_size
        self.groupB_size = total_dim - self.groupA_size
        # Create one subnet per token
        subnet_args = dict(subnet_args)
        subnet_args.pop("attn_embed_dim", None)
        subnet_args.pop("attn_heads", None)
        self.subnets = nn.ModuleList([Subnet(**subnet_args) for _ in range(total_dim)])
        out_dim = self.subnets[0].output_dim
        # If necessary, project tokens to attn_embed_dim
        self.proj = nn.Linear(out_dim, attn_embed_dim) if out_dim != attn_embed_dim else None
        # Self-attention for group A and group B (separate modules)
        self.self_attn_A = nn.MultiheadAttention(embed_dim=attn_embed_dim, num_heads=n_heads, batch_first=True)
        self.self_attn_B = nn.MultiheadAttention(embed_dim=attn_embed_dim, num_heads=n_heads, batch_first=True)
        # Cross-attention modules: A queries from B and B queries from A
        self.cross_attn_A = nn.MultiheadAttention(embed_dim=attn_embed_dim, num_heads=n_heads, batch_first=True)
        self.cross_attn_B = nn.MultiheadAttention(embed_dim=attn_embed_dim, num_heads=n_heads, batch_first=True)
        # Final combination layer
        self.final_fc = nn.Linear(attn_embed_dim * 2, 1)
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, x):
        # x shape: (B, total_dim)
        B = x.size(0)
        tokens = []
        for i, net in enumerate(self.subnets):
            token = net(x[:, i].unsqueeze(1))  # (B, out_dim)
            tokens.append(token)
        tokens = torch.stack(tokens, dim=1)  # (B, total_dim, out_dim)
        if self.proj is not None:
            tokens = self.proj(tokens)  # (B, total_dim, attn_embed_dim)
        # Split tokens into two groups:
        A_tokens = tokens[:, :self.groupA_size, :]  # (B, groupA_size, attn_embed_dim)
        B_tokens = tokens[:, self.groupA_size:, :]  # (B, groupB_size, attn_embed_dim)
        # Apply self-attention within each group:
        A_self, _ = self.self_attn_A(A_tokens, A_tokens, A_tokens)
        B_self, _ = self.self_attn_B(B_tokens, B_tokens, B_tokens)
        # Apply cross-attention: A queries from B, and B queries from A:
        A_cross, _ = self.cross_attn_A(A_tokens, B_tokens, B_tokens)
        B_cross, _ = self.cross_attn_B(B_tokens, A_tokens, A_tokens)
        # Combine self and cross outputs (here we average each group’s outputs):
        A_comb = (A_self + A_cross) / 2  # (B, groupA_size, attn_embed_dim)
        B_comb = (B_self + B_cross) / 2  # (B, groupB_size, attn_embed_dim)
        # Pool each group (mean over tokens):
        A_pool = A_comb.mean(dim=1)  # (B, attn_embed_dim)
        B_pool = B_comb.mean(dim=1)  # (B, attn_embed_dim)
        # Concatenate pooled representations and map to a scalar:
        combined = torch.cat([A_pool, B_pool], dim=1)  # (B, 2*attn_embed_dim)
        out = self.final_fc(combined)  # (B,1)
        return self.activation(out)


# --------------------- LR Scheduler --------------------- #
class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch=1,
                 constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
                    1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        if self.iter >= len(self.lr_schedule):
            lr = self.lr_schedule[-1]
        else:
            lr = self.lr_schedule[self.iter]
        for param_group in self.optimizer.param_groups:
            if self.constant_predictor_lr and param_group.get('name', None) == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                param_group['lr'] = lr
        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr


# --------------------- Final Merged SPINN Model --------------------- #
class SPINN_Attn_Merged(nn.Module):
    """
    Final model combining:
      - solution_u: global self-attention over per-feature subnets (as in Idea 1)
      - dynamical_F: merged module that uses per-feature subnets and then applies both
        scaled dot-product (self) attention within groups and cross-attention between groups
        on the tokens. The grouping is controlled via groupA_size.
    """

    def __init__(self, args, x_dim, architecture_args):
        super(SPINN_Attn_Merged, self).__init__()
        if args.save_folder and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        log_dir = args.log_dir if not args.save_folder else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)
        # Set up solution_u (as in Idea 1)
        attn_embed_dim_u = architecture_args.get("attn_embed_dim_u", 16)
        attn_heads_u = architecture_args.get("attn_heads_u", 1)
        self.solution_u = Solution_u_spinn_Attn(
            input_dim=x_dim,
            subnet_args=architecture_args["solution_u_subnet_args"],
            attn_embed_dim=attn_embed_dim_u,
            n_heads=attn_heads_u
        ).to(device)
        # Set up dynamical_F: total_dim = x_dim*2 + 1 as before.
        total_dim = x_dim * 2 + 1
        attn_embed_dim_F = architecture_args.get("attn_embed_dim_F", 16)
        attn_heads_F = architecture_args.get("attn_heads_F", 1)
        # For grouping, let groupA_size be provided; if not, default to half of total_dim.
        groupA_size = architecture_args.get("dynamical_F_groupA_size", total_dim // 2)
        self.dynamical_F = Dynamical_F_Attn_Cross_Merged(
            total_dim=total_dim,
            subnet_args=architecture_args["dynamical_F_subnet_args"],
            attn_embed_dim=attn_embed_dim_F,
            n_heads=attn_heads_F,
            groupA_size=groupA_size
        ).to(device)
        # Optimizers and scheduler (same as before)
        self.optimizer1 = torch.optim.Adam(self.solution_u.parameters(), lr=args.warmup_lr)
        self.optimizer2 = torch.optim.Adam(self.dynamical_F.parameters(), lr=args.lr_F)
        self.scheduler = LR_Scheduler(self.optimizer1, args.warmup_epochs, args.warmup_lr, args.epochs, args.lr,
                                      args.final_lr)
        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()
        self.alpha = args.alpha
        self.beta = args.beta
        self.args = args
        self.best_model = None
        # History tracking
        self.train_data_loss_history = []
        self.train_pde_loss_history = []
        self.train_phys_loss_history = []
        self.valid_mse_history = []
        self.test_mse_history = []

    def clear_logger(self):
        self.logger.handlers.clear()

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.solution_u.load_state_dict(checkpoint['solution_u'])
        self.dynamical_F.load_state_dict(checkpoint['dynamical_F'])
        for param in self.solution_u.parameters():
            param.requires_grad = True

    def predict(self, xt):
        return self.solution_u(xt)

    def Test(self, testloader):
        self.eval()
        true_labels = []
        pred_labels = []
        with torch.no_grad():
            for x1, _, y1, _ in testloader:
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_labels.append(y1.numpy())
                pred_labels.append(u1.cpu().numpy())
        true_labels = np.concatenate(true_labels, axis=0)
        pred_labels = np.concatenate(pred_labels, axis=0)
        return true_labels, pred_labels

    def TestMSE(self, testloader):
        self.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x1, _, y1, _ in testloader:
                x1 = x1.to(device)
                out = self.predict(x1)
                preds.append(out.cpu().numpy())
                trues.append(y1.numpy())
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        return np.mean((preds - trues) ** 2)

    def ValidMSE(self, validloader):
        self.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x1, _, y1, _ in validloader:
                x1 = x1.to(device)
                out = self.predict(x1)
                preds.append(out.cpu().numpy())
                trues.append(y1.numpy())
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        return np.mean((preds - trues) ** 2)

    def forward(self, xt):
        xt.requires_grad = True
        x = xt[:, 0:-1]
        t = xt[:, -1:].clone()
        u = self.solution_u(torch.cat([x, t], dim=1))
        u_t = grad(u.sum(), t, create_graph=True)[0]
        u_x = grad(u.sum(), x, create_graph=True)[0]
        F_in = torch.cat([xt, u, u_x, u_t], dim=1)
        F = self.dynamical_F(F_in)
        f = u_t - F
        return u, f

    def train_one_epoch(self, epoch, dataloader):
        self.train()
        data_meter = AverageMeter()
        pde_meter = AverageMeter()
        phys_meter = AverageMeter()
        for it, (x1, x2, y1, y2) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            x1, x2 = x1.to(device), x2.to(device)
            y1, y2 = y1.to(device), y2.to(device)
            u1, f1 = self.forward(x1)
            u2, f2 = self.forward(x2)
            loss_data = 0.5 * self.loss_func(u1, y1) + 0.5 * self.loss_func(u2, y2)
            zero_target = torch.zeros_like(f1)
            loss_pde = 0.5 * self.loss_func(f1, zero_target) + 0.5 * self.loss_func(f2, zero_target)
            loss_phys = self.relu((u2 - u1) * (y1 - y2)).sum()
            loss = loss_data + self.alpha * loss_pde + self.beta * loss_phys
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()
            data_meter.update(loss_data.item())
            pde_meter.update(loss_pde.item())
            phys_meter.update(loss_phys.item())
        return data_meter.avg, pde_meter.avg, phys_meter.avg

    def Train(self, trainloader, testloader=None, validloader=None):
        min_valid_mse = 1e30
        early_stop_counter = 0
        for e in range(1, self.args.epochs + 1):
            data_loss, pde_loss, phys_loss = self.train_one_epoch(e, trainloader)
            current_lr = self.scheduler.step()
            self.train_data_loss_history.append(data_loss)
            self.train_pde_loss_history.append(pde_loss)
            self.train_phys_loss_history.append(phys_loss)
            val_mse = self.ValidMSE(validloader) if validloader is not None else 0.0
            self.valid_mse_history.append(val_mse)
            test_mse = self.TestMSE(testloader) if testloader is not None else 0.0
            self.test_mse_history.append(test_mse)
            info = (f"[Epoch {e}] lr={current_lr:.6f}, data={data_loss:.6f}, pde={pde_loss:.6f}, "
                    f"phys={phys_loss:.6f}, val_mse={val_mse:.6f}, test_mse={test_mse:.6f}")
            self.logger.info(info)
            if validloader is not None and val_mse < min_valid_mse:
                min_valid_mse = val_mse
                early_stop_counter = 0
                true_label, pred_label = self.Test(testloader)
                metrics = eval_metrix(pred_label, true_label)
                self.logger.info(f"[Test Metrics] MAE={metrics[0]:.6f}, MAPE={metrics[1]:.6f}, "
                                 f"MSE={metrics[2]:.6f}, RMSE={metrics[3]:.6f}")
                self.best_model = {'solution_u': self.solution_u.state_dict(),
                                   'dynamical_F': self.dynamical_F.state_dict()}
                if self.args.save_folder:
                    np.save(os.path.join(self.args.save_folder, 'true_label.npy'), true_label)
                    np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), pred_label)
            else:
                early_stop_counter += 1
            if self.args.early_stop and early_stop_counter > self.args.early_stop:
                self.logger.info(f"Early stopping at epoch {e}")
                break
        if self.best_model and self.args.save_folder:
            torch.save(self.best_model, os.path.join(self.args.save_folder, 'model.pth'))
        # Plotting training losses
        epochs_range = range(1, len(self.train_data_loss_history) + 1)
        plt.figure()
        plt.plot(epochs_range, self.train_data_loss_history, label='Train-DataLoss')
        plt.plot(epochs_range, self.train_pde_loss_history, label='Train-PDELoss')
        plt.plot(epochs_range, self.train_phys_loss_history, label='Train-PhysLoss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Losses (Merged)')
        if self.args.save_folder:
            plt.savefig(os.path.join(self.args.save_folder, 'training_losses.png'))
            plt.close()
        else:
            plt.show()
        plt.figure()
        plt.plot(epochs_range, self.valid_mse_history, label='Valid MSE')
        plt.plot(epochs_range, self.test_mse_history, label='Test MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.title('Validation & Test MSE (Merged)')
        if self.args.save_folder:
            plt.savefig(os.path.join(self.args.save_folder, 'valid_test_mse.png'))
            plt.close()
        else:
            plt.show()
        self.clear_logger()
