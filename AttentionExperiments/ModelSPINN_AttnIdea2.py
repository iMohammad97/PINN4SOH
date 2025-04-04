# File: ModelSPINN_AttnIdea2_Full.py
import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- Helpers (same as in Idea 1) ---- #
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
        return self.sum/self.count if self.count > 0 else 0

def eval_metrix(pred_label, true_label):
    mae = np.mean(np.abs(pred_label - true_label))
    mape = np.mean(np.abs((true_label - pred_label)/(true_label+1e-9)))*100
    mse = np.mean((pred_label - true_label)**2)
    rmse = np.sqrt(mse)
    return [mae, mape, mse, rmse]

# ---- Subnet with Internal Attention ---- #
class SubnetWithInternalAttention(nn.Module):
    """
    Expands a single scalar into multiple tokens, applies self-attention among them,
    pools, then passes through an MLP.
    """
    def __init__(self, output_dim=16, hidden_dim=32, num_tokens=4, attn_embed_dim=16, n_heads=1, dropout=0.0, activation="leaky-relu"):
        super(SubnetWithInternalAttention, self).__init__()
        self.num_tokens = num_tokens
        self.attn_embed_dim = attn_embed_dim
        self.initial_fc = nn.Linear(1, num_tokens * attn_embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=attn_embed_dim, num_heads=n_heads, batch_first=True)
        self.act = nn.LeakyReLU(0.01) if activation=="leaky-relu" else torch.sin
        self.post_fc = nn.Sequential(
            nn.Linear(attn_embed_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.output_dim = output_dim
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        B = x.size(0)
        embed = self.initial_fc(x)  # (B, num_tokens*attn_embed_dim)
        tokens = embed.view(B, self.num_tokens, self.attn_embed_dim)  # (B, num_tokens, attn_embed_dim)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        pooled = attn_out.mean(dim=1)  # (B, attn_embed_dim)
        out = self.post_fc(pooled)     # (B, output_dim)
        return out

# ---- Global Aggregator Using Internal Attention Subnets ---- #
class Solution_u_spinn_WithSubnetAttention(nn.Module):
    def __init__(self, input_dim, subnet_args):
        super(Solution_u_spinn_WithSubnetAttention, self).__init__()
        self.input_dim = input_dim
        # Here, use SubnetWithInternalAttention – note: do not pass a "layers_num" key; use "num_tokens" instead.
        self.subnets = nn.ModuleList()
        for _ in range(input_dim):
            self.subnets.append(SubnetWithInternalAttention(**subnet_args))
        out_dim = self.subnets[0].output_dim
        self.final_fc = nn.Linear(out_dim, 1)
        self.act = nn.LeakyReLU(0.01)
    def forward(self, x):
        B = x.size(0)
        outputs = []
        for i in range(self.input_dim):
            scalar_input = x[:, i].unsqueeze(1)  # (B,1)
            out_i = self.subnets[i](scalar_input)  # (B, output_dim)
            outputs.append(out_i)
        stacked = torch.stack(outputs, dim=1)  # (B, input_dim, output_dim)
        reduced = stacked.sum(dim=1)           # (B, output_dim)
        out = self.final_fc(reduced)           # (B,1)
        return self.act(out)

class Dynamical_F_spinn_WithSubnetAttention(nn.Module):
    def __init__(self, x_dim, subnet_args):
        super(Dynamical_F_spinn_WithSubnetAttention, self).__init__()
        self.input_dim = x_dim * 2 + 1
        self.subnets = nn.ModuleList()
        for _ in range(self.input_dim):
            self.subnets.append(SubnetWithInternalAttention(**subnet_args))
        out_dim = self.subnets[0].output_dim
        self.final_fc = nn.Linear(out_dim, 1)
        self.act = nn.LeakyReLU(0.01)
    def forward(self, x):
        outputs = []
        for i, net in enumerate(self.subnets):
            scalar_input = x[:, i].unsqueeze(1)
            outputs.append(net(scalar_input))
        stacked = torch.stack(outputs, dim=1)
        reduced = stacked.sum(dim=1)
        out = self.final_fc(reduced)
        return self.act(out)

# ---- LR Scheduler (same as before) ---- #
class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch=1, constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))
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

# ---- Full SPINN Model (Idea 2) with Metrics and Plotting ---- #
class SPINN_AttnIdea2(nn.Module):
    def __init__(self, args, x_dim, architecture_args):
        super(SPINN_AttnIdea2, self).__init__()
        if args.save_folder and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        log_dir = args.log_dir if not args.save_folder else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)
        self.solution_u = Solution_u_spinn_WithSubnetAttention(
            input_dim=x_dim,
            subnet_args=architecture_args["solution_u_subnet_args"]
        ).to(device)
        self.dynamical_F = Dynamical_F_spinn_WithSubnetAttention(
            x_dim=x_dim,
            subnet_args=architecture_args["dynamical_F_subnet_args"]
        ).to(device)
        self.optimizer1 = torch.optim.Adam(self.solution_u.parameters(), lr=args.warmup_lr)
        self.optimizer2 = torch.optim.Adam(self.dynamical_F.parameters(), lr=args.lr_F)
        self.scheduler = LR_Scheduler(self.optimizer1, args.warmup_epochs, args.warmup_lr, args.epochs, args.lr, args.final_lr)
        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()
        self.alpha = args.alpha
        self.beta = args.beta
        self.args = args
        self.best_model = None
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
                self.logger.info(f"[Test Metrics] MAE={metrics[0]:.6f}, MAPE={metrics[1]:.6f}, MSE={metrics[2]:.6f}, RMSE={metrics[3]:.6f}")
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
        # Plotting
        epochs_range = range(1, len(self.train_data_loss_history) + 1)
        plt.figure()
        plt.plot(epochs_range, self.train_data_loss_history, label='Train-DataLoss')
        plt.plot(epochs_range, self.train_pde_loss_history, label='Train-PDELoss')
        plt.plot(epochs_range, self.train_phys_loss_history, label='Train-PhysLoss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Losses (Idea 2)')
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
        plt.title('Validation & Test MSE (Idea 2)')
        if self.args.save_folder:
            plt.savefig(os.path.join(self.args.save_folder, 'valid_test_mse.png'))
            plt.close()
        else:
            plt.show()
        self.clear_logger()
