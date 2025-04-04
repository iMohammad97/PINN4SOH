# File: ModelSPINN_AttnIdea3_Full.py
import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- Helpers ---- #
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
        self.sum = 0.
        self.count = 0
    def update(self, val):
        self.sum += val
        self.count += 1
    @property
    def avg(self):
        return self.sum/self.count if self.count>0 else 0

def eval_metrix(pred_label, true_label):
    mae = np.mean(np.abs(pred_label-true_label))
    mape = np.mean(np.abs((true_label-pred_label)/(true_label+1e-9)))*100
    mse = np.mean((pred_label-true_label)**2)
    rmse = np.sqrt(mse)
    return [mae, mape, mse, rmse]

# ---- Simple MLP for solution_u ---- #
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, layers_num=3, hidden_dim=32, activation='leaky-relu'):
        super(SimpleMLP, self).__init__()
        mods = []
        in_dim = input_dim
        for i in range(layers_num):
            out_dim = hidden_dim if i < layers_num-1 else 1
            mods.append(nn.Linear(in_dim, out_dim))
            if i < layers_num-1:
                if activation=='sin':
                    mods.append(nn.Sin())
                else:
                    mods.append(nn.LeakyReLU(0.01))
            in_dim = hidden_dim
        self.net = nn.Sequential(*mods)
    def forward(self, x):
        return self.net(x)

# ---- Cross-Attention Components ---- #
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=32, n_heads=1):
        super(CrossAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)
    def forward(self, Q_tokens, KV_tokens):
        out, attn_w = self.attn(Q_tokens, KV_tokens, KV_tokens)
        return out

class DynamicalF_CrossAttention(nn.Module):
    """
    Uses cross-attention between two feature groups:
      groupA (e.g., [first two dims of x, t, u]) and groupB (e.g., aggregated uₓ and uₜ).
    """
    def __init__(self, groupA_dim=3, groupB_dim=2, embed_dim=32, n_heads=1):
        super(DynamicalF_CrossAttention, self).__init__()
        self.embedA = nn.Linear(1, embed_dim)
        self.embedB = nn.Linear(1, embed_dim)
        self.crossAB = CrossAttentionBlock(embed_dim=embed_dim, n_heads=n_heads)
        self.crossBA = CrossAttentionBlock(embed_dim=embed_dim, n_heads=n_heads)
        self.final_fc = nn.Linear(embed_dim, 1)
        self.act = nn.LeakyReLU(0.01)
        self.groupA_dim = groupA_dim
        self.groupB_dim = groupB_dim
    def forward(self, groupA_feats, groupB_feats):
        B = groupA_feats.size(0)
        A_tokens = []
        for i in range(self.groupA_dim):
            A_i = groupA_feats[:, i].unsqueeze(1)
            A_tokens.append(self.embedA(A_i))
        A_tokens = torch.stack(A_tokens, dim=1)
        B_tokens = []
        for j in range(self.groupB_dim):
            B_j = groupB_feats[:, j].unsqueeze(1)
            B_tokens.append(self.embedB(B_j))
        B_tokens = torch.stack(B_tokens, dim=1)
        crossAB_out = self.crossAB(A_tokens, B_tokens)
        crossBA_out = self.crossBA(B_tokens, A_tokens)
        pooledA = crossAB_out.mean(dim=1)
        pooledB = crossBA_out.mean(dim=1)
        combined = (pooledA + pooledB) * 0.5
        out = self.final_fc(combined)
        return self.act(out)

# ---- LR Scheduler (same as before) ---- #
class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch=1, constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch*(num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5*(base_lr - final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))
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
            if self.constant_predictor_lr and param_group.get('name', None)=='predictor':
                param_group['lr'] = self.base_lr
            else:
                param_group['lr'] = lr
        self.iter += 1
        self.current_lr = lr
        return lr
    def get_lr(self):
        return self.current_lr

# ---- Full SPINN Model (Idea 3) with Cross-Attention, Metrics, and Plotting ---- #
class SPINN_AttnIdea3(nn.Module):
    def __init__(self, args, x_dim, architecture_args):
        super(SPINN_AttnIdea3, self).__init__()
        if args.save_folder and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)
        # solution_u as simple MLP
        mlp_args = architecture_args.get("solution_u_mlp_args", {"layers_num": 3, "hidden_dim": 32, "activation": "leaky-relu"})
        self.solution_u = SimpleMLP(x_dim, **mlp_args).to(device)
        # dynamical_F uses cross-attention
        embed_dim = architecture_args.get("cross_embed_dim", 32)
        n_heads = architecture_args.get("cross_n_heads", 1)
        self.dynamical_F = DynamicalF_CrossAttention(groupA_dim=3, groupB_dim=2, embed_dim=embed_dim, n_heads=n_heads).to(device)
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
    def load_model(self, path):
        ckp = torch.load(path)
        self.solution_u.load_state_dict(ckp['solution_u'])
        self.dynamical_F.load_state_dict(ckp['dynamical_F'])
    def predict(self, xt):
        return self.solution_u(xt)
    def Test(self, testloader):
        self.eval()
        true_labels = []
        pred_labels = []
        with torch.no_grad():
            for x1, _, y1, _ in testloader:
                x1 = x1.to(device)
                out = self.predict(x1)
                true_labels.append(y1.numpy())
                pred_labels.append(out.cpu().numpy())
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
        return np.mean((preds - trues)**2)
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
        return np.mean((preds - trues)**2)
    def forward(self, xt):
        xt.requires_grad = True
        x = xt[:, 0:-1]
        t = xt[:, -1:].clone()
        inp_u = torch.cat([x, t], dim=1)
        u = self.solution_u(inp_u)
        u_t = grad(u.sum(), t, create_graph=True)[0]
        u_x = grad(u.sum(), x, create_graph=True)[0]
        # Define groupA: (first two features of x, and u)
        groupA = torch.cat([x[:, 0:2], u], dim=1)  # shape: (B,3)
        # Define groupB: mean of u_x and u_t (each as scalar)
        u_x_mean = u_x.mean(dim=1, keepdim=True)
        groupB = torch.cat([u_x_mean, u_t], dim=1)   # shape: (B,2)
        F_out = self.dynamical_F(groupA, groupB)
        f = u_t - F_out
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
            zero_ = torch.zeros_like(f1)
            loss_pde = 0.5 * self.loss_func(f1, zero_) + 0.5 * self.loss_func(f2, zero_)
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
        for e in range(1, self.args.epochs+1):
            data_loss, pde_loss, phys_loss = self.train_one_epoch(e, trainloader)
            current_lr = self.scheduler.step()
            self.train_data_loss_history.append(data_loss)
            self.train_pde_loss_history.append(pde_loss)
            self.train_phys_loss_history.append(phys_loss)
            val_mse = self.ValidMSE(validloader) if validloader is not None else 0.0
            self.valid_mse_history.append(val_mse)
            test_mse = self.TestMSE(testloader) if testloader is not None else 0.0
            self.test_mse_history.append(test_mse)
            msg = (f"[Epoch {e}] lr={current_lr:.6f}, data={data_loss:.6f}, pde={pde_loss:.6f}, "
                   f"phys={phys_loss:.6f}, val_mse={val_mse:.6f}, test_mse={test_mse:.6f}")
            self.logger.info(msg)
            if validloader is not None and val_mse < min_valid_mse:
                min_valid_mse = val_mse
                early_stop_counter = 0
                true_label, pred_label = self.Test(testloader)
                metrics = eval_metrix(pred_label, true_label)
                self.logger.info(f"[Test Metrics] MAE={metrics[0]:.6f}, MAPE={metrics[1]:.6f}, MSE={metrics[2]:.6f}, RMSE={metrics[3]:.6f}")
                self.best_model = {"solution_u": self.solution_u.state_dict(),
                                   "dynamical_F": self.dynamical_F.state_dict()}
                if self.args.save_folder:
                    np.save(os.path.join(self.args.save_folder, "true_label.npy"), true_label)
                    np.save(os.path.join(self.args.save_folder, "pred_label.npy"), pred_label)
            else:
                early_stop_counter += 1
            if self.args.early_stop and early_stop_counter > self.args.early_stop:
                self.logger.info(f"Early stopping at epoch {e}")
                break
        if self.best_model and self.args.save_folder:
            torch.save(self.best_model, os.path.join(self.args.save_folder, "model.pth"))
        # Plotting
        ep_range = range(1, len(self.train_data_loss_history)+1)
        plt.figure()
        plt.plot(ep_range, self.train_data_loss_history, label="Train-DataLoss")
        plt.plot(ep_range, self.train_pde_loss_history, label="Train-PDELoss")
        plt.plot(ep_range, self.train_phys_loss_history, label="Train-PhysLoss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Losses (Idea 3)")
        if self.args.save_folder:
            plt.savefig(os.path.join(self.args.save_folder, "training_losses.png"))
            plt.close()
        else:
            plt.show()
        plt.figure()
        plt.plot(ep_range, self.valid_mse_history, label="Valid MSE")
        plt.plot(ep_range, self.test_mse_history, label="Test MSE")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.title("Validation & Test MSE (Idea 3)")
        if self.args.save_folder:
            plt.savefig(os.path.join(self.args.save_folder, "valid_test_mse.png"))
            plt.close()
        else:
            plt.show()
        self.clear_logger()
