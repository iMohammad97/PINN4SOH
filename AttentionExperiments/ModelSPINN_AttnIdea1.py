import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
import os
from tqdm import tqdm
from functools import reduce

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")

# ------------------------- Basic Building Blocks ------------------------- #
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class Subnet(nn.Module):
    """
    Same as your original Subnet, used to encode a single dimension of input.
    """
    def __init__(self, output_dim, layers_num, hidden_dim, dropout, activation):
        super(Subnet, self).__init__()
        input_dim = 1
        assert layers_num >= 2, "layers must be >= 2"
        self.output_dim = output_dim

        layers = []
        for i in range(layers_num):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
                if activation == "sin":
                    layers.append(Sin())
                elif activation == "leaky-relu":
                    layers.append(nn.LeakyReLU(negative_slope=0.01))
            elif i == layers_num - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if activation == "sin":
                    layers.append(Sin())
                elif activation == "leaky-relu":
                    layers.append(nn.LeakyReLU(negative_slope=0.01))
                layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # x shape = (Batch, 1)
        return self.net(x)


# -------------- Attention-Based Replacement for "Solution_u_spinn" -------------- #
class Solution_u_spinn_Attn(nn.Module):
    """
    Instead of a simple sum of subnet outputs with a trainable weight vector,
    we apply scaled dot-product self-attention across the dimension embeddings.
    """
    def __init__(self, input_dim, subnet_args, attn_embed_dim=16, n_heads=1):
        """
        :param input_dim: Number of input features (e.g. 17).
        :param subnet_args: A dict specifying the Subnet architecture (output_dim, layers_num, etc).
        :param attn_embed_dim: The dimension used inside the nn.MultiheadAttention.
        :param n_heads: Number of attention heads to use.
        """
        super(Solution_u_spinn_Attn, self).__init__()
        self.input_dim = input_dim

        # Create one subnet per input dimension
        self.subnets = nn.ModuleList()
        for _ in range(input_dim):
            self.subnets.append(Subnet(**subnet_args))

        # We expect each Subnet to output shape (Batch, output_dim).
        # We'll transform that to the required embed dimension for MultiheadAttention.
        # For simplicity, let's ensure output_dim == attn_embed_dim.
        # If they differ, we add a linear projection.
        self.proj = None
        if self.subnets[0].output_dim != attn_embed_dim:
            self.proj = nn.Linear(self.subnets[0].output_dim, attn_embed_dim)

        # MultiheadAttention in PyTorch expects input shape (seq_len, batch, embed_dim)
        # or (batch, seq_len, embed_dim) if batch_first=True. We'll set batch_first=True.
        self.self_attn = nn.MultiheadAttention(
            embed_dim=attn_embed_dim,
            num_heads=n_heads,
            batch_first=True
        )

        # After attention, we want to reduce across the `seq_len = input_dim` dimension,
        # resulting in a single scalar. We'll do a simple linear: (attn_embed_dim -> 1).
        self.final_fc = nn.Linear(attn_embed_dim, 1)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        """
        x shape = (Batch, input_dim).
        We pass x[:, i] through the i-th Subnet, producing a token.
        Then we run attention across these tokens, and reduce to (Batch, 1).
        """
        # x is on device, shape (B, input_dim)
        B = x.shape[0]
        # 1) Gather each dimension's input -> pass through subnets
        # We'll get a list of [ (B, out_dim), (B, out_dim), ..., (B, out_dim) ] for each dimension
        outputs = []
        for idx, net in enumerate(self.subnets):
            dim_input = x[:, idx].unsqueeze(1)   # shape (B,1)
            dim_out = net(dim_input)             # shape (B, out_dim)
            outputs.append(dim_out)

        # 2) Stack them: shape (B, input_dim, out_dim)
        tokens = torch.stack(outputs, dim=1)

        # 3) Possibly project tokens to attn_embed_dim if needed
        if self.proj is not None:
            tokens = self.proj(tokens)  # shape (B, input_dim, attn_embed_dim)

        # 4) Apply self-attention across the input_dim dimension
        #    We treat each dimension as a "token".
        #    MultiheadAttention with batch_first=True expects shape (B, seq_len, embed_dim).
        #    We'll do self-attn in a standard "Q=K=V" scenario.
        attn_output, attn_weights = self.self_attn(tokens, tokens, tokens)
        # attn_output shape: (B, input_dim, attn_embed_dim)

        # 5) We can reduce across the input_dim dimension.
        #    A simple approach is to take the mean or sum across tokens, then a final linear -> 1.
        #    We'll do a mean here:
        pooled = attn_output.mean(dim=1)  # shape (B, attn_embed_dim)

        # 6) Map to a single scalar
        out = self.final_fc(pooled)  # shape (B,1)
        out = self.activation(out)
        return out


# -------------- Attention-Based Replacement for "Dynamical_F_spinn" -------------- #
class Dynamical_F_spinn_Attn(nn.Module):
    """
    Similar approach for the PDE part. The input is (x, t, u, u_x, u_t).
    In your original code, you create one Subnet per dimension (so total input_dim = x_dim*2 + 1).
    Then you do a weighted sum. We do the same but with attention.
    """
    def __init__(self, x_dim, subnet_args, attn_embed_dim=16, n_heads=1):
        super(Dynamical_F_spinn_Attn, self).__init__()
        # input_dim = x_dim*2 + 1 + ???
        # But in your code, for spinn: "Dynamical_F_spinn" uses subnets for each feature among
        # [x_t, u, u_x, u_t], so the dimension = x_dim + x_dim + 1 + ?
        # Actually your original "Dynamical_F_spinn" sets "self.input_dim = x_dim*2 + 1".
        self.input_dim = x_dim * 2 + 1

        # Make subnets
        self.subnets = nn.ModuleList()
        for _ in range(self.input_dim):
            self.subnets.append(Subnet(**subnet_args))

        # Possibly project to attention dimension
        self.proj = None
        if self.subnets[0].output_dim != attn_embed_dim:
            self.proj = nn.Linear(self.subnets[0].output_dim, attn_embed_dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=attn_embed_dim,
            num_heads=n_heads,
            batch_first=True
        )
        self.final_fc = nn.Linear(attn_embed_dim, 1)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        # x shape = (B, self.input_dim)
        B = x.shape[0]
        outputs = []
        for idx, net in enumerate(self.subnets):
            dim_input = x[:, idx].unsqueeze(1)  # shape (B,1)
            dim_out = net(dim_input)            # shape (B, out_dim)
            outputs.append(dim_out)

        tokens = torch.stack(outputs, dim=1)    # shape (B, self.input_dim, out_dim)

        # Project if needed
        if self.proj is not None:
            tokens = self.proj(tokens)

        attn_output, attn_weights = self.self_attn(tokens, tokens, tokens)
        # reduce across dimension tokens:
        pooled = attn_output.mean(dim=1)        # shape (B, attn_embed_dim)
        out = self.final_fc(pooled)             # shape (B,1)
        out = self.activation(out)
        return out


# ------------------------- Full SPINN with Attention ------------------------- #
class LR_Scheduler(object):
    """
    Same LR scheduler as your original code, unchanged.
    """
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
        for param_group in self.optimizer.param_groups:
            if self.constant_predictor_lr and param_group.get('name', None) == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Helper MSE aggregator
class AverageMeter(object):
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1

    @property
    def avg(self):
        if self.count == 0:
            return 0
        else:
            return self.sum / self.count


def eval_metrix(pred_label, true_label):
    # same as your original code's snippet
    # Return [MAE, MAPE, MSE, RMSE]
    mae = np.mean(np.abs(pred_label - true_label))
    mape = np.mean(np.abs((true_label - pred_label) / (true_label + 1e-8))) * 100
    mse = np.mean((pred_label - true_label) ** 2)
    rmse = np.sqrt(mse)
    return [mae, mape, mse, rmse]


def get_logger(log_path):
    # minimal logger
    import logging
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        # If already has handlers, use them
        return logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, mode='a')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return logger


def write_to_txt(save_name, content):
    with open(save_name, 'a') as f:
        f.write(str(content) + '\n')


# ----------------------- The SPINN_AttnIdea1 Model ----------------------- #
class SPINN_AttnIdea1(nn.Module):
    """
    Full model that uses the attention-based solution_u and dynamical_F modules from above.
    The rest of the training logic stays the same as your original SPINN.
    """
    def __init__(self, args, x_dim, architecture_args):
        super(SPINN_AttnIdea1, self).__init__()
        if args.save_folder is not None and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)


        attn_embed_dim_u = architecture_args.get("attn_embed_dim_u", 16)
        attn_heads_u = architecture_args.get("attn_heads_u", 1)
        attn_embed_dim_F = architecture_args.get("attn_embed_dim_F", 16)
        attn_heads_F = architecture_args.get("attn_heads_F", 1)

        self.solution_u = Solution_u_spinn_Attn(
            input_dim=x_dim,
            subnet_args=architecture_args["solution_u_subnet_args"],
            attn_embed_dim=attn_embed_dim_u,
            n_heads=attn_heads_u
        ).to(device)

        self.dynamical_F = Dynamical_F_spinn_Attn(
            x_dim=x_dim,
            subnet_args=architecture_args["dynamical_F_subnet_args"],
            attn_embed_dim=attn_embed_dim_F,
            n_heads=attn_heads_F
        ).to(device)


        self.optimizer1 = torch.optim.Adam(self.solution_u.parameters(), lr=args.warmup_lr)
        self.optimizer2 = torch.optim.Adam(self.dynamical_F.parameters(), lr=args.lr_F)
        self.scheduler = LR_Scheduler(
            optimizer=self.optimizer1,
            warmup_epochs=args.warmup_epochs,
            warmup_lr=args.warmup_lr,
            num_epochs=args.epochs,
            base_lr=args.lr,
            final_lr=args.final_lr
        )

        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()

        self.alpha = args.alpha
        self.beta = args.beta
        self.args = args
        self.best_model = None

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
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
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter, (x1, _, y1, _) in enumerate(testloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1.numpy())
                pred_label.append(u1.cpu().numpy())

        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        return true_label, pred_label

    def Valid(self, validloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter, (x1, _, y1, _) in enumerate(validloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1.numpy())
                pred_label.append(u1.cpu().numpy())

        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        mse = self.loss_func(torch.from_numpy(pred_label), torch.from_numpy(true_label))
        return mse.item()

    def forward(self, xt):
        # just like original
        xt.requires_grad = True
        x = xt[:, 0:-1]
        t = xt[:, -1:].clone()

        # pass through solution_u
        u = self.solution_u(torch.cat((x, t), dim=1))

        # partial derivatives
        u_t = grad(u.sum(), t, create_graph=True, only_inputs=True, allow_unused=True)[0]
        u_x = grad(u.sum(), x, create_graph=True, only_inputs=True, allow_unused=True)[0]

        # build input for dynamical_F
        # in original code, we do "torch.cat([xt, u, u_x, u_t], dim=1)"
        # but note that x, t have shape (B, x_dim?), u, u_x, u_t have shape (B, ?). We'll just replicate that logic
        F_in = torch.cat([xt, u, u_x, u_t], dim=1)
        F = self.dynamical_F(F_in)
        f = u_t - F
        return u, f

    def train_one_epoch(self, epoch, dataloader):
        self.train()
        loss1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()

        for iter, (x1, x2, y1, y2) in tqdm(enumerate(dataloader), total=len(dataloader)):
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)
            u1, f1 = self.forward(x1)
            u2, f2 = self.forward(x2)

            loss_data = 0.5 * self.loss_func(u1, y1) + 0.5 * self.loss_func(u2, y2)
            zero_target = torch.zeros_like(f1)
            loss_pde = 0.5 * self.loss_func(f1, zero_target) + 0.5 * self.loss_func(f2, zero_target)

            # physics-based monotonic constraint
            loss_phys = self.relu((u2 - u1) * (y1 - y2)).sum()

            loss = loss_data + self.alpha * loss_pde + self.beta * loss_phys

            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()

            loss1_meter.update(loss_data.item())
            loss2_meter.update(loss_pde.item())
            loss3_meter.update(loss_phys.item())

        return loss1_meter.avg, loss2_meter.avg, loss3_meter.avg

    def Train(self, trainloader, testloader=None, validloader=None):
        min_valid_mse = 1e8
        early_stop_counter = 0

        for e in range(1, self.args.epochs + 1):
            self.logger.info(f"Epoch {e}/{self.args.epochs}")
            loss_data, loss_pde, loss_phys = self.train_one_epoch(e, trainloader)
            current_lr = self.scheduler.step()
            self.logger.info(f'[Train] epoch:{e}, lr:{current_lr:.6f}, total loss:{(loss_data + self.alpha*loss_pde + self.beta*loss_phys):.6f}')

            if validloader is not None:
                valid_mse = self.Valid(validloader)
                self.logger.info(f'[Valid] epoch:{e}, MSE: {valid_mse:.6f}')

                if valid_mse < min_valid_mse and testloader is not None:
                    min_valid_mse = valid_mse
                    true_label, pred_label = self.Test(testloader)
                    MAE, MAPE, MSE, RMSE = eval_metrix(pred_label, true_label)
                    self.logger.info(f'[Test] MSE:{MSE:.8f}, MAE:{MAE:.6f}, MAPE:{MAPE:.6f}, RMSE:{RMSE:.6f}')
                    early_stop_counter = 0

                    self.best_model = {
                        'solution_u': self.solution_u.state_dict(),
                        'dynamical_F': self.dynamical_F.state_dict()
                    }
                    if self.args.save_folder is not None:
                        np.save(os.path.join(self.args.save_folder, 'true_label.npy'), true_label)
                        np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), pred_label)
                else:
                    early_stop_counter += 1

                if self.args.early_stop is not None and early_stop_counter > self.args.early_stop:
                    self.logger.info(f'Early stop at epoch {e}')
                    break

        # Save final best model
        if self.best_model is not None and self.args.save_folder is not None:
            torch.save(self.best_model, os.path.join(self.args.save_folder, 'model.pth'))
        self.clear_logger()
