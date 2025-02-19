import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
from utils.util import AverageMeter,get_logger,eval_metrix
import os
from tqdm import tqdm
from functools import reduce


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)

class Subnet(nn.Module):
    def __init__(self, output_dim, layers_num, hidden_dim, dropout, activation):
        super(Subnet, self).__init__()
        input_dim = 1
        assert layers_num >= 2, "layers must be greater than 2"
        assert activation=="leaky-relu" or activation=="sin", "activation must be leaky-relu or sin"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim

        self.layers = []
        for i in range(layers_num):
            if i == 0:
                self.layers.append(nn.Linear(input_dim,hidden_dim))
                if activation == "sin":
                    self.layers.append(Sin())
                elif activation == "leaky-relu":
                    self.layers.append(nn.LeakyReLU(negative_slope=0.01))
            elif i == layers_num-1:
                self.layers.append(nn.Linear(hidden_dim,output_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim,hidden_dim))
                if activation == "sin":
                    self.layers.append(Sin())
                elif activation == "leaky-relu":
                    self.layers.append(nn.LeakyReLU(negative_slope=0.01))
                self.layers.append(nn.LeakyReLU(negative_slope=0.01))
                self.layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*self.layers)
        self._init()

    def _init(self):
        for layer in self.net:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self,x):
        assert x.shape[1] == self.input_dim
        x = self.net(x)
        return x

class Solution_u(nn.Module):
    def __init__(self, input_dim, layers_num, hidden_dim, dropout, activation):
        super(Solution_u, self).__init__()
        self.input_dim = input_dim
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim

        self.layers = []
        for i in range(layers_num):
            if i == 0:
                self.layers.append(nn.Linear(input_dim,hidden_dim))
                if activation == "sin":
                    self.layers.append(Sin())
                elif activation == "leaky-relu":
                    self.layers.append(nn.LeakyReLU(negative_slope=0.01))
            elif i == layers_num-1:
                self.layers.append(nn.Linear(hidden_dim,1))
            else:
                self.layers.append(nn.Linear(hidden_dim,hidden_dim))
                if activation == "sin":
                    self.layers.append(Sin())
                elif activation == "leaky-relu":
                    self.layers.append(nn.LeakyReLU(negative_slope=0.01))
                self.layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*self.layers)
        self._init()

    def forward(self, x):
        x = self.net(x)
        return x

    def _init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)

class Solution_u_spinn(nn.Module):
    def __init__(self, input_dim, subnet_args):
        super(Solution_u_spinn, self).__init__()
        self.input_dim = input_dim
        self.subnets = nn.ModuleList()  
        for _ in range(input_dim):
            self.subnets.append(Subnet(**subnet_args))
        # Define a trainable weight vector with input_dim elements
        self.weights = nn.Parameter(torch.ones(input_dim))
        self._init_()

    def forward(self, x):
        assert x.shape[1] == self.input_dim
        outs = []
        for idx, net in enumerate(self.subnets):
            outs.append(net(x[:, idx].unsqueeze(1)))
        
        # Combine outputs using the trainable weights
        weighted_outs = [w * out for w, out in zip(self.weights, outs)]
        result = sum(weighted_outs)
        result = result.sum(dim=1, keepdim=True)
        return result

    def _init_(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)


class Dynamical_F(nn.Module):
    def __init__(self, x_dim, layers_num, hidden_dim, dropout, activation):
        super(Dynamical_F, self).__init__()
        self.input_dim = x_dim*2 + 1
        self.output_dim = 1
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim

        self.layers = []
        for i in range(layers_num):
            if i == 0:
                self.layers.append(nn.Linear(self.input_dim,hidden_dim))
                if activation == "sin":
                    self.layers.append(Sin())
                elif activation == "leaky-relu":
                    self.layers.append(nn.LeakyReLU(negative_slope=0.01))
            elif i == layers_num-1:
                self.layers.append(nn.Linear(hidden_dim,self.output_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim,hidden_dim))
                if activation == "sin":
                    self.layers.append(Sin())
                elif activation == "leaky-relu":
                    self.layers.append(nn.LeakyReLU(negative_slope=0.01))
                self.layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*self.layers)

    def forward(self,x):
        assert x.shape[1] == self.input_dim
        x = self.net(x)
        return x

    def _init_(self):
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)
            elif isinstance(layer,nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)

class Dynamical_F_spinn(nn.Module):
    def __init__(self, x_dim, subnet_args):
        super(Dynamical_F_spinn, self).__init__()
        self.input_dim = x_dim*2 + 1
        self.subnets = nn.ModuleList()  
        for _ in range(self.input_dim):
            self.subnets.append(Subnet(**subnet_args))
        # Define a trainable weight vector with input_dim elements
        self.weights = nn.Parameter(torch.ones(self.input_dim))
        self._init_()
        

    def forward(self, x):
        assert x.shape[1] == self.input_dim
        outs = []
        for idx, net in enumerate(self.subnets):
            outs.append(net(x[:, idx].unsqueeze(1)))
        
        # Combine outputs using the trainable weights
        weighted_outs = [w * out for w, out in zip(self.weights, outs)]
        result = sum(weighted_outs)
        result = result.sum(dim=1, keepdim=True)
        return result

    def _init_(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)

def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has {} trainable parameters'.format(count))
    return count


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
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr


class SPINN(nn.Module):
    def __init__(self,args, x_dim, architecture_args):
        super(SPINN, self).__init__()
        self.args = args
        if args.save_folder is not None and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)
        self._save_args()
        spinn_enabled = architecture_args['spinn_enabled']
        if spinn_enabled['solution_u']:
            subnet_args = architecture_args['solution_u_subnet_args']
            self.solution_u = Solution_u_spinn(input_dim=x_dim, subnet_args=subnet_args).to(device)
        else:
            solution_u_args = architecture_args['solution_u_args']
            self.solution_u = Solution_u(input_dim=x_dim, **solution_u_args).to(device)
        if spinn_enabled['dynamical_F']:
            subnet_args = architecture_args['dynamical_F_subnet_args']
            self.dynamical_F = Dynamical_F_spinn(x_dim=x_dim, subnet_args=subnet_args).to(device)
        else:
            dynamical_F_args = architecture_args['dynamical_F_args']
            self.dynamical_F = Dynamical_F(x_dim=x_dim, **dynamical_F_args).to(device)

        self.optimizer1 = torch.optim.Adam(self.solution_u.parameters(), lr=args.warmup_lr)
        self.optimizer2 = torch.optim.Adam(self.dynamical_F.parameters(), lr=args.lr_F)

        self.scheduler = LR_Scheduler(optimizer=self.optimizer1,
                                      warmup_epochs=args.warmup_epochs,
                                      warmup_lr=args.warmup_lr,
                                      num_epochs=args.epochs,
                                      base_lr=args.lr,
                                      final_lr=args.final_lr)

        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()

        self.best_model = None

        self.alpha = self.args.alpha
        self.beta = self.args.beta

    def _save_args(self):
        if self.args.log_dir is not None:
            self.logger.info("Args:")
            for k, v in self.args.__dict__.items():
                self.logger.critical(f"\t{k}:{v}")

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.solution_u.load_state_dict(checkpoint['solution_u'])
        self.dynamical_F.load_state_dict(checkpoint['dynamical_F'])
        for param in self.solution_u.parameters():
            param.requires_grad = True

    def predict(self,xt):
        return self.solution_u(xt)

    def Test(self,testloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x1,_,y1,_) in enumerate(testloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)

        return true_label,pred_label

    def Valid(self,validloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x1,_,y1,_) in enumerate(validloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)
        mse = self.loss_func(torch.tensor(pred_label),torch.tensor(true_label))
        return mse.item()

    def forward(self,xt):
        xt.requires_grad = True
        x = xt[:,0:-1]
        t = xt[:,-1:]

        u = self.solution_u(torch.cat((x,t),dim=1))

        u_t = grad(u.sum(),t,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]
        u_x = grad(u.sum(),x,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]

        F = self.dynamical_F(torch.cat([xt,u,u_x,u_t],dim=1))

        f = u_t - F
        return u,f

    def train_one_epoch(self,epoch,dataloader):
        self.train()
        loss1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()
        printed = False
        for iter,(x1,x2,y1,y2) in tqdm(enumerate(dataloader)):
            x1,x2,y1,y2 = x1.to(device),x2.to(device),y1.to(device),y2.to(device)
            u1,f1 = self.forward(x1)
            u2,f2 = self.forward(x2)

            # data loss
            loss1 = 0.5*self.loss_func(u1,y1) + 0.5*self.loss_func(u2,y2)

            # PDE loss
            f_target = torch.zeros_like(f1)
            loss2 = 0.5*self.loss_func(f1,f_target) + 0.5*self.loss_func(f2,f_target)

            # physics loss  y2-y1<0
            loss3 = self.relu(torch.mul(u2-u1,y1-y2)).sum()

            # total loss
            loss = loss1 + self.alpha*loss2 + self.beta*loss3
            
            # if not printed:
            #     print(f"u1 is {u1}")
            #     printed = True

            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()
            # for name, param in self.solution_u.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name} grad mean: {param.grad.abs().mean()}")
            #     else:
            #         print(f"{name} has no grad")
            # raise Exception("wait")
            self.optimizer1.step()
            self.optimizer2.step()

            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
            loss3_meter.update(loss3.item())
            
            if (iter+1) % 50 == 0:
                print("[epoch:{} iter:{}] data loss:{:.6f}, PDE loss:{:.6f}, physics loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3))
        return loss1_meter.avg,loss2_meter.avg,loss3_meter.avg

    def Train(self,trainloader,testloader=None,validloader=None):
        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10
        for e in range(1,self.args.epochs+1):
            print(f"Epoch {e}/{self.args.epochs}")
            early_stop += 0
            loss1,loss2,loss3 = self.train_one_epoch(e,trainloader)
            current_lr = self.scheduler.step()
            info = '[Train] epoch:{}, lr:{:.6f}, ' \
                   'total loss:{:.6f}'.format(e,current_lr,loss1+self.alpha*loss2+self.beta*loss3)
            self.logger.info(info)
            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(validloader)
                info = '[Valid] epoch:{}, MSE: {}'.format(e,valid_mse)
                self.logger.info(info)
            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label,pred_label = self.Test(testloader)
                [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE)
                self.logger.info(info)
                early_stop = 0

                ############################### save ############################################
                self.best_model = {'solution_u':self.solution_u.state_dict(),
                                   'dynamical_F':self.dynamical_F.state_dict()}
                if self.args.save_folder is not None:
                    np.save(os.path.join(self.args.save_folder, 'true_label.npy'), true_label)
                    np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), pred_label)
                ##################################################################################
            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                info = 'early stop at epoch {}'.format(e)
                self.logger.info(info)
                break
        self.clear_logger()
        if self.args.save_folder is not None:
            torch.save(self.best_model,os.path.join(self.args.save_folder,'model.pth'))