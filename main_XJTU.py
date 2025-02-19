from dataloader.dataloader import XJTUdata
from dataloader.dataloader2 import XJTUdata as XJTUdata2
from Model.Model import PINN, count_parameters
from Model.ModelSPINN import SPINN
import argparse
import os
import shutil
import math
from tqdm import tqdm
from itertools import product
import pickle
import json


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_data(args,small_sample=None):
    root = 'data/XJTU data'
    data = XJTUdata(root=root, args=args)
    train_list = []
    test_list = []
    files = os.listdir(root)
    for file in files:
        if args.batch in file:
            if '4' in file or '8' in file:
                test_list.append(os.path.join(root, file))
            else:
                train_list.append(os.path.join(root, file))
    if small_sample is not None:
        train_list = train_list[:small_sample]

    train_loader = data.read_all(specific_path_list=train_list)
    test_loader = data.read_all(specific_path_list=test_list)
    dataloader = {'train': train_loader['train_2'],
                  'valid': train_loader['valid_2'],
                  'test': test_loader['test_3']}
    return dataloader
def load_data2(args,small_sample=None):
    root = 'data/XJTU data'
    data = XJTUdata2(root=root, args=args)
    train_list = []
    test_list = []
    files = os.listdir(root)
    for file in files:
        if args.batch in file:
            if '4' in file or '8' in file:
                test_list.append(os.path.join(root, file))
            else:
                train_list.append(os.path.join(root, file))
    if small_sample is not None:
        train_list = train_list[:small_sample]

    train_loader = data.read_all(specific_path_list=train_list)
    test_loader = data.read_all(specific_path_list=test_list)
    dataloader = {'train': train_loader['train'],
                  'valid': train_loader['valid'],
                  'test': test_loader['test']}
    return dataloader


def grid_search():
    # Initialize arguments and batches
    args = get_args()
    batches = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
    main_results_dir = 'our-experiments-2'
    if not os.path.exists(main_results_dir):
        os.makedirs(main_results_dir)
    
    with open('permutations.pkl', 'rb') as file:
        permutations = pickle.load(file)

    # Perform grid search
    for idx, architecture_args in enumerate(permutations, 1):
        # Iterate through all batch configurations
        for i, batch in enumerate(batches):
            print(f'Doing batch {i+1} with {architecture_args}')
            setattr(args, 'batch', batch)  # Set current batch

            # Dummy loop, iterate as needed
            for e in range(1):
                save_folder = f'{main_results_dir}/results of reviewer-{idx}/XJTU results/{i}-{i}'
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                log_dir = 'logging.txt'
                setattr(args, "save_folder", save_folder)
                setattr(args, "log_dir", log_dir)

                print("Loading data...")
                dataloader = load_data(args)

                # Initialize model
                spinn = SPINN(args, x_dim=17, architecture_args=architecture_args)
                print("---------------XXXXXXXX_________________")
                num_params = count_parameters(spinn)

                print("Training...")
                spinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])

                # Write the number of parameters to a file
                with open(f"{main_results_dir}/results of reviewer-{idx}/XJTU results/{i}-{i}/num_param.txt", 'w') as f:
                    f.write(str(num_params))
                with open(f"{main_results_dir}/results of reviewer-{idx}/XJTU results/{i}-{i}/hyper_params.json", 'w') as json_file:
                    json.dump(architecture_args, json_file, indent=4)

def grid_search_constructor():
    # Define the permutations dictionary
    permutations = {
        "spinn_enabled": {
            "solution_u": [True, False],
            "dynamical_F": [True, False]
        },
        "dynamical_F_args": {
            "layers_num": [2, 3, 5],
            "hidden_dim": [20, 40, 60],
            "dropout": [0],
            "activation": ["leaky-relu"]
        },
        "solution_u_args": {
            "layers_num": [2, 3, 5],
            "hidden_dim": [20, 40, 60],
            "dropout": [0],
            "activation": ["leaky-relu"]
        },
        "dynamical_F_subnet_args": {
            "output_dim": [5, 10, 15],
            "layers_num": [2, 3, 5],
            "hidden_dim": [-1],
            "dropout": [0],
            "activation": ["leaky-relu"]
        },
        "solution_u_subnet_args": {
            "output_dim": [5, 10, 15],
            "layers_num": [2, 3, 5],
            "hidden_dim": [-1],
            "dropout": [0],
            "activation": ["leaky-relu"]
        }
    }

    # Extract all combinations for each parameter set
    spinn_enabled_combinations = list(product(permutations["spinn_enabled"]["solution_u"],
                                              permutations["spinn_enabled"]["dynamical_F"]))

    dynamical_F_args_combinations = list(product(permutations["dynamical_F_args"]["layers_num"],
                                                 permutations["dynamical_F_args"]["hidden_dim"],
                                                 permutations["dynamical_F_args"]["dropout"],
                                                 permutations["dynamical_F_args"]["activation"]))

    solution_u_args_combinations = list(product(permutations["solution_u_args"]["layers_num"],
                                                permutations["solution_u_args"]["hidden_dim"],
                                                permutations["solution_u_args"]["dropout"],
                                                permutations["solution_u_args"]["activation"]))

    dynamical_F_subnet_args_combinations = list(product(permutations["dynamical_F_subnet_args"]["output_dim"],
                                                        permutations["dynamical_F_subnet_args"]["layers_num"],
                                                        permutations["dynamical_F_subnet_args"]["hidden_dim"],
                                                        permutations["dynamical_F_subnet_args"]["dropout"],
                                                        permutations["dynamical_F_subnet_args"]["activation"]))

    solution_u_subnet_args_combinations = list(product(permutations["solution_u_subnet_args"]["output_dim"],
                                                       permutations["solution_u_subnet_args"]["layers_num"],
                                                       permutations["solution_u_subnet_args"]["hidden_dim"],
                                                       permutations["solution_u_subnet_args"]["dropout"],
                                                       permutations["solution_u_subnet_args"]["activation"]))

    # Generate all possible architecture arguments
    all_possible_permutations = []

    for spinn_enabled in spinn_enabled_combinations:
        solution_u_enabled, dynamical_F_enabled = spinn_enabled

        # Skip combinations where both are False
        if not (solution_u_enabled or dynamical_F_enabled):
            continue
        
        if solution_u_enabled:
            solution_u_args_list = [None]
            solution_u_subnet_args_list = solution_u_subnet_args_combinations
        else:
            solution_u_args_list = solution_u_args_combinations
            solution_u_subnet_args_list = [None]
            
        if dynamical_F_enabled:
            dynamical_F_args_list = [None]
            dynamical_F_subnet_args_list = dynamical_F_subnet_args_combinations
        else:
            dynamical_F_args_list = dynamical_F_args_combinations
            dynamical_F_subnet_args_list = [None]
        
        for solution_u_args in solution_u_args_list:
            for dynamical_F_args in dynamical_F_args_list:
                for solution_u_subnet_args in solution_u_subnet_args_list:
                    for dynamical_F_subnet_args in dynamical_F_subnet_args_list:
                        architecture_args = {
                            "solution_u_args": None if solution_u_args is None else {
                                "layers_num": solution_u_args[0],
                                "hidden_dim": solution_u_args[1],
                                "dropout": solution_u_args[2],
                                "activation": solution_u_args[3]
                            },
                            "dynamical_F_args": None if dynamical_F_args is None else {
                                "layers_num": dynamical_F_args[0],
                                "hidden_dim": dynamical_F_args[1],
                                "dropout": dynamical_F_args[2],
                                "activation": dynamical_F_args[3]
                            },
                            "solution_u_subnet_args": None if solution_u_subnet_args is None else {
                                "output_dim": solution_u_subnet_args[0],
                                "layers_num": solution_u_subnet_args[1],
                                "hidden_dim": math.ceil(1.5 * solution_u_subnet_args[0]),
                                "dropout": solution_u_subnet_args[3],
                                "activation": solution_u_subnet_args[4]
                            },
                            "dynamical_F_subnet_args": None if dynamical_F_subnet_args is None else {
                                "output_dim": dynamical_F_subnet_args[0],
                                "layers_num": dynamical_F_subnet_args[1],
                                "hidden_dim": math.ceil(1.5 * dynamical_F_subnet_args[0]),
                                "dropout": dynamical_F_subnet_args[3],
                                "activation": dynamical_F_subnet_args[4]
                            },
                            "spinn_enabled": {
                                "solution_u": solution_u_enabled,
                                "dynamical_F": dynamical_F_enabled
                            }
                        }
                        all_possible_permutations.append(architecture_args)

    # Let's see the total number of unique combinations generated
    print(f"Total number of unique permutations: {len(all_possible_permutations)}")
    with open('permutations.pkl', 'wb') as file:
        pickle.dump(all_possible_permutations, file)

# Call the function
def main():
    # grid_search_constructor()
    # grid_search()
    main2()
def main2():
    args = get_args()
    batchs = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
    for i in range(6):
        print(f'doing batch {i+1}')
        batch = batchs[i]
        setattr(args, 'batch', batch)
        for e in range(1):
            save_folder = 'results of reviewer/XJTU results/' + str(i) + '-' + str(i)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            log_dir = 'logging.txt'
            setattr(args, "save_folder", save_folder)
            setattr(args, "log_dir", log_dir)

            print("loading data...")
            dataloader = load_data2(args)
            pinn = PINN(args)
            print("---------------XXXXXXXX_________________")
            count_parameters(pinn)
            # for p in pinn.parameters():
            #     print(p)
            # break
            
            solution_u_subnet_args = {
                "output_dim" : 15, 
                "layers_num" : 3, 
                "hidden_dim" : 20,
                "dropout" : 0.2,
                "activation" : "sin"
            }
            dynamical_F_subnet_args = {
                "output_dim" : 15, 
                "layers_num" : 3, 
                "hidden_dim" : 20,
                "dropout" : 0.2,
                "activation" : "sin"
            }
            solution_u_args = {
                "layers_num" : 3,
                "hidden_dim" : 60,
                "dropout" : 0.2,
                "activation" : "sin"
            }
            dynamical_F_args = {
                "layers_num" : 3, 
                "hidden_dim" : 60, 
                "dropout" : 0.2,
                "activation" : "sin"
            }
            spinn_enabled = {
                "solution_u" : True,
                "dynamical_F" : True
            }
            architecture_args = {
                "solution_u_args" : solution_u_args,
                "dynamical_F_args" : dynamical_F_args,
                "solution_u_subnet_args" : solution_u_subnet_args,
                "dynamical_F_subnet_args" : dynamical_F_subnet_args,
                "spinn_enabled" : spinn_enabled
            }
            spinn = SPINN(args, x_dim=17, architecture_args=architecture_args)
            print("---------------XXXXXXXX_________________")
            count_parameters(spinn)
            
            print("training...")    
            pinn.Train(trainloader=dataloader['train'],validloader=dataloader['valid'],testloader=dataloader['test'])

def small_sample():
    args = get_args()
    batchs = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
    for n in [1,2,3,4]:
        for i in range(6):
            batch = batchs[i]
            setattr(args, 'batch', batch)
            setattr(args,'batch_size',128)
            setattr(args,'alpha',0.5)
            setattr(args,'beta',10)
            for e in range(10):
                save_folder = f'results/XJTU results (small sample {n})/' + str(i) + '-' + str(i) + '/Experiment' + str(e + 1)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                log_dir = 'logging.txt'
                setattr(args, "save_folder", save_folder)
                setattr(args, "log_dir", log_dir)
                dataloader = load_data(args,small_sample=n)
                pinn = PINN(args)
                pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'],
                           testloader=dataloader['test'])

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for XJTU dataset')
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--train_batch', type=int, default=0, choices=[-1,0,1,2,3,4,5],
                        help='如果是-1，读取全部数据，并随机划分训练集和测试集;否则，读取对应的batch数据'
                             '(if -1, read all data and random split train and test sets; '
                             'else, read the corresponding batch data)')
    parser.add_argument('--test_batch', type=int, default=1, choices=[-1,0,1,2,3,4,5],
                        help='如果是-1，读取全部数据，并随机划分训练集和测试集;否则，读取对应的batch数据'
                             '(if -1, read all data and random split train and test sets; '
                             'else, read the corresponding batch data)')
    parser.add_argument('--batch',type=str,default='2C',choices=['2C','3C','R2.5','R3','RW','satellite'])
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=120, help='epoch')
    parser.add_argument('--early_stop', type=int, default=20, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=0.002, help='warmup lr')
    parser.add_argument('--lr', type=float, default=0.01, help='base lr')
    parser.add_argument('--final_lr', type=float, default=0.0002, help='final lr')
    parser.add_argument('--lr_F', type=float, default=0.001, help='lr of F')

    # model related
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

    # loss related
    parser.add_argument('--alpha', type=float, default=0.7, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta', type=float, default=20, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    parser.add_argument('--log_dir', type=str, default='text log.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='results of reviewer/XJTU results', help='save folder')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print("reached here")
    main()

