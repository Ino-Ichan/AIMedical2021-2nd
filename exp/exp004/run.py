import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import random
import os

from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

import pickle

import warnings


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, mode="train", add_noise=False):
        self.df = df
        self.mode = mode
        self.human_auto_w = [1, 1]
        self.add_noise = add_noise
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ecg = np.load(row.ecg_path)
        if self.add_noise:
            ecg = ecg + np.random.random([800, 12]) * 0.01
        age = row.age
        sex = 0 if row.sex == 'male' else 1
        label_type = self.human_auto_w[0] if row.sex == 'human' else self.human_auto_w[1]
        
        if self.mode == "train":
            return {
                'ecg': torch.tensor(ecg, dtype=torch.float).permute(1, 0),
                'age': torch.tensor(age, dtype=torch.float),
                'sex': torch.tensor(sex, dtype=torch.float),
                'target': torch.tensor(row.target, dtype=torch.float),
                'label_type': torch.tensor(label_type, dtype=torch.float),
            }
        else:
            return {
                'ecg': torch.tensor(ecg, dtype=torch.float).permute(1, 0),
                'age': torch.tensor(age, dtype=torch.float),
                'sex': torch.tensor(sex, dtype=torch.float),
            }


class PreDataset(torch.utils.data.Dataset):
    def __init__(self, df, mode="train", add_noise=False):
        self.df = df
        self.mode = mode
        self.human_auto_w = [1, 1]
        self.add_noise = add_noise
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ecg = np.load(row.ecg_path)
        noise_ecg = ecg + np.random.random([800, 12]) * 0.001
        age = row.age
        sex = 0 if row.sex == 'male' else 1
        label_type = self.human_auto_w[0] if row.sex == 'human' else self.human_auto_w[1]
        
        return {
            'ecg': torch.tensor(ecg, dtype=torch.float).permute(1, 0),
            'noise_ecg': torch.tensor(ecg, dtype=torch.float).permute(1, 0),
            'age': torch.tensor(age, dtype=torch.float),
            'sex': torch.tensor(sex, dtype=torch.float),
        }


class AIMedicalModel(nn.Module):
    def __init__(self, in_channel=12, ninp=512, nhead=2, nhid=1024, dropout=0.1, nlayers=1):
        super(AIMedicalModel, self).__init__()

        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer = nn.TransformerEncoder(encoder_layers, nlayers)
        
        d_model = ninp
        self.top_layer1 = nn.Sequential(
                             nn.Conv1d(in_channel, d_model//4, kernel_size=3, padding=1),
                             nn.ReLU(inplace=True),
                         )
        
        self.top_layer2 = nn.Sequential(
                             nn.Conv1d(in_channel, d_model//4, kernel_size=5, padding=2),
                             nn.ReLU(inplace=True),
                         )
        
        self.top_layer3 = nn.Sequential(
                             nn.Conv1d(in_channel, d_model//4, kernel_size=7, padding=3),
                             nn.ReLU(inplace=True),
                         )
        
        self.top_layer4 = nn.Sequential(
                             nn.Conv1d(in_channel, d_model//4, kernel_size=11, padding=5),
                             nn.ReLU(inplace=True),
                         )
        
        self.top_conv = nn.Sequential(
                             nn.Conv1d(d_model, d_model, kernel_size=1),
                             nn.ReLU(inplace=True),
                         )
        
        
#         self.bottom_layer = torch.nn.Conv1d(d_model, 1, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model//2, num_layers=1,
                                    batch_first=True, bidirectional=True)
        
        self.bottom_linear = nn.Sequential(
                                 nn.Linear(d_model, d_model//2),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(dropout),
                                 nn.Linear(d_model//2, 1)
                             )

        self.bottom_linear_ae = nn.Sequential(
                                        nn.Linear(d_model, d_model//2),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(dropout),
                                        nn.Linear(d_model//2, 12)
                                    )

    def forward(self, x):
        x = x.squeeze(1)
        x1 = self.top_layer1(x)
        x2 = self.top_layer2(x)
        x3 = self.top_layer3(x)
        x4 = self.top_layer4(x)
        x = torch.cat([x1, x2, x3, x4], axis=1)
        x = self.top_conv(x).transpose(1, 2)
        x = self.transformer(x)
        # x_t, _ = self.lstm(x)
        x = x_t.transpose(1, 2)
        x = F.max_pool1d(x, kernel_size=x.size()[2:])
        x = x.view(x.size()[0], -1)
        x = self.bottom_linear(x).squeeze(-1)

        # ae
        out_ae = self.bottom_linear_ae(x_t)
        return x, out_ae


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_top3_model(model, pred, score, epoch, cv, save_dict):
    # score[順位]: [val_score, path, epoch, val_pred] 
    top1_list = save_dict["cv{}_top1".format(cv)]
    top2_list = save_dict["cv{}_top2".format(cv)]
    top3_list = save_dict["cv{}_top3".format(cv)]
    
    if score > top1_list[0]:
        # これまでの上位を一つずらす
        save_dict["cv{}_top3".format(cv)] = [top2_list[0], top3_list[1], top2_list[2], top2_list[3]]
        if os.path.exists(top2_list[1]):
            os.rename(top2_list[1], top3_list[1])
        
        save_dict["cv{}_top2".format(cv)] = [top1_list[0], top2_list[1], top1_list[2], top1_list[3]]
        if os.path.exists(top1_list[1]):
            os.rename(top1_list[1], top2_list[1])

        save_dict["cv{}_top1".format(cv)] = [score, top1_list[1], epoch, np.array(pred)]
        torch.save(model.state_dict(), top1_list[1])
    elif score > top2_list[0]:
        # これまでの上位を一つずらす
        save_dict["cv{}_top3".format(cv)] = [top2_list[0], top3_list[1], top2_list[2], top2_list[3]]
        if os.path.exists(top2_list[1]):
            os.rename(top2_list[1], top3_list[1])
        
        save_dict["cv{}_top2".format(cv)] = [score, top2_list[1], epoch, np.array(pred)]
        torch.save(model.state_dict(), top2_list[1])
    elif score > top3_list[0]:
        save_dict["cv{}_top3".format(cv)] = [score, top3_list[1], epoch, np.array(pred)]
        torch.save(model.state_dict(), top3_list[1])
    return save_dict

if __name__ == "__main__":
    warnings.simplefilter('ignore')
    warnings.filterwarnings('ignore')
    seed_everything(516)

    df = pd.read_csv('/workspace/data/df_train.csv')
    df_test = pd.read_csv('/workspace/data/df_test.csv')

    device = "cuda:0" if torch.cuda.is_available() else "cpu"


    # # ==================================================================================
    # # Pretrain
    # # ==================================================================================
    # df_all = pd.concat([df, df_test]).reset_index(drop=True)
    # model = AIMedicalModel().to(device)
    # params = [{"params": model.parameters()}]
    
    # optimizer = Adam(params, lr=1e-3)
    # criterion = nn.MSELoss()

    # dataset = PreDataset(df_all)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=False, drop_last=True)
    
    # for e in range(1, 20):
    #     train_loss = 0
    #     # training
    #     model.train()
    #     for data in dataloader:
    #         inp = data["noise_ecg"].to(device, dtype=torch.float)
    #         target = data["ecg"].to(device, dtype=torch.float).permute(0, 2, 1)
    #         optimizer.zero_grad()
            
    #         out, out_ae = model(inp)

    #         # human, autoでweightをかける
    #         loss = criterion(out_ae, target)
    #         loss.backward()
    #         train_loss += loss.item()
    #         optimizer.step()
    #     print(f'## epoch {e}, train_loss: {train_loss:.5f}')
    # torch.save(model.state_dict(), './pretrain.pth')

    # # ================================================
    # # Pretrain ここまで
    # # ================================================

    epoch = 100

    save_top_3_dict = {}

    oof = np.zeros(len(df))

    criterion = nn.BCEWithLogitsLoss(reduction='none')

    for i in range(5):
    #     if i == 1:
    #         break
            
        print('-'*30)
        print('Start CV{}'.format(i))

        save_dir_path = f"./cv{i}"
        if os.path.exists(save_dir_path) == False:
            os.makedirs(save_dir_path)
        save_top_3_dict_cv = {
            # score[順位]: [val_score, path, epoch, val_pred]
            "cv{}_top1".format(i): [0.00, "{}/best1.pth".format(save_dir_path), 0, None],
            "cv{}_top2".format(i): [0.00, "{}/best2.pth".format(save_dir_path), 0, None],
            "cv{}_top3".format(i): [0.00, "{}/best3.pth".format(save_dir_path), 0, None],
        }
        save_top_3_dict.update(save_top_3_dict_cv)


        # dataset 作成
        df_train = df[df.cv != i].reset_index(drop=True)
        df_val = df[df.cv == i].reset_index(drop=True)
        # index per label type
        human_index = df_val[df_val.label_type == 'human'].index
        auto_index = df_val[df_val.label_type == 'auto'].index
        val_index = df[df.cv == i].index

        # dataset. dataloader
        train_dataset = Dataset(df_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=False, drop_last=True)
        
        val_dataset = Dataset(df_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=False, drop_last=False)
        
        model = AIMedicalModel().to(device)
        # model.load_state_dict(torch.load('./pretrain.pth'))
        params = [{"params": model.parameters()}]
        
        optimizer = Adam(params, lr=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-5)
        
        # class balanced loss parameter
        cv_train_loss = []
        cv_train_score = []
        cv_val_loss = []
        cv_val_score = []
        
        for e in range(1, epoch+1):
    #         if e == 2:
    #             break
                
            start_time = time()
            train_loss = 0
            val_loss = 0
            
            train_out_list = []
            train_target_list = []
            
            val_out_list = []
            val_target_list = []
            
                
            # training
            model.train()
            for data in train_loader:
                inp = data["ecg"].to(device, dtype=torch.float)
                target = data["target"].to(device, dtype=torch.float)
                h_a_weight = data["label_type"].to(device, dtype=torch.float)
                
                optimizer.zero_grad()
                
                out, _ = model(inp)

                # human, autoでweightをかける
                loss = (criterion(out, target) * h_a_weight).mean()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                scheduler.step()
                
                train_out_list += list(out.sigmoid().data.cpu().numpy())
                train_target_list += list(target.data.cpu().numpy())
                
            cv_train_score.append(roc_auc_score(np.array(train_target_list), np.array(train_out_list)))
            cv_train_loss.append(train_loss)
            
            
            
            # val
            model.eval()
            with torch.no_grad():
                for data in val_loader:
                    inp = data["ecg"].to(device, dtype=torch.float)
                    target = data["target"].to(device, dtype=torch.float)
                    h_a_weight = data["label_type"].to(device, dtype=torch.float)

                    out, _ = model(inp)
                    # human, autoでweightをかける
                    loss = (criterion(out, target) * h_a_weight).mean()
                    val_loss += loss.item()

                    val_out_list += list(out.sigmoid().data.cpu().numpy())
                    val_target_list += list(target.data.cpu().numpy())
                
            val_score = roc_auc_score(np.array(val_target_list), np.array(val_out_list))
            val_score_human = roc_auc_score(np.array(val_target_list)[human_index], np.array(val_out_list)[human_index])
            val_score_auto = roc_auc_score(np.array(val_target_list)[auto_index], np.array(val_out_list)[auto_index])
            save_top_3_dict = save_top3_model(model=model, pred=val_out_list, score=val_score,
                                            cv=i, epoch=e, save_dict=save_top_3_dict)
            cv_val_score.append(val_score)
            cv_val_loss.append(val_loss)
            
            print(f'## epoch {e}, {time()-start_time/60:.2f} min  train_loss: {train_loss:.5f}, val_loss: {val_loss:.5f},  train_score: {cv_train_score[-1]:.5f}, val_score: {val_score:.5f}, val_score human: {val_score_human:.5f}, val_score auto: {val_score_auto:.5f}')
        
        ### oofの作成
        oof[val_index] = (save_top_3_dict["cv{}_top1".format(i)][3]\
                            +save_top_3_dict["cv{}_top1".format(i)][3]\
                                +save_top_3_dict["cv{}_top1".format(i)][3])/3

    os.makedirs("./save_dict", exist_ok=True)
    with open('./save_dict/{}.pickle', "wb") as f:
        pickle.dump(save_top_3_dict, f)
        
    with open('./save_dict/{}.pickle', "rb") as f:
        save_top_3_dict = pickle.load(f)

    cv_mean = []
    for _, l in save_top_3_dict.items():
        cv_mean.append(l[0])
    print("Mean CV score: ", np.mean(cv_mean))
    print('-'*10)
    for i in range(5):
        print('Just Mean CV{} score: '.format(i), np.mean(cv_mean[i*3:i*3+3]))
    print('-'*10)
    print("Oof CV score", roc_auc_score(df.target, oof))

    result_pred = []

    print("** Start Inference **")

    for i, save_key in enumerate(save_top_3_dict.keys()):
        print('## ', save_key)
        
        out_list = []
        
        # load model
        model_path = save_top_3_dict[save_key][1]
        model.load_state_dict(torch.load(model_path))
        model.eval()
        with torch.no_grad():
            for data_num in range(len(df_test)):
                inp = np.load(df_test.ecg_path.iloc[data_num])
                inp = torch.tensor(inp.astype(np.float32))
                inp = inp.to(device, dtype=torch.float).permute(1, 0).unsqueeze(0)

                out, _ = model(inp)
                out_list += list(out.sigmoid().data.cpu().numpy())

        result_pred.append(out_list)

    result_pred = np.array(result_pred)

    score = result_pred.mean(axis=0)

    sample_sub = pd.read_csv('/workspace/data/sample_submission.csv')
    sample_sub.target = score
    sample_sub.to_csv("./submit.csv", index=False)

    df_oof = df.copy()
    df_oof['pred'] = oof
    df_oof.to_csv('./oof.csv', index=False)