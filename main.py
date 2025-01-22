# -*- coding = utf-8 -*-
# @Time : 1/20/25 10:26
# @Author : Tracy
# @File : main.py
# @Software : PyCharm

import model.TCNGAN
import model.data_loader
import train
import utils.visualize

import torch.utils.data
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler






if __name__ == '__main__':
    path = "/Users/tracy/Desktop/留学/正在学/DS/DL/CV/QuantGAN/Project01/data/ShanghaiSE_daily.csv"
    df_stocks = pd.read_csv(path)
    df_stocks['Date'] = pd.to_datetime(df_stocks['Date'])
    df_stocks['Return_log'] = np.log(df_stocks['CLOSE'].shift(1) / df_stocks['CLOSE'])

    df_ret = df_stocks['Return_log'].fillna(0)

    scaler1 = StandardScaler()
    df_ret_normed = scaler1.fit_transform(df_ret.values.reshape(-1, 1))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    generator = model.TCNGAN.Generator()
    discriminator = model.TCNGAN.Discriminator(seq_len=127)

    dataset = model.data_loader.FTDataset(df_ret_normed, seq_len=127)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=80)

    trainer = train.TCNGANTrainer(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        nz=1,
        clip=0.01,
        lr=0.0002,
        device=device,
        generator_path="/Users/tracy/Desktop/留学/正在学/DS/DL/CV/QuantGAN/Project01/checkpoints/",
        file_name="TCNGAN_net_1"
    )

    trainer.train(num_epochs=3)

    generator.eval()
    noise = torch.randn(80,1,127).to(device)
    y_tensor = generator(noise).cpu().detach().squeeze();

    y_tensor = (y_tensor - y_tensor.mean(axis=0))/y_tensor.std(axis=0)
    y = y_tensor.detach().cpu().numpy()

    y = scaler1.inverse_transform(y_tensor.numpy())

    y = y[(y.max(axis=1) <= 2 * df_ret.max()) & (y.min(axis=1) >= 2 * df_ret.min())]
    y -= y.mean(axis=0)

    fig_path = "/Users/tracy/Desktop/留学/正在学/DS/DL/CV/QuantGAN/Project01/checkpoints/"
    utils.visualize.visualize_return(y, fig_path)
    utils.visualize.visualize_window(df_ret, y, [1, 5, 20, 100], fig_path)





