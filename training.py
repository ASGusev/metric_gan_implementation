import numpy as np
import torch
from pystoi import stoi
from tqdm import tqdm

import data_utils


def sg_ph_stoi(c_sg, c_ph, n_sg, n_ph):
    c_w = data_utils.sg_to_wav(c_sg, c_ph)
    n_w = data_utils.sg_to_wav(n_sg, n_ph)
    return stoi(c_w, n_w, data_utils.SR)


def train_discriminator(gen, disc, disc_opt, data, device):
    losses = []
    for (noisy_sg, noisy_ph), (clean_sg, clean_ph) in tqdm(data):
        disc_opt.zero_grad()
        noisy_sg, clean_sg = torch.tensor(noisy_sg, device=device), torch.tensor(clean_sg, device=device)
        with torch.no_grad():
            mask = gen(noisy_sg.reshape((1,) + noisy_sg.shape[::-1]))
            pred_sg = mask.reshape((mask.shape[2], mask.shape[1])) * noisy_sg
        pred_gen_stoi = disc(pred_sg.reshape((1, 1) + pred_sg.shape), clean_sg.reshape((1, 1) + clean_sg.shape))
        real_gen_stoi = sg_ph_stoi(clean_sg.cpu().numpy(), clean_ph, pred_sg.cpu().numpy(), noisy_ph)
        pred_clean_stoi = disc(clean_sg.reshape((1, 1) + clean_sg.shape), clean_sg.reshape((1, 1) + clean_sg.shape))

        loss = (pred_gen_stoi - torch.tensor(real_gen_stoi)) ** 2 + (pred_clean_stoi - 1) ** 2
        loss.backward()
        disc_opt.step()
        losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)


def val_discriminator(gen, disc, data, device):
    losses = []
    with torch.no_grad():
        for (noisy_sg, noisy_ph), (clean_sg, clean_ph) in tqdm(data):
            noisy_sg, clean_sg = torch.tensor(noisy_sg, device=device), torch.tensor(clean_sg, device=device)
            mask = gen(noisy_sg.reshape((1,) + noisy_sg.shape[::-1]))
            pred_sg = mask.reshape((mask.shape[2], mask.shape[1])) * noisy_sg
            pred_gen_stoi = disc(pred_sg.reshape((1, 1) + pred_sg.shape), clean_sg.reshape((1, 1) + clean_sg.shape))
            real_gen_stoi = sg_ph_stoi(clean_sg.cpu().numpy(), clean_ph, pred_sg.cpu().numpy(), noisy_ph)
            pred_clean_stoi = disc(clean_sg.reshape((1, 1) + clean_sg.shape), clean_sg.reshape((1, 1) + clean_sg.shape))
            loss = (pred_gen_stoi - torch.tensor(real_gen_stoi)) ** 2 + (pred_clean_stoi - 1) ** 2
            losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)


def train_generator(gen, disc, gen_opt, data, device, target_stoi):
    losses = []
    for (noisy_sg, noisy_ph), (clean_sg, clean_ph) in tqdm(data):
        gen_opt.zero_grad()
        noisy_sg, clean_sg = torch.tensor(noisy_sg, device=device), torch.tensor(clean_sg, device=device)
        mask = gen(noisy_sg.reshape((1,) + noisy_sg.shape[::-1]))
        pred_sg = mask.reshape((mask.shape[2], mask.shape[1])) * noisy_sg
        pred_gen_stoi = disc(pred_sg.reshape((1, 1) + pred_sg.shape), clean_sg.reshape((1, 1) + clean_sg.shape))
        loss = (pred_gen_stoi - target_stoi) ** 2
        loss.backward()
        gen_opt.step()
        losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)


def val_generator(gen, disc, data, device, target_stoi):
    losses = []
    real_stois = []
    with torch.no_grad():
        for (noisy_sg, noisy_ph), (clean_sg, clean_ph) in tqdm(data):
            noisy_sg = torch.tensor(noisy_sg, device=device)
            mask = gen(noisy_sg.reshape((1,) + noisy_sg.shape[::-1]))
            pred_sg = mask.reshape((mask.shape[2], mask.shape[1])) * noisy_sg

            real_gen_stoi = sg_ph_stoi(clean_sg, clean_ph, pred_sg.cpu().numpy(), noisy_ph)
            clean_sg = torch.tensor(clean_sg, device=device)
            pred_gen_stoi = disc(pred_sg.reshape((1, 1) + pred_sg.shape), clean_sg.reshape((1, 1) + clean_sg.shape))
            loss = (pred_gen_stoi - target_stoi) ** 2
            real_stois.append(real_gen_stoi)
            losses.append(loss.cpu().numpy())
    return np.mean(losses), np.mean(real_stois)


def train_gan(gen, disc, gen_opt, disc_opt, tr_data, val_data, n_epochs, device, target_stoi=1):
    for epoch in range(1, n_epochs + 1):
        tr_disc_loss = train_discriminator(gen, disc, disc_opt, tr_data, device)
        val_disc_loss = val_discriminator(gen, disc, val_data, device)
        tr_gen_loss = train_generator(gen, disc, gen_opt, tr_data, device, target_stoi)
        val_gen_loss, val_gen_stoi = val_generator(gen, disc, val_data, device, target_stoi)
        print(f'Epoch {epoch} tr_disc_loss: {tr_disc_loss:.4f} val_disc_loss: {val_disc_loss:.4f} ' +
              f'tr_gen_loss: {tr_gen_loss:.4f} val_gen_loss: {val_gen_loss:.4f}, gen_stoi: {val_gen_stoi:.4f}')
