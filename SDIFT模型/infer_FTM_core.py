import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import scipy.io as scio
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from train_FTM import evaluating, loss_fn, loss_fn2
from utils import load_large_data, total_variation_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def training_step(basis_function, tucker_core, train_loader, optimizer, ind_input, mask_tr, device):
    basis_function.eval()
    basis_function.mode = "training"
    loss_list = []
    for data, batch_ind in train_loader:
        data = data.to(device, non_blocking=True)
        batch_ind = batch_ind.to(device, non_blocking=True)
        mask_tmp = mask_tr.unsqueeze(0).repeat(data.shape[0], 1, 1, 1, 1)
        optimizer.zero_grad()

        basises = basis_function(input_ind_train=ind_input)
        output = torch.einsum("mi, btijk->btmjk", basises[0], tucker_core[batch_ind, :, :, :, :])
        output = torch.einsum("nj, btmjk->btmnk", basises[1], output)
        output = torch.einsum("ok, btmnk->btmno", basises[2], output)
        loss = loss_fn(output, data, mask=mask_tmp) + total_variation_loss(tucker_core[batch_ind, :, :, :, :], weight=1e-7)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    return float(np.mean(loss_list))


def main():
    parser = argparse.ArgumentParser(description="Infer FTM core for a dataset using a frozen basis model.")
    parser.add_argument("--basis_path", required=True)
    parser.add_argument("--data_name", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--R", type=int, nargs=3, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--max_iter", type=int, default=600)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--out_core_path", default="")
    parser.add_argument("--out_json", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ind_uni, data_extract, mask_tr_np = load_large_data(args.data_path, args.metadata_path)
    u_ind_uni = torch.FloatTensor(ind_uni[0]).to(device)
    v_ind_uni = torch.FloatTensor(ind_uni[1]).to(device)
    w_ind_uni = torch.FloatTensor(ind_uni[2]).to(device)
    ind_input = (u_ind_uni, v_ind_uni, w_ind_uni)
    mask_tr = torch.FloatTensor(mask_tr_np).to(device)

    basis_function = safe_torch_load(args.basis_path, device)
    basis_function = basis_function.to(device)
    basis_function.eval()
    basis_function.mode = "training"
    for p in basis_function.parameters():
        p.requires_grad = False

    data = torch.FloatTensor(data_extract)
    batch_ind = torch.arange(data_extract.shape[0])
    loader = DataLoader(TensorDataset(data, batch_ind), batch_size=args.batch_size, shuffle=False)

    R = tuple(args.R)
    tucker_core = (torch.ones(data_extract.shape[0], data_extract.shape[1], R[0], R[1], R[2]) / 2.0).to(device)
    tucker_core.requires_grad = True
    optimizer = optim.AdamW([tucker_core], lr=args.learning_rate)

    best_rmse = float("inf")
    history = []
    for iteration in tqdm(range(args.max_iter), desc="infer_core"):
        train_loss = training_step(basis_function, tucker_core, loader, optimizer, ind_input, mask_tr, device)
        if iteration % max(1, args.save_every) == 0 or iteration == args.max_iter - 1:
            rmse, mae = evaluating(
                basis_function,
                tucker_core,
                loader,
                loss_fn,
                loss_fn2,
                device_override=device,
                ind_input_override=ind_input,
            )
            history.append({"iter": int(iteration), "train_loss": float(train_loss), "rmse": float(rmse), "mae": float(mae)})
            if rmse < best_rmse:
                best_rmse = rmse

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    out_core_path = args.out_core_path or os.path.join("./data", f"core_{args.data_name}_{R[0]}x{R[1]}x{R[2]}_{timestamp}_encoded.mat")
    os.makedirs(os.path.dirname(out_core_path) or ".", exist_ok=True)
    scio.savemat(out_core_path, {"core": tucker_core.detach().cpu().numpy()})

    summary = {
        "basis_path": args.basis_path,
        "data_name": args.data_name,
        "data_path": args.data_path,
        "metadata_path": args.metadata_path,
        "R": list(R),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "max_iter": int(args.max_iter),
        "seed": int(args.seed),
        "best_rmse": float(best_rmse),
        "history": history,
        "out_core_path": out_core_path,
    }
    out_json = args.out_json or os.path.splitext(out_core_path)[0] + "_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"saved: {out_core_path}")
    print(f"saved: {out_json}")


if __name__ == "__main__":
    main()
