import argparse
import json
import os
import random
import time

import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_meta(meta_path):
    d = np.load(meta_path, allow_pickle=True).item()["data"]
    u_uni = d["u_ind_uni"]
    v_uni = d["v_ind_uni"]
    w_uni = d["w_ind_uni"]
    t_uni = d["t_ind_uni"]
    u_real = d.get("u_ind_real", u_uni)
    v_real = d.get("v_ind_real", v_uni)
    w_real = d.get("w_ind_real", w_uni)
    t_real = d.get("t_ind_real", t_uni)
    return u_uni, v_uni, w_uni, t_uni, u_real, v_real, w_real, t_real


def load_sensor_observations(sensor_path):
    d = np.load(sensor_path, allow_pickle=True).item()
    sensor_xyz = d["sensor_xyz"]
    t_vals = d["t"]
    y = d["y"]
    sensor_idx = d.get("sensor_idx", None)

    y_group = []
    ind_conti_group = []
    y_time_group = []
    y_time_ind_group = []
    for i in range(t_vals.shape[0]):
        y_group.append(y[i])
        ind_conti_group.append(sensor_xyz)
        y_time_group.append(t_vals[i])
        y_time_ind_group.append(i)

    return y_group, ind_conti_group, y_time_group, y_time_ind_group, sensor_idx, t_vals


def build_sensor_mask(sensor_idx, shape):
    if sensor_idx is None:
        return None
    mask = np.zeros(shape, dtype=np.int8)
    for idx in sensor_idx:
        mask[:, idx[0], idx[1], idx[2]] = 1
    return mask


def estimate_leak_source(field, u_axis, v_axis, w_axis, time_window=5, radius=1):
    if field.ndim != 4:
        raise ValueError("field must be [T, U, V, W]")
    t_window = min(time_window, field.shape[0])
    avg_field = field[:t_window].mean(axis=0)
    max_idx = np.unravel_index(np.argmax(avg_field), avg_field.shape)

    u0 = max(0, max_idx[0] - radius)
    u1 = min(avg_field.shape[0], max_idx[0] + radius + 1)
    v0 = max(0, max_idx[1] - radius)
    v1 = min(avg_field.shape[1], max_idx[1] + radius + 1)
    w0 = max(0, max_idx[2] - radius)
    w1 = min(avg_field.shape[2], max_idx[2] + radius + 1)
    local_mean = float(avg_field[u0:u1, v0:v1, w0:w1].mean())
    peak = float(avg_field[max_idx])

    return {
        "index": [int(max_idx[0]), int(max_idx[1]), int(max_idx[2])],
        "position": [
            float(u_axis[max_idx[0]]),
            float(v_axis[max_idx[1]]),
            float(w_axis[max_idx[2]]),
        ],
        "strength": local_mean,
        "peak": peak,
        "time_window": int(t_window),
        "radius": int(radius),
    }


def load_leak_rate_calibration(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "type" not in cfg:
        raise ValueError("leak-rate calibration must contain 'type'")
    return cfg


def estimate_leak_rate(strength, calibration):
    if calibration is None:
        return None
    ctype = calibration.get("type", "").lower()
    if ctype == "linear":
        a = float(calibration.get("a", 1.0))
        b = float(calibration.get("b", 0.0))
        return a * float(strength) + b
    if ctype == "power":
        a = float(calibration.get("a", 1.0))
        p = float(calibration.get("p", 1.0))
        b = float(calibration.get("b", 0.0))
        return a * (max(float(strength), 0.0) ** p) + b
    raise ValueError(f"Unsupported leak-rate calibration type: {calibration.get('type')}")


def get_ktT(y_tt, core_t, gp_gamma=1, gp_sigma=1):
    r = torch.sqrt(torch.square(y_tt - core_t))
    return gp_sigma * torch.exp(-torch.square(r) * gp_gamma)


def get_kTT_inv(t, gp_gamma=1, gp_sigma=1):
    r = torch.sqrt(torch.square(t - t.transpose(-1, -2)))
    diag = torch.eye(t.shape[-2]).to(t) * 1e-3
    k = gp_sigma * torch.exp(-torch.square(r) * gp_gamma) + diag
    return torch.inverse(k)


def compute_continuous_poest(x_0, basis_function, core_mean, core_std, core_t, y_group, ind_conti_group, y_time_group, y_time_ind_group):
    core_tensor_shape = x_0.shape
    x_0 = x_0.view(core_tensor_shape[0], core_tensor_shape[1], -1)
    poest_matrix1 = torch.zeros_like(x_0).to(device)
    poest_matrix2 = torch.zeros_like(x_0).to(device)
    x_0 = x_0 * core_std + core_mean

    for y, y_tt, y_t_ind, ind in zip(y_group, y_time_group, y_time_ind_group, ind_conti_group):
        if y.shape[0] == 0:
            continue

        x_0_t = x_0[0, y_t_ind, :]
        y = torch.DoubleTensor(y).to(device)
        ind_conti = torch.FloatTensor(ind).to(device)
        a = basis_function(input_ind_sampl=ind_conti).detach().double()
        poest_matrix1[0, y_t_ind, :] = (a.T @ (y - a @ x_0_t))

        t_remove_group = y_time_ind_group.copy()
        t_remove_group.remove(y_t_ind)
        core_t_remove = core_t[:, t_remove_group, :]
        x_0_remove = x_0[:, t_remove_group, :]

        ktT = get_ktT(y_tt, core_t_remove).squeeze(2)
        kTT_inv = get_kTT_inv(core_t_remove)
        coeff = (ktT @ kTT_inv).to(device).squeeze(1)

        x_0_aggregate = (coeff @ x_0_remove).squeeze()
        post = (a.T @ (y - a @ x_0_aggregate))
        temp = torch.zeros_like(x_0).to(device)
        temp[:, t_remove_group, :] = torch.kron(coeff.unsqueeze(2), post.unsqueeze(0).unsqueeze(0))
        poest_matrix2 += temp

    return (poest_matrix1 + MPDPS * poest_matrix2).view(core_tensor_shape)


@torch.no_grad()
def edm_post_sampler(
    edm, basis_function, latents, t, y_group, ind_conti_group, y_time_group, y_time_ind_group,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7, use_ema=False,
):
    sigma_min = max(sigma_min, edm.sigma_min)
    sigma_max = min(sigma_max, edm.sigma_max)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    i_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    i_steps = torch.cat([edm.round_sigma(i_steps), torch.zeros_like(i_steps[:1])])

    x_next = latents.to(torch.float64) * i_steps[0]
    print("sampling")
    t_start = time.time()

    for i, (i_cur, i_next) in tqdm(enumerate(zip(i_steps[:-1], i_steps[1:]))):
        x_hat = x_next
        i_hat = i_cur

        denoised = edm(x_hat, i_hat, t, use_ema=use_ema).to(torch.float64)
        denoised_core1 = denoised.detach().clone()
        d_cur = (x_hat - denoised) / i_hat
        x_next = x_hat + (i_next - i_hat) * d_cur

        if i < num_steps - 1:
            denoised = edm(x_next, i_next, t, use_ema=use_ema).to(torch.float64)
            denoised_core2 = denoised.detach().clone()
            d_prime = (x_next - denoised) / i_next
            x_next = x_hat + (i_next - i_hat) * (0.5 * d_cur + 0.5 * d_prime)
            denoised_core = (denoised_core1 + denoised_core2) / 2
            llk_grad = compute_continuous_poest(
                denoised_core, basis_function, core_mean, core_std, t,
                y_group, ind_conti_group, y_time_group, y_time_ind_group,
            )
            x_next = x_next + (zeta / (i + 1)) * llk_grad

    print(f"Elapsed time: {time.time() - t_start:.4f} seconds")
    print("sampling ", num_steps, " steps completed")
    return x_next


def create_mask(shape, r):
    num_ones = int(np.prod(shape) * r)
    mask = np.zeros(np.prod(shape), dtype=int)
    ones_indices = np.random.choice(np.prod(shape), num_ones, replace=False)
    mask[ones_indices] = 1
    return mask.reshape(shape)


def get_te_observations(d, rho=0.02, mode=1, ind=0):
    d_shape = d.shape
    data_extract = d[ind]
    mask = create_mask(data_extract.shape, rho)
    if mode == 2:
        mask[1::2, :, :, :] = 0

    t_ind_uni = np.linspace(0, d_shape[1] - 1, d_shape[1]).astype(int) / (d_shape[1] - 1)
    if d_shape[2] == 1:
        u_ind_uni = np.ones_like([1])
    else:
        u_ind_uni = np.linspace(0, d_shape[2] - 1, d_shape[2]).astype(int) / (d_shape[2] - 1)
    v_ind_uni = np.linspace(0, d_shape[3] - 1, d_shape[3]).astype(int) / (d_shape[3] - 1)
    w_ind_uni = np.linspace(0, d_shape[4] - 1, d_shape[4]).astype(int) / (d_shape[4] - 1)

    indices_ob = np.where(mask == 1)
    ob_ind_conti = np.array([t_ind_uni[indices_ob[0]], u_ind_uni[indices_ob[1]], v_ind_uni[indices_ob[2]], w_ind_uni[indices_ob[3]]]).T
    ob_ind = np.array(indices_ob).T
    ob_y = data_extract[indices_ob]
    ob_time_ind = ob_ind[:, 0]
    ob_conti = ob_ind_conti[:, 1:]
    y_group = []
    ind_conti_group = []
    y_time_group = []
    y_time_ind_group = []
    for i in range(t_ind_uni.shape[0]):
        y_temp = ob_y[ob_time_ind == i]
        y_group.append(y_temp + 0 * np.random.randn(*y_temp.shape))
        ind_conti_group.append(ob_conti[ob_time_ind == i])
        y_time_group.append(t_ind_uni[i])
        y_time_ind_group.append(i)

    return data_extract, mask, y_group, ind_conti_group, y_time_group, y_time_ind_group, u_ind_uni, v_ind_uni, w_ind_uni


def decoder(u_ind_uni, v_ind_uni, w_ind_uni, core, basis_function):
    u_ind_uni = torch.FloatTensor(u_ind_uni).to(device)
    v_ind_uni = torch.FloatTensor(v_ind_uni).to(device)
    w_ind_uni = torch.FloatTensor(w_ind_uni).to(device)

    ind_input = (u_ind_uni, v_ind_uni, w_ind_uni)
    basis_function.eval()
    prev_mode = getattr(basis_function, "mode", None)
    basis_function.mode = "training"

    core = core.to(torch.float32)
    basises = basis_function(input_ind_train=ind_input)
    output = torch.einsum("mi, tijk->tmjk", basises[0], core)
    output = torch.einsum("nj, tmjk->tmnk", basises[1], output)
    output = torch.einsum("ok, tmnk->tmno", basises[2], output)
    output = output.cpu().detach().numpy()

    if prev_mode is not None:
        basis_function.mode = prev_mode
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", type=str, default="sampling", help="experiment name")
    parser.add_argument("--dataset", type=str, default="am")
    parser.add_argument("--seed", default=123, type=int, help="global seed")
    parser.add_argument("--metadata_path", type=str, default="")
    parser.add_argument("--sensor_path", type=str, default="")
    parser.add_argument("--te_data_path", type=str, default="")
    parser.add_argument("--te_index", type=int, default=0)
    parser.add_argument("--source_time_window", type=int, default=5)
    parser.add_argument("--source_radius", type=int, default=1)
    parser.add_argument("--core_mean_std_path", type=str, default="")
    parser.add_argument("--basis_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--leak_rate_calibration_json", type=str, default="")
    parser.add_argument("--leak_rate_unit", type=str, default="mL/min")
    parser.add_argument("--source_gt", type=float, nargs=3, default=None)
    parser.add_argument("--leak_rate_gt", type=float, default=None)

    parser.add_argument("--gt_guide_type", default="l2", type=str, help="gt_guide_type loss type")
    parser.add_argument("--sigma_min", default=0.002, type=float, help="sigma_min")
    parser.add_argument("--sigma_max", default=80.0, type=float, help="sigma_max")
    parser.add_argument("--rho", default=7.0, type=float, help="Schedule hyper-parameter")
    parser.add_argument("--sigma_data", default=0.5, type=float, help="sigma_data used in EDM for c_skip and c_out")
    parser.add_argument("--total_steps", default=20, type=int, help="total_steps")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--sample_mode", type=str, default="fid", help="sample mode")
    parser.add_argument("--begin_ckpt", default=0, type=int, help="begin_ckpt")

    parser.add_argument("--img_size", type=int, default=48)
    parser.add_argument("--img_size_3d", type=int, nargs=3, default=None, help="D H W for 3D mode")
    parser.add_argument("--channels", default=1, type=int, help="input_output_channels")
    parser.add_argument("--spatial_dims", default=2, type=int, choices=[2, 3])
    parser.add_argument("--model_channels", default=40, type=int, help="model_channels")
    parser.add_argument("--channel_mult", default=[1, 2, 2], type=int, nargs="+", help="channel_mult")
    parser.add_argument("--attn_resolutions", default=[], type=int, nargs="+", help="attn_resolutions")
    parser.add_argument("--num_layers", default=4, type=int, help="num_layers")
    parser.add_argument("--layers_per_block", default=4, type=int, help="num_blocks")
    parser.add_argument("--num_temporal_latent", default=8, type=int, help="num_temporal_latent")
    parser.add_argument("--obs_rho", default=0.01, type=float, help="observation ratio for random sampling")
    parser.add_argument("--missing_type", default=1, type=int, help="missing type for random sampling")
    parser.add_argument("--mpdps_weight", default=0.4, type=float, help="MPDPS weight")
    parser.add_argument("--zeta", default=0.009, type=float, help="posterior step size")
    parser.add_argument("--num_posterior_samples", default=1, type=int, help="number of posterior samples per case")
    parser.add_argument("--posterior_seed_stride", default=9973, type=int, help="seed stride across posterior samples")
    parser.add_argument("--save_recon_samples", action="store_true", default=False)
    config = parser.parse_args()

    from train_GPSD import EDM, create_model, get_gp_covariance

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device
    if config.spatial_dims == 3 and config.channels != 1:
        print("3D mode expects channels=1; overriding channels to 1.")
        config.channels = 1

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    my_net = create_model(config)
    edm = EDM(model=my_net, cfg=config)
    torch.cuda.empty_cache()

    core_mean_std_path = config.core_mean_std_path or r"./exps/gp-edm_am_12M/core_mean_std.mat"
    d = sio.loadmat(core_mean_std_path)
    core_mean = torch.tensor(d["core_mean"], dtype=torch.float32).to(device)
    core_std = torch.tensor(d["core_std"], dtype=torch.float32).to(device)

    basis_path = config.basis_path or r"./ckp/basis4_am_2D_1x48x48_2025_05_04_14.pth"
    basis_function = torch.load(basis_path, map_location=device, weights_only=False)
    basis_function.eval()
    basis_function.mode = "sampling"

    model_path = config.model_path or r"./exps/gp-edm_am_12M/checkpoints/ema_10000.pth"
    checkpoint = torch.load(model_path, map_location=device)
    edm.model.load_state_dict(checkpoint)
    for param in edm.model.parameters():
        param.requires_grad = False
    edm.model.eval()

    rmse_list = []
    recon_list = []
    recon_samples_list = []
    mask_list = []
    case_summaries = []
    rho = config.obs_rho
    mt = config.missing_type
    MPDPS = config.mpdps_weight
    zeta = config.zeta
    leak_rate_calibration = load_leak_rate_calibration(config.leak_rate_calibration_json)

    use_sensor = bool(config.sensor_path)
    full_data = None
    if config.te_data_path:
        d = np.load(config.te_data_path, allow_pickle=True).item()
        full_data = d["data"]

    if use_sensor:
        if not config.metadata_path:
            raise ValueError("metadata_path is required when sensor_path is provided.")
        u_ind_uni, v_ind_uni, w_ind_uni, t_ind_uni, u_real, v_real, w_real, t_real = load_meta(config.metadata_path)
        y_group, ind_conti_group, y_time_group, y_time_ind_group, sensor_idx, t_vals = load_sensor_observations(config.sensor_path)
        mask = build_sensor_mask(sensor_idx, (t_vals.shape[0], len(u_ind_uni), len(v_ind_uni), len(w_ind_uni)))
        if mask is not None:
            mask_list.append(mask)
        num_cases = 1
        t_grid = torch.tensor(t_vals, dtype=torch.float64, device=device).view(1, -1, 1)
    else:
        if full_data is None:
            raise ValueError("te_data_path is required when sensor_path is not provided.")
        num_cases = full_data.shape[0]
        u_ind_uni = v_ind_uni = w_ind_uni = None
        u_real = v_real = w_real = None
        t_grid = None

    for i in tqdm(range(num_cases)):
        if use_sensor:
            data_extract = full_data[config.te_index] if full_data is not None else None
        else:
            data_extract, mask, y_group, ind_conti_group, y_time_group, y_time_ind_group, u_ind_uni, v_ind_uni, w_ind_uni = get_te_observations(full_data, rho=rho, mode=mt, ind=i)
            mask_list.append(mask)
            if config.metadata_path:
                u_ind_uni, v_ind_uni, w_ind_uni, t_ind_uni, u_real, v_real, w_real, t_real = load_meta(config.metadata_path)
                t_grid = torch.tensor(t_ind_uni, dtype=torch.float64, device=device).view(1, -1, 1)
            else:
                t_grid = torch.linspace(0, 1, data_extract.shape[0], device=device).view(1, -1, 1).double()

        if u_ind_uni is None or v_ind_uni is None or w_ind_uni is None:
            raise ValueError("u/v/w indices are not set; provide metadata_path.")

        sample_shape = [1, t_grid.shape[1], basis_function.r_1, basis_function.r_2, basis_function.r_3]
        cov_sample = get_gp_covariance(t_grid)
        l_sample = torch.linalg.cholesky(cov_sample).to(device)
        posterior_recons = []

        for s in range(config.num_posterior_samples):
            seed_s = config.seed + i * config.posterior_seed_stride + s
            torch.manual_seed(seed_s)
            np.random.seed(seed_s)
            random.seed(seed_s)
            basis_function.mode = "sampling"
            noise_sample = torch.randn(sample_shape).to(device).double()
            x_t = (l_sample @ noise_sample.view(sample_shape[0], sample_shape[1], -1)).view(sample_shape)
            sample = edm_post_sampler(
                edm,
                basis_function,
                x_t,
                t_grid,
                y_group,
                ind_conti_group,
                y_time_group,
                y_time_ind_group,
                num_steps=config.total_steps,
                use_ema=False,
            ).detach()
            core_sample = sample * core_std + core_mean
            out = decoder(u_ind_uni, v_ind_uni, w_ind_uni, core_sample[0], basis_function)
            posterior_recons.append(out.astype(np.float32))

        posterior_recons = np.stack(posterior_recons, axis=0)
        recon_mean = posterior_recons.mean(axis=0)
        recon_list.append(recon_mean.astype(np.float32))
        if config.save_recon_samples:
            recon_samples_list.append(posterior_recons.astype(np.float32))

        if data_extract is not None:
            rmse = np.sqrt(np.mean((recon_mean - data_extract) ** 2))
            rmse_list.append(rmse)
            print("RMSE:", rmse)

        if u_real is None or v_real is None or w_real is None:
            u_axis, v_axis, w_axis = u_ind_uni, v_ind_uni, w_ind_uni
        else:
            u_axis, v_axis, w_axis = u_real, v_real, w_real

        source_samples = []
        leak_rate_samples = []
        for s in range(posterior_recons.shape[0]):
            source = estimate_leak_source(
                posterior_recons[s],
                u_axis,
                v_axis,
                w_axis,
                time_window=config.source_time_window,
                radius=config.source_radius,
            )
            source_samples.append(source)
            q_hat = estimate_leak_rate(source["strength"], leak_rate_calibration)
            if q_hat is not None:
                leak_rate_samples.append(float(q_hat))

        source_pos = np.array([x["position"] for x in source_samples], dtype=np.float64)
        source_strength = np.array([x["strength"] for x in source_samples], dtype=np.float64)
        source_peak = np.array([x["peak"] for x in source_samples], dtype=np.float64)
        source_idx = np.array([x["index"] for x in source_samples], dtype=np.float64)
        low_q = 2.5
        high_q = 97.5

        source_summary = {
            "index": np.mean(source_idx, axis=0).tolist(),
            "position": np.mean(source_pos, axis=0).tolist(),
            "strength": float(np.mean(source_strength)),
            "peak": float(np.mean(source_peak)),
            "index_mean": np.mean(source_idx, axis=0).tolist(),
            "index_ci_low": np.percentile(source_idx, low_q, axis=0).tolist(),
            "index_ci_high": np.percentile(source_idx, high_q, axis=0).tolist(),
            "position_mean": np.mean(source_pos, axis=0).tolist(),
            "position_ci_low": np.percentile(source_pos, low_q, axis=0).tolist(),
            "position_ci_high": np.percentile(source_pos, high_q, axis=0).tolist(),
            "strength_mean": float(np.mean(source_strength)),
            "strength_ci_low": float(np.percentile(source_strength, low_q)),
            "strength_ci_high": float(np.percentile(source_strength, high_q)),
            "peak_mean": float(np.mean(source_peak)),
            "peak_ci_low": float(np.percentile(source_peak, low_q)),
            "peak_ci_high": float(np.percentile(source_peak, high_q)),
            "time_window": int(config.source_time_window),
            "radius": int(config.source_radius),
            "num_posterior_samples": int(config.num_posterior_samples),
        }

        leak_rate_summary = None
        if leak_rate_samples:
            leak_rate_arr = np.array(leak_rate_samples, dtype=np.float64)
            leak_rate_summary = {
                "unit": config.leak_rate_unit,
                "mean": float(leak_rate_arr.mean()),
                "ci_low": float(np.percentile(leak_rate_arr, low_q)),
                "ci_high": float(np.percentile(leak_rate_arr, high_q)),
                "num_samples": int(leak_rate_arr.size),
            }
            if config.leak_rate_gt is not None:
                leak_rate_summary["gt"] = float(config.leak_rate_gt)
                leak_rate_summary["abs_error"] = float(abs(leak_rate_summary["mean"] - float(config.leak_rate_gt)))
                if float(config.leak_rate_gt) != 0.0:
                    leak_rate_summary["rel_error"] = float(abs((leak_rate_summary["mean"] - float(config.leak_rate_gt)) / float(config.leak_rate_gt)))

        source_error = None
        if config.source_gt is not None:
            gt_pos = np.array(config.source_gt, dtype=np.float64)
            pred_pos = np.array(source_summary["position_mean"], dtype=np.float64)
            source_error = {"gt": gt_pos.tolist(), "l2_error": float(np.linalg.norm(pred_pos - gt_pos))}

        case_summaries.append(
            {
                "case_index": int(i),
                "source": source_summary,
                "leak_rate": leak_rate_summary,
                "source_error": source_error,
            }
        )

    recon_list = np.array(recon_list)
    mask_list = np.array(mask_list)
    rmse_list = np.array(rmse_list)
    if rmse_list.size > 0:
        rmse_mean = np.mean(rmse_list)
        rmse_std = np.std(rmse_list)
        print("mean:", rmse_mean, ";std:", rmse_std)
        result_name = f"{config.dataset}_mpdps_{MPDPS}_recon_rho{rho}_mode_{mt}_mean_{rmse_mean}_std_{rmse_std}"
    else:
        result_name = f"{config.dataset}_mpdps_{MPDPS}_recon_rho{rho}_mode_{mt}"

    os.makedirs("./results", exist_ok=True)
    result_path = os.path.join("./results", result_name + ".mat")
    result_payload = {"recon_list": recon_list, "mask_list": mask_list}
    if rmse_list.size > 0:
        result_payload["rmse_list"] = rmse_list
    if config.save_recon_samples and len(recon_samples_list) > 0:
        result_payload["recon_samples"] = np.array(recon_samples_list)
    sio.savemat(result_path, result_payload)

    source_est = case_summaries[0]["source"] if len(case_summaries) > 0 else {}
    np.save(os.path.join("./results", result_name + "_source_est.npy"), source_est)

    summary = {
        "dataset": config.dataset,
        "result_name": result_name,
        "num_cases": int(num_cases),
        "num_posterior_samples": int(config.num_posterior_samples),
        "leak_rate_unit": config.leak_rate_unit,
        "rmse_mean": float(rmse_list.mean()) if rmse_list.size > 0 else None,
        "rmse_std": float(rmse_list.std()) if rmse_list.size > 0 else None,
        "cases": case_summaries,
    }
    with open(os.path.join("./results", result_name + "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
