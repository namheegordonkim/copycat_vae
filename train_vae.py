import os
from argparse import ArgumentParser

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tensorboardX import SummaryWriter
from torch.optim import RAdam
from torch.utils.data import DataLoader
from tqdm import tqdm

import my_logging
from torch_nets import VAE
from train_utils import quat_to_expm, ThroughDataset

body_names = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
]

upper_body = [
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
]
lower_body = [
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
]

left_arm = ["L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand"]
right_arm = ["R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"]
left_leg = ["Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe"]
right_leg = ["R_Hip", "R_Knee", "R_Ankle", "R_Toe"]
main_body = ["Torso", "Spine", "Chest", "Neck", "Head"]
upper_body_idxs = [body_names.index(b) for b in upper_body]
lower_body_idxs = [body_names.index(b) for b in lower_body]


def main():
    if args.debug_yes:
        import pydevd_pycharm

        pydevd_pycharm.settrace(
            "localhost",
            port=12345,
            stdoutToServer=True,
            stderrToServer=True,
            suspend=False,
        )
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    device = torch.device("cuda")

    logdir = f"log/{args.out_name}"
    outdir = f"out/{args.out_name}"
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    logger = my_logging.get_logger(f"{args.out_name}", logdir)
    logger.info(f"Starting")
    writer = SummaryWriter(logdir)
    writer.add_text("args", str(args))

    # Data loading
    copycat_path = os.path.join(f"data/amass_copycat_take5_train_medium.pkl")
    with open(copycat_path, "rb") as f:
        d = joblib.load(f)
    # Most sane pose data prep for copycat
    xyzs = []
    quats = []
    for k in d.keys():
        xyz = d[k]["trans_orig"]
        quat = d[k]["pose_quat"]
        xyzs.append(xyz)
        quats.append(quat)
    xyzs = np.concatenate(xyzs, axis=0)
    quats = np.concatenate(quats, axis=0)

    # Offsetting the z-axis rotation
    euls = Rotation.from_quat(quats[:, 0]).as_euler(
        "xyz"
    )  # "xyz" important because it's extrinsic
    euls[:, 2] = 0
    quats[:, 0] = Rotation.from_euler("xyz", euls).as_quat()

    # Exponential map representation
    expms = quat_to_expm(quats)

    xyzexpms = np.concatenate([xyzs[:, None], expms], axis=1)

    # Remove absolute xy coordinates: not important
    zexpms = xyzexpms.reshape(xyzexpms.shape[0], -1)[..., 2:]

    # `zexpms` is the tensor we will be working with
    zexpms = torch.tensor(zexpms, dtype=torch.float)

    # Setting up 80-20 train/valid split
    train_idxs = np.random.choice(
        zexpms.shape[0], int(0.8 * zexpms.shape[0]), replace=False
    )
    valid_idxs = np.setdiff1d(np.arange(zexpms.shape[0]), train_idxs)
    train_zexpms = zexpms[train_idxs]
    valid_zexpms = zexpms[valid_idxs]

    # Training configuration
    n_total_epochs = args.n_total_epochs
    pbar = tqdm(total=n_total_epochs)
    epochs_elapsed = 0
    batch_size = args.batch_size
    save_every = n_total_epochs // 10
    eval_every = save_every // 10
    model = None  # The model will be dynamically initialized at epoch 0

    # For consistency, we will use the same samples for training and validation error calculation
    unseen_samples = valid_zexpms[:batch_size]
    seen_samples = train_zexpms[:batch_size]

    # The training loop
    while epochs_elapsed <= n_total_epochs:
        # Dynamically initialize the model
        if model is None:
            if args.checkpoint_path is None:
                model = VAE(
                    input_size=seen_samples.shape[-1],
                    hidden_size=args.hidden_size,
                    latent_size=args.latent_size,
                )
                model.input_rms.update(train_zexpms)
                model = model.to(device)
                optimizer = RAdam(model.parameters(), lr=args.lr)
            else:
                model_d = torch.load(args.checkpoint_path)
                model = model_d["model_cls"](
                    *model_d["model_args"], **model_d["model_kwargs"]
                )
                model.load_state_dict(model_d["model_state_dict"])
                model = model.to(device)
                optimizer = RAdam(model.parameters(), lr=args.lr)
                optimizer.load_state_dict(model_d["optimizer_state_dict"])
                epochs_elapsed = model_d["epochs_elapsed"]

        # Save the model every `save_every` epochs
        if epochs_elapsed % save_every == 0 or epochs_elapsed >= n_total_epochs:
            model_d = {
                "model_cls": model.__class__,
                "model_args": model.args,
                "model_kwargs": model.kwargs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epochs_elapsed": epochs_elapsed,
                "args": args,
            }
            save_path = f"out/{args.out_name}/vae_{epochs_elapsed:06d}.pkl"
            torch.save(model_d, save_path)
            logger.info(f"Saved to {save_path}")

        # Evaluate the model on the seen and unseen samples
        if epochs_elapsed % eval_every == 0:
            model.eval()
            with torch.no_grad():
                for name, samples in [
                    ("seen", seen_samples),
                    ("unseen", unseen_samples),
                ]:
                    samples = samples.to(device)
                    z, decoded, mu, log_var = model.forward(samples)
                    mse_loss = torch.nn.functional.mse_loss(decoded, samples)
                    kld_loss = -0.5 * torch.mean(
                        1 + log_var - mu.pow(2) - log_var.exp()
                    )
                    loss = mse_loss + args.kld_weight * kld_loss
                    writer.add_scalar(
                        f"{name}/MSELoss", mse_loss.item(), epochs_elapsed
                    )
                    writer.add_scalar(
                        f"{name}/KLDLoss", kld_loss.item(), epochs_elapsed
                    )
                    mse_kld_ratio = np.maximum(
                        mse_loss.item() / kld_loss.item(),
                        kld_loss.item() / mse_loss.item(),
                    )
                    writer.add_scalar(
                        f"{name}/MSEKLDRaio", mse_kld_ratio, epochs_elapsed
                    )

                    writer.add_scalar(f"{name}/loss", loss.item(), epochs_elapsed)
                    logger.info(
                        f"Epoch {epochs_elapsed}: {name} MSELoss: {mse_loss.item():.2e} KLDLoss: {kld_loss.item():.2e} MSEKLDRatio: {mse_kld_ratio.item():.2e}"
                    )
            model.train()

        # Stop training here
        if epochs_elapsed >= n_total_epochs:
            break

        # Then supervised learning from input
        dataset = ThroughDataset(train_zexpms)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for (x,) in dataloader:
            optimizer.zero_grad()
            x = x.pin_memory().to(device)
            z, decoded, mu, log_var = model.forward(x)

            mse_loss = torch.nn.functional.mse_loss(decoded, x)
            kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            loss = mse_loss + args.kld_weight * kld_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        epochs_elapsed += 1
        pbar.update(1)
        pbar.set_postfix({"epochs": epochs_elapsed, "loss": f"{loss.item():.2e}"})

    pbar.close()

    logger.info(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--kld_weight", type=float, default=1e-2)
    parser.add_argument("--n_total_epochs", type=int, default=int(1e3))
    parser.add_argument("--continue_yes", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--debug_yes", action="store_true")
    parser.add_argument("--latent_size", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=2048)
    args = parser.parse_args()

    main()
