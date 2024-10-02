import os
from argparse import ArgumentParser

import joblib
import numpy as np
import torch
from imgui_bundle import immapp
from imgui_bundle._imgui_bundle import imgui, hello_imgui
from pyvista_imgui import ImguiPlotter
from scipy.spatial.transform import Rotation
from tensorboardX import SummaryWriter
import pyvista as pv
import my_logging
from poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree
from torch_nets import VAE
from train_utils import quat_to_expm, expm_to_quat, expm_to_quat_torch
from utils import set_skel_pose, load_skel, add_skel_meshes, load_axes, add_axes_meshes

MJCF_PATH = "assets/my_smpl_humanoid.xml"
device = torch.device("cuda")


class AppState:
    def __init__(
        self,
        skel_state: SkeletonState,
        skel_actors,
        axes_actors,
        zexpms: torch.Tensor,
        vae: VAE,
    ):
        self.skel_state = skel_state
        self.skel_actors = skel_actors
        self.axes_actors = axes_actors
        self.show_axes = False
        self.zexpms = zexpms
        self.pose_idx = 0
        self.vae = vae
        self.latents = torch.zeros(vae.latent_size, dtype=torch.float, device=device)
        self.first_yes = True


def setup_and_run_gui(pl: ImguiPlotter, app_state: AppState):
    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "Viewer"
    runner_params.app_window_params.window_geometry.size = (1280, 720)

    def gui():
        hello_imgui.apply_theme(hello_imgui.ImGuiTheme_.imgui_colors_dark)

        viewport_size = imgui.get_window_viewport().size

        # PyVista portion
        imgui.set_next_window_size(imgui.ImVec2(viewport_size.x // 2, viewport_size.y))
        imgui.set_next_window_pos(imgui.ImVec2(viewport_size.x // 2, 0))
        imgui.set_next_window_bg_alpha(1.0)
        imgui.begin(
            "ImguiPlotter",
            flags=imgui.WindowFlags_.no_bring_to_front_on_focus
            | imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_decoration
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move,
        )
        # render the plotter's contents here
        pl.render_imgui()
        imgui.end()

        # GUI portion
        imgui.set_next_window_size(imgui.ImVec2(viewport_size.x // 2, viewport_size.y))
        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_bg_alpha(1.0)
        imgui.begin(
            "Controls",
            flags=imgui.WindowFlags_.no_bring_to_front_on_focus
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move,
        )

        changed, app_state.pose_idx = imgui.slider_int(
            "Pose Index", app_state.pose_idx, 0, app_state.zexpms.shape[0] - 1
        )
        # If pose idx changes, then use the current pose to calculate latents
        if changed or app_state.first_yes:
            zexpm = app_state.zexpms[app_state.pose_idx]
            with torch.no_grad():
                encoded = app_state.vae.encode(
                    torch.tensor(zexpm, dtype=torch.float, device=device)[None]
                )
                mu = app_state.vae.mu(encoded)
                log_var = app_state.vae.log_var(encoded)
                latents = app_state.vae.reparameterize(mu, log_var)
                app_state.latents = latents[0]

        changed, app_state.show_axes = imgui.checkbox("Show Axes", app_state.show_axes)

        changeds = np.zeros(app_state.vae.latent_size, dtype=bool)
        for i in range(app_state.vae.latent_size):
            changeds[i], app_state.latents[i] = imgui.slider_float(
                f"Latent {i}", app_state.latents[i], -3, 3
            )

        imgui.end()

        with torch.no_grad():
            zexpm = app_state.vae.decode(app_state.latents[None])[0]
            global_translation = torch.zeros(3)
            global_translation[-1] = zexpm[0]
            local_rotation = torch.as_tensor(
                expm_to_quat_torch(zexpm[1:].reshape(-1, 3))
            )

        # Set the character pose
        app_state.skel_state = SkeletonState.from_rotation_and_root_translation(
            app_state.skel_state.skeleton_tree,
            local_rotation.cpu().detach(),
            global_translation.cpu().detach(),
            is_local=True,
        )

        set_skel_pose(
            app_state.skel_state,
            app_state.skel_actors,
            app_state.axes_actors,
            app_state.show_axes,
        )

        app_state.first_yes = False

    runner_params.callbacks.show_gui = gui
    runner_params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.no_default_window
    )
    immapp.run(runner_params=runner_params)


def main():
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

    pl = ImguiPlotter()
    pl.enable_shadows()
    pl.add_axes()
    pl.camera.position = (5, -5, 3)
    pl.camera.focal_point = (0, 0, 1)
    pl.camera.up = (0, 0, 1)

    # Initialize meshes
    floor = pv.Plane(i_size=10, j_size=10)
    skels = load_skel(MJCF_PATH)
    axes = load_axes(MJCF_PATH)

    # Register meshes, get actors for object manipulation
    pl.add_mesh(floor, show_edges=True, pbr=True, roughness=0.24, metallic=0.1)
    sk_actors = add_skel_meshes(pl, skels)
    ax_actors = add_axes_meshes(pl, axes)

    # Set character pose to default
    # Center the character root at the origin
    root_translation = torch.zeros(3)
    # Set global rotation as unit quaternion
    body_part_global_rotation = torch.zeros(24, 4)
    body_part_global_rotation[..., -1] = 1

    # `poselib` handles the forward kinematics
    sk_tree = SkeletonTree.from_mjcf(MJCF_PATH)
    sk_state = SkeletonState.from_rotation_and_root_translation(
        sk_tree, body_part_global_rotation, root_translation, is_local=False
    )
    set_skel_pose(sk_state, sk_actors, ax_actors, show_axes=False)

    # Load the model
    vae_path = args.vae_path
    model_d = torch.load(vae_path)
    vae = model_d["model_cls"](*model_d["model_args"], **model_d["model_kwargs"])
    vae.load_state_dict(model_d["model_state_dict"])
    vae = vae.to(device)

    # Run the GUI
    app_state = AppState(sk_state, sk_actors, ax_actors, zexpms, vae)
    setup_and_run_gui(pl, app_state)

    print(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--vae_path", type=str, required=True)
    args = parser.parse_args()

    main()
