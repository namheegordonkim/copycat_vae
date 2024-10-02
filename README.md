# Embodied AI Seminar: the Copycat VAE Demo Codebase

## Set up the environment

```bash
conda create -y -n copycat python=3.11
conda activate copycat
pip install -r requirements.txt
```

## Put the data in the right place

Make sure that your `.pkl` files are in the `data` directory.

## Run the code

Run `enjoy.py` to check that the pose browser is running successfully.

```bash
python enjoy.py
```

Training: everything is done for you! Just run `train_vae.py` to train the VAE.

```bash
python train_vae.py --out_name [OUT_NAME]
```

`[OUT_NAME]` can be whatever you want to name the output. I usually use 3-digit numbers like 000, 001, etc. to track my experiments.

You can visualize the results of your VAE training using
    
```bash
python enjoy_vae.py --vae_path out/[OUT_NAME]/[PKL_NAME]
```

See the contents of the `out` directory for the `[PKL_NAME]` of the VAE you want to visualize.