# MRDiff

## Installation
```
python=3.6
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install denoising_diffusion_pytorch
conda install pandas
pip install sklearn
pip install matplotlib
pip install tadpak
pip install transformers
pip install wandb==0.15.11
```

## Training
```
python DiffusionAE/train_diffusion_val.py --dataset point_global --denoise_steps 50 --batch_size 8 --training diffusion --lr 1e-3 --window_size 100 --noise_steps 100
```

## Test
```
python DiffusionAE/train_diffusion_val.py --dataset pattern_seasonal --denoise_steps 80 --batch_size 8 --training diffusion --lr 1e-3 --window_size 100 --noise_steps 100 --test_only True
```
## Our paper
```
@inproceedings{na2024mrdiff,
  title={MRDiff: Time Series Anomaly Detection Using Multi-level Reconstruction Diffusion},
  author={Na, Dagyeong and Kwon, Junseok},
  booktitle={2024 IEEE International Conference on Data Mining Workshops (ICDMW)},
  pages={688--695},
  year={2024},
  organization={IEEE}
}
```
