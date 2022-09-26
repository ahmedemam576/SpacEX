conda deactivate
conda create --name coding_streak_environment2 python=3.9
conda activate coding_streak_environment2

conda install numpy pandas scikit-learn matplotlib tqdm

conda install -c pytorch pytorch torchvision torchaudio cudatoolkit=11.3
conda install -c conda-forge tensorboard
conda install -c conda-forge wandb

pip install git+https://gitlab.jsc.fz-juelich.de/kiste/asos@main_without_gdal

