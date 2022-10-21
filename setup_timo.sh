conda deactivate
conda create --name spacex python=3.9
conda activate spacex

pip install git+https://gitlab.jsc.fz-juelich.de/kiste/asos@update

conda install numpy pandas scikit-learn matplotlib tqdm

conda install -c pytorch pytorch torchvision torchaudio cudatoolkit=11.3
conda install -c conda-forge tensorboard
conda install -c conda-forge wandb

<<<<<<< HEAD:setup_timo.sh
conda install -c conda-forge earthengine-api
pip install git+https://gitlab.jsc.fz-juelich.de/kiste/asos@main_without_gdal

=======
>>>>>>> develop_timo:setup.sh
