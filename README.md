# $\rm O^2$-Recon

[[`Paper`](https://arxiv.org/abs/2308.09591)]


> $\rm O^2$-Recon: Completing 3D Reconstruction of Occluded Objects in the Scene with a Pre-trained 2D Diffusion Model <br>
[Yubin Hu](https://github.com/AlbertHuyb), Sheng Ye, Wang Zhao, Matthieu Lin, [Yuze He](https://github.com/hyz317/), Yu-Hui Wen, Ying He, Yong-Jin Liu <br>
AAAI 2024
> 
We will release the full code as soon as possible.

<<<<<<< HEAD
[x] Release the code of training on an example object data.
[ ] Release the dataset pre-processing code.
[ ] Update readme.

## Environmental Setup

conda create -n O2-recon python=3.7

conda activate O2-recon

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

git clone --recurse-submodules https://github.com/THU-LYJ-Lab/O2-Recon.git

cd O2-Recon

pip install -r requirements.txt

pip install git+https://github.com/openai/CLIP.git

Download the normal prediction model `scannet_neuris_retrain.pt` from [here](https://connecthkuhk-my.sharepoint.com/personal/jiepeng_connect_hku_hk/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fjiepeng%5Fconnect%5Fhku%5Fhk%2FDocuments%2FGitHub%2FNeuRIS%2Fpretrained%20normal%20network%2Fsnu) and store it to `./preprocess/surface_normal_uncertainty/checkpoints/`





## Dataset Preprocessing

You can download the in-painted dataset from here: .
After downloading, unzip the file into the directory `./dataset/indoor-paper`. 

Or you can prepare the dataset from scratch. We provide an example to prepare the dataset from scannet format.

Firstly download the scannet dataset. You can download the scenes we utilized from here. Unzip the files into `./scannet/scenexxxx_xx_scannet` directories.

Depending on which scenes you'd like to process, you need to modify L150 in `preprocess/object_mask_with_clip.py`. And then run 

```
python preprocess/object_mask_with_clip.py
```

This script extracts objects from the scannet scenes according to instance masks and semantic categories. As a result, the object data is extracted into `./scannet/object_original_with_clip/scenexxxx_xx_scannet_obj_x`. 

Based on these object directories, next we generate the in-paintings and predict the monocular cues.

Depending on which scenes you'd like to process, you need to modify L41 in `exp_preprocess.py`. 
Before running the preprocess script, make sure that you have placed the annotated masks to the correct directory. For in-painting, this script utilizes the annotated masks from sparse views, which are stored in `./scannet/object_original_with_clip/scenexxxx_xx_scannet_obj_x/only-seg-xxxx-objx`

And then run 

```
python exp_preprocess.py --data_type scannet-with-inpaint --scannet_root=/path/to/O2-Recon/scannet/ --neus_root=/path/to/O2-Recon/dataset/indoor-paper/ --dir_snu_code /path/to/O2-Recon/preprocess/surface_normal_uncertainty/
```

=======
- [x] Release the code of training on an example object data.
- [ ] Release the dataset pre-processing code.
- [ ] Update readme.
>>>>>>> 9c10f1098e6341729f5ac537715ac542be12a037
