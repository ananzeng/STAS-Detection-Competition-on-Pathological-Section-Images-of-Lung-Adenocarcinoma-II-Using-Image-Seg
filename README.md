安裝
```
conda create --name env_lian_semask python=3.7
conda activate env_lian_semask
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
pip install mmcv-full==1.2.0 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
https://github.com/ananzeng/STAS-Detection-Competition-on-Pathological-Section-Images-of-Lung-Adenocarcinoma-II-Using-Image-Seg.git
pip install -e .
```
下載預訓練權重([semask_large_fpn_ade20k.pth](https://drive.google.com/file/d/1u5flfAQCiQJbMZbZPIlGUGTYBz9Ca7rE/view "semask_large_fpn_ade20k.pth")) 下載STAS權重(iter_60000.pth)
```
mkdir checkpoint
cp semask_large_fpn_ade20k.pth checkpoint
cp iter_60000.pth work_dirs/semfpn_semask_swin_large_patch4_window12_640x640_80k_ade20k
```
使用TTA 推論模型
輸出的結果位在work_dirs/com_Private_Public/semfpn_semask_swin_large_patch4_window12_640x640_80k_ade20k_ITER60000_1st
```
python tta_com.py
```