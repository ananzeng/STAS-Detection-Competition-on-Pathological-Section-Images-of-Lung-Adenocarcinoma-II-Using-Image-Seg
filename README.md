**安裝**
```
conda create --name env_lian_semask python=3.7
conda activate env_lian_semask
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
pip install mmcv-full==1.2.0 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
git clone https://github.com/ananzeng/STAS-Detection-Competition-on-Pathological-Section-Images-of-Lung-Adenocarcinoma-II-Using-Image-Seg.git
pip install -e .
```
**下載預訓練權重**([semask_large_fpn_ade20k.pth](https://drive.google.com/file/d/1u5flfAQCiQJbMZbZPIlGUGTYBz9Ca7rE/view "semask_large_fpn_ade20k.pth")) 下載STAS權重([iter_60000.pth](https://drive.google.com/file/d/1NtPSIaTmjYFXSMOvlUMGbgauXIboaaxX/view?usp=sharing "iter_60000.pth"))
```
mkdir checkpoint
cp semask_large_fpn_ade20k.pth checkpoint
cp iter_60000.pth work_dirs/semfpn_semask_swin_large_patch4_window12_640x640_80k_ade20k
```
**使用TTA 推論模型**
將[此行](https://github.com/ananzeng/STAS-Detection-Competition-on-Pathological-Section-Images-of-Lung-Adenocarcinoma-II-Using-Image-Seg/blob/main/mmseg/models/segmentors/encoder_decoder.py#L287 "此行")取消註解  將[此行](https://github.com/ananzeng/STAS-Detection-Competition-on-Pathological-Section-Images-of-Lung-Adenocarcinoma-II-Using-Image-Seg/blob/main/mmseg/models/segmentors/encoder_decoder.py#L288 "此行")註解
輸出的結果位在work_dirs/com_Private_Public/semfpn_semask_swin_large_patch4_window12_640x640_80k_ade20k_ITER60000_1st
```
python tta_com.py
```

**產生訓練資料**
將比賽方提供的資料夾SEG_Train_Datasets做預處理產生ground truth以及ADE20K資料格式的資料集
```
python make_gt_image.py
```

**進行訓練**
將[此行](https://github.com/ananzeng/STAS-Detection-Competition-on-Pathological-Section-Images-of-Lung-Adenocarcinoma-II-Using-Image-Seg/blob/main/mmseg/models/segmentors/encoder_decoder.py#L287 "此行")註解  將[此行](https://github.com/ananzeng/STAS-Detection-Competition-on-Pathological-Section-Images-of-Lung-Adenocarcinoma-II-Using-Image-Seg/blob/main/mmseg/models/segmentors/encoder_decoder.py#L288 "此行")取消註解
產生的權重檔案會在./work_dirs/semfpn_semask_swin_large_patch4_window12_640x640_80k_ade20k
```
python tools/train.py configs/semfpn_semask_swin_large_patch4_window12_640x640_80k_ade20k.py --seed 69 --deterministic
```