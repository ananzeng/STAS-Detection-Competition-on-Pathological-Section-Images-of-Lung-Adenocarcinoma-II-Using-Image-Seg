# Check Pytorch installation
import torch, torchvision
import cv2
import os
print(torch.__version__, torch.cuda.is_available())
# Check MMSegmentation installation
import mmseg
print(mmseg.__version__)
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import numpy as np

model_array = ["semfpn_semask_swin_large_patch4_window12_640x640_80k_ade20k"]

for model_name in model_array:
    #model_name = "twins_svt-l_uperhead_8x2_512x512_160k_ade20k"
    model_dir = os.path.join("work_dirs")
    config_file = os.path.join(model_dir, model_name, model_name+".py")
    checkpoint_file = os.path.join(model_dir, model_name, "iter_60000.pth")
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

    save_dir = os.path.join("work_dirs", "com_Private_Public", model_name+"_ITER60000_1st") 
    if not os.path.isdir(save_dir):
        os.mkdir(os.path.join(save_dir))
        
    test_dir = 'Private_Public'
    for i in os.listdir(test_dir):
        print(str(i))
        all_img = np.zeros((942, 1716, 15))
        for index, scale in  enumerate([(858, 471), (1287, 707), (1716, 942), (2145, 1178), (2574, 1413)]): 
            for filp_state_index, filp_state in enumerate([0, 1, -1]):
                    img = cv2.imread(os.path.join(test_dir,i))
                    img = cv2.flip(img, filp_state)
                    img = cv2.resize(img, scale, interpolation=cv2.INTER_LANCZOS4)
                    #img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
                    result = inference_segmentor(model, img)
                    predict_img = result[1][0][1]
                    predict_img = predict_img.cpu().numpy()
                    predict_img = cv2.resize(predict_img, (1716, 942), interpolation=cv2.INTER_LANCZOS4)
                    #print(predict_img.shape)a
                    predict_img = cv2.flip(predict_img, filp_state)
                    all_img[:,:,(index*3)+filp_state_index] = predict_img
                    #print(predict_img)
                    if index == 4 and filp_state_index == 2:
                        mean_image = np.mean(all_img, axis=2)
                        #print(mean_image)
                        #print(mean_image.shape)
                        mask = (mean_image >= 0.5)
                        #print(mask)
                        mask = np.array(mask,np.uint8)*255     
                        cv2.imwrite(os.path.join(save_dir, i.split(".")[0]+".png"), mask)

            
