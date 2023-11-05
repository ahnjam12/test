from BoneSeg import BoneSegmentation
import os
import torch
import argparse
# import vessl
'''
3D Bone Segmentation
'''
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def main():
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.tqdm_disable = False
    step = BoneSegmentation(args)
    step.do()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep learning parameters')
    parser.add_argument("--project", default="VertebraeSeg", type=str)
    parser.add_argument("--mode", default="train", choices=["train", "test"], type=str)
    parser.add_argument("--train_continue", default="off", choices=["on", "off"], type=str)

    parser.add_argument("--data_aug", default=True, choices=[True, False], type=bool)
    parser.add_argument("--crop_size", default=64, type=int)

    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--epoch_start", default=0, type=int)
    parser.add_argument("--num_epoch", default=100, type=int)

    parser.add_argument("--save_png", default=True, choices=[True, False], type=bool)
    parser.add_argument("--window", default=[0., 1.], type=list)

    # Data
    parser.add_argument('--save_dir', type=str, default='Data')
    parser.add_argument('--data_dir', type=str, default='MAT')
    #parser.add_argument('--save_dir', type=str, default='C:/Users/AhnJunhyun/Project/VertebraeSeg/Data/') 
    #parser.add_argument('--data_dir', type=str, default='C:/Users/AhnJunhyun/OneDrive - 연세대학교 (Yonsei University)/Data/Verse19/MAT/')  

    args = parser.parse_args()
    main()




