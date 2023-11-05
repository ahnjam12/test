from model import *
from Dataset import *
from utility import *

import torch
import time
import matplotlib.pyplot as plt
# import vessl
from scipy import io

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from statistics import mean


class BoneSegmentation:
    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        # ETC Settings
        self.cmap = 'gray'

        # Address Settings
        self.data_dir = args.data_dir
        self.tensorboard_dir = args.save_dir + '/Tensorboard/' + args.model_name + '/'  
        self.model_dir = args.save_dir + '/TrainModel/'+ args.model_name + '/' 
        self.results_dir = args.save_dir + '/TrainResult/' + args.model_name + '/' 

        create_dir(self.tensorboard_dir + 'train/', opts='del')
        create_dir(self.tensorboard_dir + 'val/', opts='del')
        create_dir(self.model_dir)
        create_dir(self.results_dir)
        create_dir(self.results_dir + 'train/')
        create_dir(self.results_dir + 'train/png/')
        create_dir(self.results_dir + 'val/')
        create_dir(self.results_dir + 'val/png/')

        # SummaryWriter settings for Tensorboard
        #self.writer_train = SummaryWriter(log_dir=self.tensorboard_dir + 'train/')
        #self.writer_val = SummaryWriter(log_dir=self.tensorboard_dir + 'val/')
        # 이거는 안 쓰는 것이 좋다고 하였다
    def do(self):
        # Initializing
        batch_size = self.args.batch_size
        device = self.device
        learning_rate = self.args.learning_rate
        num_epoch = self.args.num_epoch
        # loss_weight = self.args.loss_weight

        train_continue = self.args.train_continue
        crop_size = self.args.crop_size
        save_png = self.args.save_png
        window = self.args.window
        cmap = self.cmap

        # Split data into train, val, test
        numlist_train = np.squeeze(loadMAT(self.data_dir + 'train/Information.mat', 'filenumlist'))
        numlist_val = np.squeeze(loadMAT(self.data_dir + 'val/Information.mat', 'filenumlist'))
        numlist_test = np.squeeze(loadMAT(self.data_dir + 'test/Information.mat', 'filenumlist'))

        filename_train = [['vertebrae'+str(numlist_train[i]).zfill(3)+'.mat'] for i in range(len(numlist_train))]
        filename_val = [['vertebrae'+str(numlist_val[i]).zfill(3)+'.mat'] for i in range(len(numlist_val))]
        filename_test = [['vertebrae'+str(numlist_test[i]).zfill(3)+'.mat'] for i in range(len(numlist_test))]

        # Dataloader setting
        t_start = time.time()
        if self.args.mode == 'train':
            if self.args.data_aug:  # data augmentation
                transform_train = transforms.Compose([RandomFlip(), RandomRot90(), RandomCrop(crop_size)]) 
                transform_val = transforms.Compose([RandomCrop(crop_size)])
            else:
                transform_train = None
                transform_val = None
            dataset_train = Dataset(self.data_dir + '/train/', transform=transform_train, filename=filename_train)
            loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

            dataset_val = Dataset(self.data_dir + '/val/', transform=transform_val, filename=filename_val)
            loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

            num_data_train = dataset_train.__len__()
            num_data_val = dataset_val.__len__()
            num_batch_train = np.ceil(num_data_train / batch_size)
            num_batch_val = np.ceil(num_data_val / batch_size)

        elif self.args.mode == 'test':
            dataset_test = Dataset(self.data_dir + '/test/', filename=filename_test)
            loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
            num_data_test = dataset_test.__len__()
            num_batch_test = np.ceil(num_data_test / batch_size)
        print("=== DATA SET-UP TIME %.2f sec ===" % (time.time() - t_start))

        # etc
        # 데이터 후처리를 위한 두개의 람다 함수 텐서를 넘파이배열으로 변환하고 그 데이터를 윈도우링 하는데 사용 
        # 보통(높이,너비,채널)인데 (채널,높이,너비)로 변경
        fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(1, 2, 3, 0)
        fn_window = lambda x, wind: np.clip((x - wind[0]) / (wind[1] - wind[0]), 0, 1)
        # to(device)는 GPU CPU로 옮기는 것을 의미한다
        ### Deep Learining Model ###
        net = UNet().to(device)

        # Optimizer
        optim = torch.optim.Adam(net.parameters(), lr=learning_rate)

        # Loss function
        fn_loss = nn.Binary_Cross_Entropy_loss()  # nn.MSE()  # nn.L1Loss()
        fn_loss = fn_loss.to(device)

        st_epoch = 0
        #모델 학습 loop
        if self.args.mode == 'train':
            if train_continue == "on":
                print("=== TRAIN CONTINUE ===")
                net, optim, st_epoch = load(ckpt_dir=self.model_dir, net=net, optim=optim)

            for epoch in range(st_epoch + 1, num_epoch + 1):
                net.train()

                loss_train = []
                tstart = time.time()
                for batch, data in enumerate(loader_train):
                    # forward pass
                    input = data['input'].to(device)
                    target = data['target'].to(device)

                    output = net(input)
                    # backward pass
                    optim.zero_grad()

                    loss = fn_loss(output, target)
                    loss.backward()
                    optim.step()

                    # 손실함수 계산
                    loss_train += [loss.item()]

                self.writer_train.add_scalar('loss', mean(loss_train), epoch) #? 
                vessl.log(step=epoch, payload={logtag + 'train_loss': mean(loss_train)})

                with torch.no_grad():
                    net.eval()
                    loss_val = []
                    for batch, data in enumerate(loader_val):
                        # forward pass
                        input = data['input'].to(device)
                        target = data['target'].to(device)

                        output = net(input)

                        # 손실함수 계산하기
                        loss = fn_loss(output, target)
                        loss_val += [loss.item()]
                        if save_png and batch % 50 == 0 and epoch % 5 == 1:
                            # Tensorboard 저장하기
                            target = fn_window(fn_tonumpy(target), window)
                            input = fn_window(fn_tonumpy(input), window)
                            output = fn_window(fn_tonumpy(output), window)
                            # 이미지 값을 윈도우링하여 픽셀 값의 범위를 조절
                            id = num_batch_val * (epoch - 1) + batch
                            # 각 검증 결과에 대한 이미지에 고유한 ID를 할당 파일 이름에 사용  
                            plt.imsave(self.results_dir + 'val/png/' + str(id).zfill(4) + '_target.png',
                                       target[0,:,:,0], cmap=cmap)
                            plt.imsave(self.results_dir + 'val/png/' + str(id).zfill(4) + '_input.png',
                                       input[0,:,:,0], cmap=cmap)
                            plt.imsave(self.results_dir + 'val/png/' + str(id).zfill(4) + '_output.png',
                                       output[0,:,:,0], cmap=cmap)
                            # 검증 결과 이미지 저장
                self.writer_val.add_scalar('loss', mean(loss_val), epoch)
                # vessl.log(step=epoch, payload={logtag + 'val_loss': mean(loss_val)})
                print("EPOCH %04d / %04d | TRAIN LOSS %.6f | VAL LOSS %.6f | TIME %.2f sec" %
                      (epoch, num_epoch, mean(loss_train), mean(loss_val), time.time() - tstart))
                if epoch % 10 == 0:
                    save(ckpt_dir=self.model_dir, net=net, optim=optim, epoch=epoch)

            self.writer_train.close()
            self.writer_val.close()

        # TEST MODE
        elif self.args.mode == 'test':
            net, optim, st_epoch = load(ckpt_dir=self.model_dir, net=net, optim=optim)

            with torch.no_grad():
                net.eval()
                for batch, data in enumerate(loader_test):
                    # forward pass
                    input = data['input'].to(device)
                    output_temp = fn_tonumpy(net(input))  # (C, W, H, N)

                    output = output_temp if batch==0 else np.concatenate((output, output_temp),axis=3)  # (C, W, H, N)

            print("Output array shape = " + str(output.shape))
            # Denormalization
            output = dataset_test.minmaxDenorm(output)
            # Inverse Plane Transformation
            output = dataset_test.inversePlaneTransform(output)
            io.savemat(self.results_dir + 'output.mat', mdict={'output': output})

    # tensorboard --logdir=
    # localhost:6006
