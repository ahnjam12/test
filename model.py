import torch
import torch.nn as nn
# GPU를 제대로 이용하기 위한 Numpy의 대체제

#pytorch를 이용한 layer모델 정의 
#괄호 안에 있는게 상속해 주는 것
class DECBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm=None, relu=0.0):
        # 입력 채널 수, 출력 채널 수, 컨볼루션 커널 크기, 컨볼루션 연산 스트라이드, 컨볼루션 패딩, 편향 여부, 
        # norm은 정규화 방법 bnorm은 배치 정규화, inorm은 인스턴스 정규화 norm을 사용하지 않으려면 None, relu는 활성화함수 0보다 클 때, 0일 때는 LeakyReLU
        super().__init__()
        # 부모 클래스에 전달해 줄 정보들 super
        layers = []
        layers += [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        # 신경망 연산 해주는거
        # nn.convTrnaspose2d를 만들 때 input값의 channel 수와 설정한 output channel 수, kernel size를 고려해 weight를 만든다.
        # weight의 크기는 (input channel, output channel, kernel size, kernel size)
        # input의 각 element 별로 weight와 곱해줍니다. 만약 stride가 1보다 크다면, 그 값만큼 이동하면서 만들어줍니다.
        # 나온 모든 결괏값을 element-wise 하게 더해서 최종 결과를 냅니다.
        if not norm is None:
            if norm == "bnorm":
                layers += [nn.BatchNorm2d(num_features=out_channels)]
                #배치 정규화를 수행하는 클래스 2차원 이미지 데이터에 대한 배치 정규화 적용 num_features 입력 채널의 개수
            elif norm == "inorm":
                layers += [nn.InstanceNorm2d(num_features=out_channels)]
                # 인스턴스 정규화

        if not relu is None and relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]
        # 이거 그냥 conv연산이랑 normalization이랑 Relu 한가지 함수에 넣어둔거
        self.cbr = nn.Sequential(*layers)
        #layer를 순차적으로 쌓아서 구성하는데 사용되는 코드

    def forward(self, x):
        return self.cbr(x)
    # 실제 데이터를 모델을 통해 전달하는 forward 연산 x를 cbr에 전달해 레이어를 통과시키고 결과 반환
# 이거 그냥 conv연산이랑 normalization이랑 Relu 한가지 함수에 넣어둔거
class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm", relu=0.0):
        super().__init__()

        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding,
                             bias=bias)]
        if not norm is None:
            if norm == 'bnorm':
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == 'inorm':
                layers += [nn.InstanceNorm2d(num_features=out_channels)]
        if not relu is None:
            layers += [nn.ReLU() if relu == 0.0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layers) #앞에랑 이제 똑같은데 이게 뭘까

    def forward(self, x):
        return self.cbr(x)


class UNet(nn.Module):
    def __init__(self, nch=1, nker=64, norm='bnorm', learning_type='residual', unpool='trans'):
        super(UNet, self).__init__()
        self.learning_type = learning_type

        # Contracting path
        self.enc1_1 = CBR2d(in_channels=nch, out_channels=1*nker, norm=norm) # 1 64     572X572X1
        self.enc1_2 = CBR2d(in_channels=1*nker, out_channels=1*nker, norm=norm) # 64 64   568X568X64
        #self.conv1=nn.Sequential(enc1_1,enc1_2)
        self.pool1 = nn.MaxPool2d(kernel_size=2) # 568X568X64 284X284X64

        self.enc2_1 = CBR2d(in_channels=1*nker, out_channels=2*nker, norm=norm) # 64 128 284X284X64
        self.enc2_2 = CBR2d(in_channels=2*nker, out_channels=2*nker, norm=norm) # 128 128 280X280X128

        self.pool2 = nn.MaxPool2d(kernel_size=2) #280X280X128 140X140X128

        self.enc3_1 = CBR2d(in_channels=2*nker, out_channels=4*nker, norm=norm) # 128 256  140X140X128
        self.enc3_2 = CBR2d(in_channels=4*nker, out_channels=4*nker, norm=norm) # 256 256  136X136X256

        self.pool3 = nn.MaxPool2d(kernel_size=2) # 136x136x256 => 68x68x256

        self.enc4_1 = CBR2d(in_channels=4*nker, out_channels=8*nker, norm=norm) #
        self.enc4_2 = CBR2d(in_channels=8*nker, out_channels=8*nker, norm=norm)
        #self.conv4 = nn.Sequential(CBR2d(256, 512, 3, 1),CBR2d(512, 512, 3, 1),nn.Dropout(p=0.5))
        self.pool4 = nn.MaxPool2d(kernel_size=2) # 64x64x512 32x32x512

        self.enc5_1 = CBR2d(in_channels=8*nker, out_channels=16*nker, norm=norm)

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=16*nker, out_channels=8*nker, norm=norm)
        #여기서 다시 확대 upsampling
        if unpool == 'trans': #transpose convolution 사용하는경우 
            self.unpool4 = nn.ConvTranspose2d(in_channels=8*nker, out_channels=8*nker,
                                              kernel_size=2, stride=2, padding=0, bias=True)
        elif unpool == 'bilinear': # Bilinear upsampling을 사용하는경우
            self.unpool4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec4_2 = CBR2d(in_channels=2 * 8*nker, out_channels=8*nker, norm=norm)
        self.dec4_1 = CBR2d(in_channels=8*nker, out_channels=4*nker, norm=norm)

        if unpool == 'trans':
            self.unpool3 = nn.ConvTranspose2d(in_channels=4*nker, out_channels=4*nker,
                                              kernel_size=2, stride=2, padding=0, bias=True)
        elif unpool == 'bilinear':
            self.unpool3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec3_2 = CBR2d(in_channels=2* 4*nker, out_channels=4*nker, norm=norm)
        self.dec3_1 = CBR2d(in_channels=4*nker, out_channels=2*nker, norm=norm)

        if unpool == 'trans':
            self.unpool2 = nn.ConvTranspose2d(in_channels=2*nker, out_channels=2*nker,
                                              kernel_size=2, stride=2, padding=0, bias=True)
        elif unpool == 'bilinear':
            self.unpool2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec2_2 = CBR2d(in_channels=2 * 2*nker, out_channels=2*nker, norm=norm)
        self.dec2_1 = CBR2d(in_channels=2*nker, out_channels=1*nker, norm=norm)

        if unpool == 'trans':
            self.unpool1 = nn.ConvTranspose2d(in_channels=1*nker, out_channels=1*nker,
                                              kernel_size=2, stride=2, padding=0, bias=True)
        elif unpool == 'bilinear':
            self.unpool1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec1_2 = CBR2d(in_channels=2 * 1*nker, out_channels=1*nker, norm=norm)
        self.dec1_1 = CBR2d(in_channels=1*nker, out_channels=1*nker, norm=norm)

        self.fc = nn.Conv2d(in_channels=1*nker, out_channels=nch, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)
        #cat함수 합치는거
        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        if self.learning_type == 'plain':
            out = self.fc(dec1_1)
        elif self.learning_type == 'residual':
            out = self.fc(dec1_1) + x   # output과 네트워크 인풋을 더해 최종 아웃풋으로 낸다(residual learning)

        return out