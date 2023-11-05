import os
import torch
import shutil #directory생성하는거 
import numpy as np


def create_dir(dir, opts=None):
    try:
        if os.path.exists(dir): #경로에 directory가 존재하는지 확인하는 작업
            if opts == 'del':
                shutil.rmtree(dir) # 삭제
        os.makedirs(dir, exist_ok=True) #이미 존재하면 그냥 넘거간다
    except OSError:
        print("Error: Failed to create the directory.")


def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    #경로에 존재하지 않는다면 direc만드는거 
    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))
#모델과 옵티마이저 상태 저장하고
#불러오고
def load(ckpt_dir, net, optim, set_epoch=None):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch #경로가 없다면 epoch를 0으로 설정한다 이게 빈 모델과 옵티 반환하는거 0번 학습 

    ckpt_lst = os.listdir(ckpt_dir) #파일 얻기
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f)))) #파일 정렬 epoch를 기준으로 
    # numbers = re.findall(r'\d+', string)
    if set_epoch is None: #epoch 지정되지 않았을 경우에 
         dict_model = torch.load('%s%s' % (ckpt_dir, ckpt_lst[-1])) #가장 최근에 것 가져오기
    else:
        dict_model = torch.load('%s%s' % (ckpt_dir, 'model_epoch'+str(set_epoch)+'.pth'))

    net.load_state_dict(dict_model['net']) #모델의 상태 로드하고 net의 가중치와 매개변수로 업데이트
    optim.load_state_dict(dict_model['optim'])
    if set_epoch is None:
        epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
    else:
        epoch = set_epoch

    return net, optim, epoch


class Normaliz():
    def __init__(self, x, window=None):
        self.meanx = np.mean(x) if window is None else window[0]
        self.stdx = np.std(x) if window is None else window[1]
        #window에 평균이랑 표준편차 담기
    def Normal(self, x):
        return (x - self.meanx) / self.stdx

    def Denorm(self, x): #원래로 돌릴 때 사용
        return (x * self.stdx) + self.meanx
   