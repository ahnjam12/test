from utility import *
from loadMAT import loadMAT


# Data Loader
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None, plane='axial', filename=None, varname=['segTemp', 'rawTemp'], arraySize=256):
        self.data_path = data_path
        self.transform = transform
        self.plane = plane
        self.filename = filename  # [2,N] shape strin list, [ref, target]
        self.varname = varname  # 2 string [ref, target]
        self.arraySize = arraySize
        self.data_array = self.preprocessing()
        self.ToTensor = ToTensor()
        print('Dataset:DATA SIZE = ' + str(self.data_array.shape))  # [:, N, Z, Y, X]

    def __len__(self):
        return self.data_array.shape[1]

    def __getitem__(self, index):
        data1 = self.data_array[0, index]  # Label  [Z, Y, X]
        data2 = self.data_array[1, index]  # Input CT image
        data = {'label': data1, 'input': data2}
        # {label z,y,x , ct image}
        if self.transform is not None:
            data = self.transform(data)

        data = self.ToTensor(data)

        return data
    #데이터 전처리
    def preprocessing(self):
        for i in range(len(self.filename)): #파일개수로
            fileID = os.path.join(self.data_path, self.filename[i][0]) #(data 경로, 현재 파일의 이름) join으로 경로 결합
            f = loadMAT(fileID)  #이미지를 불러와요
            for j in range(2): # 0 1 2 세개씩 합치는거 쨔스
                temp = f[self.varname[j]] #segtemp rawtemp
                temp = np.ascontiguousarray(temp) #메모리에서 연속적으로 저장되지 않는 배열을 연속적으로 저장되는 배열로 변환하는 것 
                # Padding 
                temp = self.padding(temp)
                #plane 따라서 코딩하기 
                if self.plane == 'axial':
                    # Write your code here  ex) temp = temp.transpose((0, 1, 2))[None, :, None, :, :]
                    # 예를 들어, 데이터를 [Z, Y, X]에서 [1, Z, 1, Y, X] 형태로 변환할 수 있습니다.
                    temp = temp.transpose((0,1,2))[None, :, None, :, :]
                elif self.plane == 'sagittal': #좌우로 나눌때 
                    # Write your code here
                    # 예를 들어, 데이터를 [Z, Y, X]에서 [Z, Y, X/2] 형태로 변환할 수 있습니다.
                    temp = temp[:, :, :temp.shape[2] // 2]
                elif self.plane == 'coronal': #앞뒤로 나눌때 
                    # Write your code here
                    # 예를 들어, 데이터를 [Z, Y, X]에서 [Z, Y/2, X] 형태로 변환할 수 있습니다.
                    temp = temp[:, :temp.shape[1] // 2, :]
                else:
                    print('ERROR: plane is not defined')
                    exit()
                datatemp = temp if j==0 else np.concatenate((datatemp, temp), axis=0) 
                #concatenate 배열 합치는거 2차원배열 axis=0 은 위->아래 axis=1 좌->우
                # 3차원 0 높이 1 행 2 열방향
            if i==0 and j==0:  # You can change how the normalize factor is determined. 
                self.minmax = [np.min(datatemp), np.max(datatemp)] #나중에 정규화하기 위해서 하는거 최대최소 범위  #데이터정규화하는거 추가가능
            data_array = datatemp if i==0 else np.concatenate((data_array, datatemp), axis=1)
        data_array = self.minmaxNorm(data_array)

        return data_array

    def minmaxNorm(self, data):
        return (data - self.minmax[0]) / (self.minmax[1] - self.minmax[0])
    
    def minmaxDenorm(self, data): #역정규화
        return data * (self.minmax[1] - self.minmax[0]) + self.minmax[0]
    
    def inversePlaneTransform(self, data): #차원재배치
        if self.plane == 'axial':
            data = data[0]
        elif self.plane == 'sagittal':
            data = data[0].transpose((1, 2, 0)) # 원래 z y x 였으니까 y x z로 
        elif self.plane == 'coronal':
            data = data[0].transpose((2, 1, 0)) #원래 z y x 였으니까 x y z로 
        else:
            print('ERROR: plane is not defined')
            exit()
        return data
    
    def padding(self, data): #제로 패딩을 쓴다더라 데이터에 패딩을 적용 배열 크기 조절에 사용  padding하는거 추가하기 
        # Make the array the same size as arraySize
        # Write your code here
        # 패딩 크기 계산
        padding_x = (self.arraySize - data.shape[2]) // 2
        padding_y = (self.arraySize - data.shape[1]) // 2
        padding_z = (self.arraySize - data.shape[0]) // 2

        # 패딩된 데이터 생성
        padded_data = np.pad(data, ((padding_z, padding_z),
                            (padding_y, padding_y), (padding_x, padding_x)), mode='constant')
        return data


class ToTensor(object): #tensor로 바꿔주는 것
    def __call__(self, data):
        for key, value in data.items():
            value = value.astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data


class RandomRot90(object): #90도 회전시키는거 
    def __call__(self, data):
        # Write your code here
        k = np.random.choice([0, 1, 2, 3])
        for key, val in data.items():
            val=np.rot90(val, k, axes=(0, 1))
            data[key]=val
        return data


class RandomFlip(object): #데이터를 무작위로 수평 수직으로
    def __call__(self, data):
        # Write your code here
        #좌우
        if np.random.rand() >0.5:
            for key, val in data.items():
                val=np.fliuplr(val)
                data[key]=val
        if np.random.rand() >0.5:
            for key, val in data.items():
                val=np.flipud(val)
                data[key]=val
        return data


class RandomCrop(object): #무작위로 크기 줄일 수 있는 것 크기 받아서 
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        # Write your code here
        depth, height, width = data.shape
        for key, val in data.items():
            shpae=self.shape
            cur_shape=val.shape
            
        # 잘린 데이터의 시작 지점을 무작위로 선택
        start_depth = np.random.randint(0, depth - self.shape[0] + 1)
        start_height = np.random.randint(0, height - self.shape[1] + 1)
        start_width = np.random.randint(0, width - self.shape[2] + 1)

        # 데이터를 선택된 부분으로 자름
        cropped_data = data[start_depth:start_depth + self.shape[0],
                            start_height:start_height + self.shape[1],
                            start_width:start_width + self.shape[2]]
        data[key]=val
        return data



#class RandomFlip(object):
#    def __call__(self, data):
#        label, input = data['label'], data['input']

#        #이미지 원본을 랜덤하게 좌우반전
#        if np.random.rand() > 0.5:
#            label = np.fliplr(label)
#            input = np.fliplr(input)

#        #이미지 원본을 랜덤하게 상하반전
#        if np.random.rand() > 0.5:
#            label = np.flipud(label)
#           input = np.flipud(input)
#
#       data = {'label': label, 'input': input}
#
#        return data