import mat73
#mat 파일 가져오기
from scipy import io
#scipy 과학기술계산 라이브러리 
#Numpy Matplotlib pandas SymPy 

# Load .mat file
def loadMAT(path, varname=None):
    # path 로드하려는 파일
    # 
    try:
        f = mat73.loadmat(path)   # only_include=self.varname[i]) 
    except: 
        f = io.loadmat(path)
        #로드 못 했을시 불러오는거
    if varname is None:
        return f
    else:
        return f[varname]