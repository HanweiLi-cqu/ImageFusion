import cv2
import numpy as np
from pywt import dwt2,idwt2
from skimage import measure as ImageMeasure,metrics as ImageMetrics

# 图片融合
class ImageFusion(object):
    """图像融合工具类
    其中小波变换类型默认使用haar
    针对低频处理办法有使用最大值(max), 均值(avg), 使用标准差作为权重(weight)
    针对高频处理办法有使用局部标准差(loc), 最大值(max), 标准差加权(locWeight)

    Args:
        leftImagePath (str): 左边图片路径
        rightImagePath (str): 右边图片路径
        dwtType (str): dwt变换类型, 通过setDwtType设置
        lowFreqMethod (str): 低频处理方法, 通过setLowFreqMethod设置,类型有max,avg,weight
        highFreqMethod (str): 高频处理方法, 通过setHighFreqMethod设置,类型有loc,max,locWeight
    """
    def __init__(self,leftImagePath,rightImagePath):
        self.leftImagePath = leftImagePath
        self.rightImagePath = rightImagePath
        # 默认dwt小波变换类型为haar
        self.dwtType = 'haar'
        # 默认低频处理方法为按照权重计算
        self.lowFreqMethod = 'weight'
        # 默认高频处理办法为局部标准差
        self.highFreqMethod = 'loc'
    
    def ImageLoad(self):
        """
        加载图片
        """
        imageLeft = cv2.imread(self.leftImagePath)
        imageRight = cv2.imread(self.rightImagePath)
        # split image to RGB channels
        bL,gL,rL = cv2.split(imageLeft)
        bR,gR,rR = cv2.split(imageRight)
        # store RGB channels
        self.RGBLeft = {
            'B':bL,
            'G':gL,
            'R':rL
        }
        self.RGBRight = {
            'B':bR,
            'G':gR,
            'R':rR
        }
    
    def setDwtType(self, dwtType:str):
        """设置小波类型

        Args:
            dwtType (str): 小波类型
        """
        self.dwtType = dwtType
    
    def setLowFreqMethod(self, method:str):
        """设置低频段处理方法

        Args:
            method (str): 方法类型
        """
        if method in ['max','avg','weight']:
            self.lowFreqMethod = method
        else:
            raise Exception("method errors, please chose one in [max,avg,weight]")
    
    def setHighFreqMethod(self, method:str):
        """设置高频段处理方法

        Args:
            method (str): 方法类型
        """
        if method in ['max','loc','locWeight']:
            self.highFreqMethod = method
        else:
            raise Exception("method errors, please chose one in [max,loc]")
    
    def ImageDwtTrans(self,channels:str,which=0):
        """图片进行小波变化

        Args:
            channels (str): 通道值，只能为R\G\B
            which (int): 0代表左图片，1代表右图片
        Returns:
            正常返回的时候范围小波变换结果
        """
        if channels not in ['R','G','B']:
            print("chose correct channel!!!")
            return None
        elif which not in [0,1]:
            print("chose correct image!!!")
            return None
        else:
            if which:
                ca,(ch,cv,cd) = dwt2(self.RGBRight[channels],self.dwtType)
            else:
                ca,(ch,cv,cd) = dwt2(self.RGBLeft[channels],self.dwtType)
        return ca,ch,cv,cd
    

    def calLocalVariance(self,v):
        """按照n*n的窗口大小进行局部标准差计算

        Args:
            v (_type_): 计算标准差的二维列表

        Returns:
            返回标准差结果
        """
        rows,cols = v.shape
        result = np.zeros(v.shape)
        for i in range(rows):
            for j in range(cols):
                up = i-3 if i-3>0 else 0
                down = i+3 if i+3<rows else rows
                left = j-3 if j-3>0 else 0
                right = j+3 if j+3<cols else cols
                window = v[up:down,left:right]
                mean,var = cv2.meanStdDev(window)
                result[i,j] = var
        return result


    def dealLowFreq(self,lowFreqLeft,lowFreqRight):
        """dwt的低频处理

        Args:
            lowFreqLeft (_type_): 左边图片的低频段
            lowFreqRight (_type_): 右边图片的低频段

        Returns:
            返回融合图片的低频段
        """
        if self.lowFreqMethod == 'weight':
            # get weight of channels
            meanL,varL = cv2.meanStdDev(lowFreqLeft)
            meanR,varR = cv2.meanStdDev(lowFreqRight)
            weightL,weightR = varL/(varL+varR),varR/(varL+varR)
        result = np.zeros(lowFreqLeft.shape)
        rows,cols = lowFreqLeft.shape
        for i in range(rows):
            for j in range(cols):
                if self.lowFreqMethod == 'max':
                    result[i,j] = max(lowFreqLeft[i,j],lowFreqRight[i,j])
                elif self.lowFreqMethod == 'avg':
                    result[i,j] = (lowFreqLeft[i,j]+lowFreqRight[i,j])/2.0
                elif self.lowFreqMethod == 'weight':
                    result[i,j] = weightL*lowFreqLeft[i,j] + weightR*lowFreqRight[i,j]
        return result

    def dealHighFreq(self,highFreqLeft,highFreqRight):
        """dwt的高频段处理

        Args:
            highFreqLeft (_type_): 左边图片的高频段
            highFreqRight (_type_): 右边图片的高频段

        Returns:
            返回融合图片的高频段
        """
        result = np.zeros(highFreqLeft.shape)
        rows,cols = highFreqLeft.shape
        if self.highFreqMethod == 'loc' or self.highFreqMethod == 'locWeight':
            varianceL = self.calLocalVariance(highFreqLeft)
            varianceR = self.calLocalVariance(highFreqRight)
        for i in range(rows):
            for j in range(cols):
                if self.highFreqMethod == 'locWeight':
                    tempL = round(varianceL[i,j],3)
                    tempR = round(varianceR[i,j],3)
                    if tempL == 0 or tempR == 0:
                        weightL,weightR = 0.5,0.5
                    else:
                        weightL = tempL/(tempR+tempL)
                        weightR = 1-weightL
                    result[i,j] = highFreqLeft[i,j]*weightL +  highFreqRight[i,j] * weightR
                elif self.highFreqMethod == 'max':
                    result[i,j] = highFreqLeft[i,j] if abs(highFreqLeft[i,j]) > abs(highFreqRight[i,j]) else highFreqRight[i,j]
                elif self.highFreqMethod == 'loc':
                    result[i,j] = highFreqLeft[i,j]*varianceL[i,j] +  highFreqRight[i,j] * varianceR[i,j]
        return result



    def ImageProcess(self,channels:str):
        """针对通道的图像处理

        Args:
            channels (str): 通道值，只能为R\G\B

        Returns:
            返回处理后的某通道的逆小波变换图
        """
        if channels not in ['R','G','B']:
            print("chose correct channel!!!")
            return None
        else:
            # get dwt result
            caL,chL,cvL,cdL = self.ImageDwtTrans(channels,0)
            caR,chR,cvR,cdR = self.ImageDwtTrans(channels,1)
            CA = self.dealLowFreq(caL,caR)
            CH = self.dealHighFreq(chL,chR)
            CV = self.dealHighFreq(cvL,cvR)
            CD = self.dealHighFreq(cdL,cdR)
            image = idwt2((CA,(CH,CV,CD)),self.dwtType)
            return image

    def ImageFusion(self,out="imageFusionRes.png"):
        """根据rgb来进行图像融合

        Args:
            out (str, optional): 图片保存路径，默认为"imageFusionRes.png".
        """
        R = self.ImageProcess('R')
        G = self.ImageProcess('G')
        B = self.ImageProcess('B')
        image = cv2.merge([B,G,R])
        cv2.imwrite(out,image)
    
    def storeDwtImage(self,channel:str = 'R',which:int = 0):
        if channel not in ['R', 'G', 'B']:
            raise Exception("channel errors")
        elif which not in [0,1]:
            raise Exception("give right selected image,0 presenting left image or 1 presenting right image")
        ca,ch,cv,cd = self.ImageDwtTrans(channel,which)
        # 因为可能存在值大于255，所以选择将他限制在[0-255]展示
        A = np.uint8(ca/np.max(ca)*255)
        H = np.uint8(ch/np.max(ch)*255)
        V = np.uint8(cv/np.max(cv)*255)
        D = np.uint8(cd/np.max(cd)*255)
        # 水平拼接
        image_top = np.concatenate([A, H], axis=1)
        image_bottom = np.concatenate([V, D], axis=1)
        # 纵向拼接
        image = np.vstack((image_top, image_bottom)) 
        cv2.imwrite('dwtImage.png', image)
        



# 图片质量评估
class EvalImage(object):
    """图像质量评估工具

    Args:
        leftImagePath (str): 左边图片路径
        rightImagePath (str): 右边图片路径
        ImageResPath (str): 融合图片路径
        imageType (str): 图像加载模式, 0表示灰度, 1表示RGB模式
    """
    def __init__(self,leftImagePath,rightImagePath,ImageResPath,imageType=0):
        self.leftImagePath = leftImagePath
        self.rightImagePath = rightImagePath
        self.ImageResPath = ImageResPath
        self.imageType = imageType if imageType==0 else 1
    
    def ImageLoad(self):
        """图片加载

        Returns:
            返回左边、右边和结果的opencv矩阵
        """
        imageL = cv2.imread(self.leftImagePath,self.imageType)
        imageR = cv2.imread(self.rightImagePath,self.imageType)
        imageRes = cv2.imread(self.ImageResPath,self.imageType)
        return imageL,imageR,imageRes
    
    def psnr(self):
        """计算峰值信噪比

        Returns:
            返回左右图和融合图的峰值信噪比均值
        """
        # RGB读入的还需要分通道评估后取均值
        # 峰值信噪比用于衡量图像有效信息与噪声之间的比率，能够反映图像是否失真，PSNR的值越大，表示融合图像的质量越好。
        imageL,imageR,imageRes = self.ImageLoad()
        if self.imageType == 0:
            psnrL = ImageMetrics.peak_signal_noise_ratio(imageL,imageRes,data_range=255)
            psnrR = ImageMetrics.peak_signal_noise_ratio(imageR,imageRes,data_range=255)
        elif self.imageType == 1:
            bL,gL,rL = cv2.split(imageL)
            bR,gR,rR = cv2.split(imageR)
            bS,gS,rS = cv2.split(imageRes)
            psnrBL = ImageMetrics.peak_signal_noise_ratio(bL,bS,data_range=255)
            psnrGL = ImageMetrics.peak_signal_noise_ratio(gL,gS,data_range=255)
            psnrRL = ImageMetrics.peak_signal_noise_ratio(rL,rS,data_range=255)
            psnrL = (psnrBL+psnrGL+psnrRL)/3.0

            psnrBR = ImageMetrics.peak_signal_noise_ratio(bR,bS,data_range=255)
            psnrGR = ImageMetrics.peak_signal_noise_ratio(gR,gS,data_range=255)
            psnrRR = ImageMetrics.peak_signal_noise_ratio(rR,rS,data_range=255)
            psnrR = (psnrBR+psnrGR+psnrRR)/3.0
            
        return (psnrL+psnrR)/2
    
    def ssim(self):
        """计算结构相似性

        Returns:
            返回左右图和融合图的结构相似性均值
        """
        imageL,imageR,imageRes = self.ImageLoad()
        #如果是RGB读入的还需要分通道评估后取均值取值范围为[-1,1]，越接近1，代表相似度越高，融合质量越好
        if self.imageType == 0:
            ssimL = ImageMetrics.structural_similarity(imageL,imageRes,data_range=255)
            ssimR = ImageMetrics.structural_similarity(imageR,imageRes,data_range=255)
        elif self.imageType == 1:
            bL,gL,rL = cv2.split(imageL)
            bR,gR,rR = cv2.split(imageR)
            bS,gS,rS = cv2.split(imageRes)
            ssimBL = ImageMetrics.structural_similarity(bL,bS,data_range=255)
            ssimGL = ImageMetrics.structural_similarity(gL,gS,data_range=255)
            ssimRL = ImageMetrics.structural_similarity(rL,rS,data_range=255)
            ssimL = (ssimBL+ssimGL+ssimRL)/3.0

            ssimBR = ImageMetrics.structural_similarity(bR,bS,data_range=255)
            ssimGR = ImageMetrics.structural_similarity(gR,gS,data_range=255)
            ssimRR = ImageMetrics.structural_similarity(rR,rS,data_range=255)
            ssimR = (ssimBR+ssimGR+ssimRR)/3.0
        return (ssimL+ssimR)/2
    
    def ImageEntropy(self):
        """计算图像结果信息熵

        Returns:
            合成图像信息熵
        """
        #信息熵越高表示融合图像的信息量越丰富，质量越好。
        imageRes = cv2.imread(self.ImageResPath,self.imageType)
        entropy = ImageMeasure.shannon_entropy(imageRes,2)
        return entropy
    
    def ImageMeanAndStd(self):
        """计算均值和标准差

        Returns:
            返回融合图像均值和标准差
        """
        # 均值衡量是一个反映亮度信息的指标,均值适中，则融合图像质量越好。
        # 标准差是度量图像信息丰富程度的一个客观评价指标。该值越大，则图像的灰度级分布就越分散，图像携带的信息量就越多，融合图像质量就越好。
        # RGB图像需要取三通道均值
        imageRes = cv2.imread(self.ImageResPath,self.imageType)
        mean, stddev = cv2.meanStdDev(imageRes)
        return np.mean(np.array(mean),axis=0)[0],np.mean(np.array(stddev),axis=0)[0]

def evalImageFusion(imageLeftPath:str,imageRightPath:str,imageFusionRes:str,evalType=1):
    # 图像评估
    print("Evalation:")
    evalImage = EvalImage(imageLeftPath,imageRightPath,imageFusionRes,evalType)
    print("psnr",evalImage.psnr())
    print("ssim:",evalImage.ssim())
    print("entropy",evalImage.ImageEntropy())  
    mean,stddev = evalImage.ImageMeanAndStd()  
    print("mean:",mean)
    print("stddev:",stddev)

if __name__ == '__main__':
    # 左图路径
    IMAGE_LEFT_PATH = "./images/p27a.jpg"
    # 右图路径
    IMAGE_RIGHT_PATH = "./images/p27b.jpg"
    # 图片融合输出路径
    IMAGE_FUSION_PATH = "./imageFusionRes.png"
    # 评估时图片加载类型，0表示灰度读入，1表示RGB读入
    EVAL_TYPE = 1

    LOW_FREQ_METHOD = ['max','avg','weight']
    HIGH_FREQ_METHOD = ['loc','max','locWeight']
    # 图像融合
    imageFusion = ImageFusion(IMAGE_LEFT_PATH,IMAGE_RIGHT_PATH)
    imageFusion.ImageLoad()

    for low in LOW_FREQ_METHOD:
        for high in HIGH_FREQ_METHOD:
            IMAGE_FUSION_PATH = f"imageFusion_low_{low}_high_{high}.png"
            imageFusion.setLowFreqMethod(low)
            imageFusion.setHighFreqMethod(high)
            imageFusion.ImageFusion(out=IMAGE_FUSION_PATH)
            print(f"低频{low}, 高频{high}")
            evalImageFusion(IMAGE_LEFT_PATH,IMAGE_RIGHT_PATH,IMAGE_FUSION_PATH)

    # # 图像评估
    # print("Evalation:")
    # evalImage = EvalImage(IMAGE_LEFT_PATH,IMAGE_RIGHT_PATH,IMAGE_FUSION_PATH,EVAL_TYPE)
    # print("psnr",evalImage.psnr())
    # print("ssim:",evalImage.ssim())
    # print("entropy",evalImage.ImageEntropy())  
    # mean,stddev = evalImage.ImageMeanAndStd()  
    # print("mean:",mean)
    # print("stddev:",stddev)

    


