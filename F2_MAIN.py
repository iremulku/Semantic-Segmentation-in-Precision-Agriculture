from __future__ import print_function
import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
#import numpy as np
import datetime
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from F1_UNET_V1_1 import UNetV1
from F3_DATASET import satellitedata
from F4_TRAIN import train_model
from F6_CROSSVAL import CrossVal
#from F7_TEST import test_model
from F7_TEST2 import test_model
from F8_IMAGES import get_images
#from F9_UNET_V2_4 import UNetV2
from F9_UNET_V2_3 import UNetV2
from F10_SEGNET_V1 import SegNet
from F12_DLINKNET_V3 import DinkNet101 
from F20_DILATEDUNET import CamDUNet
from F21_GENERAL_UNET import R2U_Net, AttU_Net, R2AttU_Net
from F22_NESTEDUNET import NestedUNet
from F23_DULANORM_UNET import DualNorm_Unet
from F24_INCEPTION_UNET import InceptionUNet
from F25_SCAG_UNET import AttU_Net_with_scAG
#from F14_DEEPLABV3PLUS_V1 import DeepLabv3_plus
#from F14_DEEPLABV3PLUS_V1_DROP2_Resnet34 import DeepLabv3_plus
from F14_DEEPLABV3PLUS_V4_xception import DeepLabv3_plus
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class Config(object):
    NAME= "dfaNet"

    #set the output every STEP_PER_EPOCH iteration
    STEP_PER_EPOCH = 100
    ENCODER_CHANNEL_CFG=ch_cfg=[[8,48,96],
                                [240,144,288],
                                [240,144,288]]

##############################################################################   
if __name__ == '__main__':

    if (torch.cuda.is_available()):
        print(torch.cuda.get_device_name(0))
    
    bg=datetime.datetime.now()
    bgh=bg.hour
    bgm=bg.minute


     
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #dev = torch.device("cpu")
    device = torch.device(dev)
        
    for i in range(0,2):
        data_folder = os.path.join("../../experiments")
        file_to_open = os.path.join(data_folder, "model{}.txt".format(i))
        with open(file_to_open) as f:
            #lines = f.readlines()
            lines = [line.rstrip() for line in f]
        trainSetSize=int(lines[0])
        fno = int(lines[1])
        fsiz = int(lines[2])
        valRatio = float(lines[3])
        miniBatchSize = int(lines[4])
        n_epochs = int(lines[5])
        learnRate = float(lines[6])
        optimizerType=str(lines[7])
        trainloss=str(lines[8])
        validationloss=str(lines[9])
        accuracy=str(lines[10])
        initialization=str(lines[11])
        step_size=int(lines[12])
        gamma=float(lines[13])
        lim=int(lines[14])
        modeltype=str(lines[15])
        chindex=str(lines[16])
        transfertype=str(lines[17])
    
        tsind,trind,vlind = CrossVal(trainSetSize,fno,fsiz)
        input_images, target_masks, trMeanR, trMeanG, trMeanB = get_images(trainSetSize, fno, fsiz, tsind, trind, vlind, chindex)
             
        params = {'batch_size': miniBatchSize, 'shuffle': False}     
         
        training_set = satellitedata(input_images[trind], target_masks[trind])
        training_generator = DataLoader(training_set, **params)
        
        validation_set = satellitedata(input_images[vlind], target_masks[vlind])
        validation_generator = DataLoader(validation_set, **params)
        
        test_set = satellitedata(input_images[tsind], target_masks[tsind])
        test_generator = DataLoader(test_set, **params)
        
        
        if modeltype=='UNetV1':
            model = UNetV1(classes=1).to(device)
        elif modeltype=='UNetV2':
            model = UNetV2(classes=1).to(device)
        elif modeltype=='SegNet':
            model = SegNet(classes=1).to(device)  
        elif modeltype=='DinkNet101':
            model = DinkNet101(num_classes=1).to(device)
        elif modeltype=='DeepLabv3_plus':
            model = DeepLabv3_plus(num_classes=1, small=True, pretrained=True).to(device) 
        elif modeltype=='CamDUNet':
            model = CamDUNet().to(device)
        elif modeltype=='R2U_Net':
            model = R2U_Net(img_ch=3,output_ch=1).to(device)        
        elif modeltype=='AttU_Net':
            model = AttU_Net(img_ch=3,output_ch=1).to(device)                
        elif modeltype=='R2AttU_Net':
            model = R2AttU_Net(img_ch=3,output_ch=1).to(device)  
        elif modeltype=='NestedUNet':
            model = NestedUNet(in_ch=3, out_ch=1).to(device)             
        elif modeltype=='DualNorm_Unet':
            model = DualNorm_Unet(n_channels=3, n_classes=1).to(device)               
        elif modeltype=='InceptionUNet':
            model = InceptionUNet(n_channels=3, n_classes=1, bilinear=True).to(device)            
        elif modeltype=='AttU_Net_with_scAG':
            model = AttU_Net_with_scAG(img_ch=3, output_ch=1,ratio=16).to(device)             
         
        
        
       
###############################################################################           
        def init_weights(m):
            if initialization == 'xavier_uniform_':
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
        
            if initialization == 'xavier_normal_':
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
        
            if initialization == 'kaiming_uniform_':
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_uniform_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
          
            if initialization == 'kaiming_normal_':
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)    
###############################################################################
                        
        if transfertype=='yestr':    
            model.load_state_dict(torch.load(os.path.join(data_folder, "2021_3_16_10_32.pt")))
        elif transfertype=='notr':
            model.apply(init_weights)            

        if optimizerType =='Adam':
            optim = torch.optim.Adam(model.parameters(),learnRate)
        elif optimizerType =='SGD':
            optim = torch.optim.SGD(model.parameters(),learnRate)
     
        scheduler = StepLR(optim, step_size, gamma)
        
        d=datetime.datetime.now()
        pathm = os.path.join(data_folder, "{}_{}_{}_{}_{}_model{}".format(d.year, d.month, d.day, d.hour, d.minute, i))
        os.mkdir(pathm)

        os.path.join(pathm, "lrFile.txt")
        lrFile = open("lrFile.txt","w")
        os.path.join(pathm, "trainaccFile.txt")
        trainaccFile = open("trainaccFile.txt","w")
        os.path.join(pathm, "valaccFile.txt")
        valaccFile = open("valaccFile.txt","w")
        os.path.join(pathm, "trainepochFile.txt")
        trainepochFile = open("trainepochFile.txt","w")
        os.path.join(pathm, "trainFile.txt")
        trainFile = open("trainFile.txt","w")
        os.path.join(pathm, "valFile.txt")
        valFile = open("valFile.txt","w")
        train_model(n_epochs, trainloss, validationloss, accuracy, model, scheduler, lrFile, training_generator, optim, lim, trainFile, trainaccFile, trainepochFile, validation_generator, valFile, valaccFile, pathm, i, modeltype)
        trainFile.close() 
        valFile.close()
        trainaccFile.close() 
        valaccFile.close()
        trainepochFile.close()
        lrFile.close()
        
    
        os.path.join(pathm, "testaccFile.txt")
        testaccFile = open("testaccFile.txt","w")
        os.path.join(pathm, "testFile.txt")
        testFile = open("testFile.txt","w")
        test_model(test_generator, lim, testFile, testaccFile, i, modeltype, pathm, trMeanR, trMeanG, trMeanB)
        testFile.close()
        testaccFile.close()
    
        x=[]
        with open("trainFile.txt") as f:
            lines = f.readlines()
            for l in lines:
                x.append(float(l))
                
        y=[]
        with open("valFile.txt") as f:
            lines = f.readlines()
            for l in lines:
                y.append(float(l)) 
                
        tt=[]
        with open("testFile.txt") as f:
            lines = f.readlines()
            for l in lines:
                tt.append(float(l))             
                
        z=[]
        with open("lrFile.txt") as f:
            lines = f.readlines()
            for l in lines:
                z.append(l) 
                
        xx=[]
        with open("trainaccFile.txt") as f:
            lines = f.readlines()
            for l in lines:
                xx.append(float(l))
                
        yy=[]
        with open("valaccFile.txt") as f:
            lines = f.readlines()
            for l in lines:
                yy.append(float(l))
                
        ta=[]
        with open("testaccFile.txt") as f:
            lines = f.readlines()
            for l in lines:
                ta.append(float(l))            
                
        e1=[]
        with open("trainepochFile.txt") as f:
            lines = f.readlines()
            for l in lines:
                e1.append(float(l))
        
                
        def logfile():
            a=datetime.datetime.now()
            myfile=os.path.join(pathm, "{}_{}_{}_{}_{}.txt".format(a.year, a.month, a.day, a.hour, a.minute))
            LogFile = open(myfile,"w");
            LogFile.write("Date:"+str(datetime.date.today())+"\n")
            LogFile.write("Ending Time:"+str(a.hour)+":"+str(a.minute)+"\n") 
            LogFile.write("Starting Time:"+str(bgh)+":"+str(bgm)+"\n") 
            LogFile.write("Data set size:"+str(trainSetSize)+"\n")
            LogFile.write("Fold number:"+str(fno)+"\n")
            LogFile.write("Fold number:"+str(fsiz)+"\n")
            LogFile.write("Number of validation images:"+str(len(vlind))+"\n")
            LogFile.write("Number of training images:"+str(len(trind))+"\n")
            LogFile.write("Mini batch size:"+str(miniBatchSize)+"\n")
            LogFile.write("Type of initialization:"+initialization+"\n")
            LogFile.write("Test accuracy:"+str(ta)+"\n")
            LogFile.write("Learning rate:"+str(learnRate)+"\n")
            LogFile.write("Model version:"+str(modeltype)+"\n")
            LogFile.write("Optimizer type:"+optimizerType+"\n")
            LogFile.write("Total number of epochs:"+str(n_epochs)+"\n")
            LogFile.write("Training loss function:"+str(trainloss)+"\n")
            LogFile.write("Validation loss function:"+str(validationloss)+"\n")
            LogFile.write("Accuracy function:"+str(accuracy)+"\n")  
            LogFile.write("Channel index:"+str(chindex)+"\n")
            LogFile.write("Transfer:"+str(transfertype)+"\n")
            LogFile.write("Model Summary:"+"\n"+str(model)+"\n")
            for i in range(len(z)):
                LogFile.write(str(z[i]))
            LogFile.close();
        
        logfile()
        
        
        plt.plot(x,"k-", label="Train Loss")
        plt.plot(y,"r--", label="Validation Loss")
        plt.title('Learning Curves')
        plt.legend(loc="upper left")
        l_curve = 'learning_curves.png'
        plt.savefig(os.path.join(pathm, l_curve))
        plt.show()
            
        plt.plot(xx, "k-", label="Train Accuracy")   
        plt.plot(yy, "r--", label="Validation Accuracy") 
        plt.title('Accuracy Curves')
        plt.legend(loc="upper left")
        a_curve = 'accuracy_curves.png'
        plt.savefig(os.path.join(pathm, a_curve))
        plt.show() 
        
        print("Memory allocated before model {}".format(i),torch.cuda.memory_allocated())
        del model
        torch.cuda.empty_cache()
        print("Memory allocated after model {}".format(i),torch.cuda.memory_allocated())
        
        

  
