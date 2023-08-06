from __future__ import print_function
import torch 
import os
import torch.nn as nn
import numpy as np
from F1_UNET_V1_1 import UNetV1
from F5_JACCARD import Jaccard
from F9_UNET_V2_4 import UNetV2
from F10_SEGNET_V1 import SegNet


dev = "cuda:0"  
device = torch.device(dev) 

def test_model(test_generator, lim, testFile, testaccFile, i, modeltype, pathm):
    
    if modeltype=='UNetV1':
        net = UNetV1(classes=1).to(device)
    elif modeltype=='UNetV2':
        net = UNetV2(classes=1).to(device) 
    elif modeltype=='SegNet':
        net = SegNet(classes=1).to(device)        
        
    net.load_state_dict(torch.load(os.path.join(pathm, "Finaliremmodel{}.pt".format(i))))

    jI = 0
    totalBatches = 0
    test_losses = []
    net.eval()
    with torch.no_grad():
        t_losses = []
        for testim, testmas in test_generator:
            images=testim.to(device)
            masks=testmas.to(device)
            outputs = net(images)
            losst=nn.BCEWithLogitsLoss()
            output = losst(outputs, masks)
            t_losses.append(output.item())
            batchLoad = len(masks)*lim*lim
            totalBatches = totalBatches + batchLoad
            thisJac = Jaccard(torch.reshape(masks,(batchLoad,1)),torch.reshape(outputs,(batchLoad,1)))*batchLoad
            jI = jI+thisJac.data[0]
                 
    dn=jI/totalBatches
    dni=dn.item()
    test_loss = np.mean(t_losses)
    test_losses.append(test_loss)
    testFile.write(str(test_losses[0])+"\n")
    testaccFile.write(str(dni)+"\n")
    print("Test Jaccard:",dni)
