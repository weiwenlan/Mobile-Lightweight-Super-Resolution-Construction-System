# Mobile-Lightweight-Super-Resolution-Construction-System
These codes enables you to construct super-resolution image on the portable devices, tested on HUAWEI Mate 20. The super-resolution part use the same result of our paper [Lightweight Image Super-Resolution With Mobile Share-Source Network](https://ieeexplore.ieee.org/abstract/document/9045996). For the portable caculation part, we implemented the [NCNN](https://github.com/Tencent/ncnn) and transfered the model.
 - MSSN_code contains all codes associate with MSSN super-resolution reconstruction network
 - MobileNetSSD contains all codes to deploy model on the Android platform

## Abstract

The current mainstream players have already had the function of improving the image quality, but most of them are based on improving the sharpness and color saturation of the image, and there is no real improvement in the resolution. In some cases, it even brings color oversaturation, loss of details and other issues. Another part of the player uses the cloud computing method to enhance the resolution. It not only occupies a large amount of network resources, but also needs to construct a server, and has certain requirements for the user's working environment. The method we use is based on a deep learning network. After compression, it can quickly perform super-resolution reconstruction. It only takes up a small amount of local resources and can achieve high-definition video output without network connection or GPU operation. This allows us to put the network on the mobile side and other low-bit network hardware, and the lighter weight network also indicates the direction of this technology. Breaking away from the huge computational hardware and redundant parameters, the entire model can have a wider range of applications and scenarios.

# Arc
## Lightweight Image Super-Resolution With Mobile Share-Source Network
Improvement made based on the MobilenetV2 and EDSR.
<div align=center><img width="300"  src="https://github.com/weiwenlan/Mobile-Lightweight-Super-Resolution-Construction-System/blob/main/MAWRU.png"/></div>
<div align=center><img  src="https://github.com/weiwenlan/Mobile-Lightweight-Super-Resolution-Construction-System/blob/main/NETWORK.png"/></div>


## ANDROID



# DEMO
 This is the final demo of our project! By using the App, you could easily reconstruct all images with a simple click!
<div align=center><img width="400" src="https://github.com/weiwenlan/Mobile-Lightweight-Super-Resolution-Construction-System/blob/main/gif.gif"/></div>
