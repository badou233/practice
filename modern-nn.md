# 批量归一化
固定每个小批量（barchsize) 中的均值和方差
$$\mathrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$
然后再用γ和β做一下调整（γ和β是模型中可以学习的参数，与inputs无关），以让调整后的的均值和方差适应整个神经网络。

使用批量规范化时，要注意每个批量不能过小，过小的话每个随机批量内部的均值和方差将不再有对整个输入集的代表性。

*作用*：1.批量规范化可以使学习率设的更大，以加快模型的训练速度      
2.减少模型对参数w变化的敏感度  
3.在激活函数之前使用，减少激活函数导致的梯度损失
# ResNet
![结构](image\resnet.jpg)
加入残差块，使每一层附加层包含原始层的输出x

作用：1.可以实现更深层数的神经网络
2.可以避免梯度消失：![结构](image\resnet防梯度消失.jpg)