#### Issue 310 梯度检查

-----

参照 `ch12 level`有关NeuralNet_3_0的相关代码

使用`MiniFramework`构建相同的网络

```python
net = NeuralNet_4_0(params, "SAFE")
fc1 = FcLayer_1_0(num_input, num_hidden1, params)
net.add_layer(fc1, "fc1")
r1 = ActivationLayer(Sigmoid())
net.add_layer(r1, "r1")

fc2 = FcLayer_1_0(num_hidden1, num_hidden2, params)
net.add_layer(fc2, "fc2")
r2 = ActivationLayer(Sigmoid())
net.add_layer(r2, "r2")

fc3 = FcLayer_1_0(num_hidden2, num_output, params)
net.add_layer(fc3, "fc3")
r3 = ActivationLayer(Sigmoid())
net.add_layer(r3, "r3")
```

10次测试结果如下

> diference =2.0312500850274795e-06
> Acceptable, but a little bit high.

> diference =1.8406475444289848e-06
> Acceptable, but a little bit high.

> diference =1.9322746575442127e-06
> Acceptable, but a little bit high.

> diference =1.9368578199186335e-06
> Acceptable, but a little bit high.

> diference =1.9518567669651487e-06
> Acceptable, but a little bit high.

> diference =2.0586026954312742e-06
> Acceptable, but a little bit high.

> diference =1.9195015372704093e-06
> Acceptable, but a little bit high.

> diference =1.9398921246159497e-06
> Acceptable, but a little bit high.

> diference =1.8445311710999132e-06
> Acceptable, but a little bit high.

> diference =2.2559941786168695e-06
> Acceptable, but a little bit high.

可以看到结果相对稳定并且该结果和测试`NeuralNet_3_0`的结果相似而且可以接受。因此`MiniFramework`的反向传播算法是正确的


