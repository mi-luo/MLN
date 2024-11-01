
## install
```python 
git clone https://github.com/mi-luo/MLN  # clone  
cd Flame-detection  
pip install -r requirements.txt  #  install
```


## Pretrained Checkpoints：

Download our model weights on Baidu Cloud Drive: https://pan.baidu.com/s/127htvPFgNs1AERfzn92Ltw?pwd=56tm 提取码: 56tm 


## Dataset
 Download our val datasets on Baidu Cloud Drive: https://pan.baidu.com/s/1pNjCiCOO-7ak_JotQSeL_A?pwd=7qe2 提取码: 7qe2 
 
## Train

GPU：
```python 
python train.py \
    ${--weights ${initial weights path}} \
    ${--cfg ${model.yaml path}} \
    ${--data ${dataset.yaml path}} \
    --device 0
```


## Val

GPU：
```python 
python val.py
    ${--data ${dataset.yaml path}} \
    ${--weights ${model.pt path(s)}} \
    --device 0
```

## Contact
For MLN bug reports and feature requests, and feel free to ask questions and engage in discussions!
