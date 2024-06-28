# Code for MDS
本代码仓库是一个面向统计人员的开源库，特别是研究方向为研究方向为MDS。

提供了一个整洁的代码库来验证Binomial、Poisson和Normal情况下的Simulation，它涵盖了五个主流任务:**parameter estimation, imputation, classification, and matrix completion.**

## 使用方法

1. 
   安装Python 3.8。为方便起见，执行以下命令。

```python
pip install -r requirements.txt
```

2. 准备数据. 你可以从以下渠道获取数据 [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing), [[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/84fbc752d0e94980a610/) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy). 然后将下载的数据放在文件夹下 `./data_provider/dataset`. 下面是支持的数据集的摘要。当需要自定义的数据集时，需要将

<p align="center">
<img src=".\pic\dataset.png" height = "200" alt="" align=center />
</p>

3. 在`./scripts/`文件夹下提供了所有基准测试的实验脚本。您可以将实验结果复制为以下示例:

   ```python
   # long-term forecast
   bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
   # short-term forecast
   bash ./scripts/short_term_forecast/TimesNet_M4.sh
   # imputation
   bash ./scripts/imputation/ETT_script/TimesNet_ETTh1.sh
   # anomaly detection
   bash ./scripts/anomaly_detection/PSM/TimesNet.sh
   # classification
   bash ./scripts/classification/TimesNet.sh
   ```

   若为Window操作系统，您可以按以下代码执行示例：

   ```python
   cd C:\Users\user\Desktop\Time-Series
   python -u run.py --task_name long_term_forecast  --model informer --data ETTh1
   ```

4. 开发你自己的模型。

- 将模型文件添加到`./models`文件夹中。你可以按照 `./models/Transformer.py`.
- 将新添加的模型包含在 `./exp/exp_basic.py`的 `Exp_Basic.model_dict`中
- 在文件夹下创建相应的脚本 `./scripts`.

## 联络
如有任何疑问或建议，欢迎联络，或者可以提相关的Issues，本人在看见后会尽可能解答

- 原作者：Yinghang Chen ([brainiaccc@foxmail.com]())

## 关于

国家重点研发计划项目(2021YFB1715200)资助。

这个库是基于以下代码库构建的:

- Forecasting: https://github.com/thuml/Autoformer

- Anomaly Detection: https://github.com/thuml/Anomaly-Transformer

- Classification: https://github.com/thuml/Flowformer

所有实验数据集都是公开的，可以下链接获取:

- Long-term Forecasting and Imputation: https://github.com/thuml/Autoformer

- Short-term Forecasting: https://github.com/ServiceNow/N-BEATS

- Anomaly Detection: https://github.com/thuml/Anomaly-Transformer

- Classification: https://www.timeseriesclassification.com/
