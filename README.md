## Code explanations
* [`main_script_train.py`](<main_script_train.py>): build model training scripts, you can easily set up and let the code run.
* [`train_model.py`](<train_model.py>): model training function, the dataset setting is here
* [`utils.py`](<utils.py>): basic files for various modules, including definition of data sets, model definition, data partitioning, model training, etc.
* [`inference_biu.py`](<inference_biu.py>): inference the trained model on the test set and save the results for evaluation
* [`eval_biu.py`](<eval_biu.py>): model evaluation module, uses 95% bootstrap confidence interval, and calculates Precision, recall, and F1-score in a weighted average way. Save the final performance into a txt file for generating LaTex files
* [`perf2latex.py`](<perf2latex.py>): read performance txt files and generate LaTex files

## Related codes

1. **[TorchVision](<https://pytorch.org/vision/stable/models.html>)**: https://pytorch.org/vision/stable/models.html

We use the ImageNet pretrained weighted for different models:
```python
from torchvision import models
# Pretrained models
vgg_model = models.vgg16(pretrained=True)
resnet_model = models.resnet50(pretrained=True)
densenet_model = models.densenet121(pretrained=True)
# And you can make better deep learning models by using transfer learning
```