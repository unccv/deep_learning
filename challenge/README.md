# Return of the Original Problem

## About This Challenge


## Data


## Your Mission 

Your job is to train a deep learning model to classify images of bricks, balls, or cylinders against a cluttered background uning **tensorflow**. You are being provided a pre-trained AlexNet model to use a starting point, located in `alexnet.py`. To use the pretrained model, you'll need to download the weight file `bvlc_alexnet.npy` and place it in the challege directory, it's available [here](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/). 

I've provided a script `sample_model.py`, that contains a Model class for you to modify for this challenge. There are 3 key steps you need to complete: 

### 1. Setup Your Graph

The sample model script will not run as is, becuase you need to complete the model graph in the `build_graph` method. Notice that in the parameter setup section of the `__init__` method I've included a parameter `num_layers_to_load` - this controls how many of the 8 layers of AlexNet we load pretrained weights for. I've set this value to 5, meaning I'm loading up the 5 convolutional layers of AlexNet, but none of the fully connected layers. Feel free to experiment with this value. 

If you do choose to change `num_layers_to_load`, please change the `alexnet_out` line accordingly. Refer to `alexnet.py` to see which tensor you would like to grab. For example, if you chose to only use the first 4 layer of AlexNet, you would want to change this line to  `alexnet_out = self.AN.conv4`. Finally, note that since different layers have different ouptut dimensions, the `alexnet_out` you choose will have some impact on the rest of your graph. You can see the dimension of `alexnet_out` with by printing the tensor: `print(alexent_out)`. 

The final layer of your network should output predictions to the variable `self.yhat`. `yhat` should be of the same dimension as `y`. I encourage you to experiment with simpler architectures first.  

### 2. Setup Your Cost Function 

You will need to setup a cost function in the `train` method. Be sure that your cost is named `cost`, as this name is used to compute gradients. 

### 3. Reduce Overfitting

Since we don't have a very large dataset, your model may overfit. If this is the case, you may need to reduce overfitting to achieve a high accuracy. [Chaper 7](https://www.deeplearningbook.org/contents/regularization.html) of Goodfellow et. al. covers many techniques to reduce overfitting. You are free to modify the model class as you see fit, but be sure not to change the name of the class or break the predict method, as this will be used in evaluation. We will run a script very similar to `eval.py` on the evaluation server, so please run this script locally before submitting to make sure everything works. Note that the test data on the evaluation server is different from the date you've been given for training. **Note** if you choose to use dropout, I ran into gradient issues with `tf.layers.dropout`, and had much better luck with `tf.nn.dropout`. Tensorflow is a great tool, but is very much a work in progress, so issues like this do arise. 



Finally, we've provided a simple training script `train.py` you can use to train your models from the terminal:

```
python train.py
```

We have built in fairly extensive tensorboard visualization you can use to visualize performance as you tune your models. You can launch tensorboard from the challenge directory with:

```
tensorboard --logdir tf_data
```


## Deliverables

1. Your modified version of `sample_model.py`. 
2. Your tensorflow checkpoint files in a folder called `checkpoints` . We will use the `restore_from_checkpoint` method in `sample_model` to load your checkpoint and pass in the checkpoint_dir `checkpoints` into the `Model.predict` method. 


## Grading

