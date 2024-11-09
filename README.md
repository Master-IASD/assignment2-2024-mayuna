# DataLabAssignement2



## Code to verify fid

first, you need 
pip install pytorch-fid

then you need download my real_images dataset, its the picture of MNIST,
thecan use code like this 

python -m pytorch_fid real_images samples --device cuda:0

if you donot have gpu 
python -m pytorch_fid real_images samples


## generate.py
Use the file *generate.py* to generate 10000 samples of MNIST in the folder samples. 
Example:
  > python3 generate.py --bacth_size 64

## requirements.txt
Among the good pratice of datascience, we encourage you to use conda or virtualenv to create python environment. 
To test your code on our platform, you are required to update the *requirements.txt*, with the different librairies you might use. 
When your code will be test, we will execute: 
  > pip install -r requirements.txt


## Checkpoints
Push the minimal amount of models in the folder *checkpoints*.

