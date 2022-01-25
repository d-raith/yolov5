
# coding: utf-8

# In[1]:


# install dependencies as necessary

import torch
import os
from IPython.display import Image, clear_output  # to display images
# clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))


# In[ ]:


# this is the YAML file Roboflow wrote for us that we're loading into this notebook with our data

YML_FOLDER = os.getcwd()
MODEL_YML = YML_FOLDER+os.path.sep+"yolov5s.yaml"
HYP_YML = YML_FOLDER+os.path.sep+"hyp_initial.yaml"

DATASET_PATH =YML_FOLDER+os.path.sep+"dataset"
DATASET_YML = DATASET_PATH+os.path.sep+"data.yaml"





# define number of classes based on YAML
import yaml
with open(MODEL_YML, 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])
    print(num_classes)
#!pip install wandb
print("YML_FOLDER",YML_FOLDER)
print("MODEL_YML", MODEL_YML)
print("HYP_YML", HYP_YML)
print("DATASET_PATH", DATASET_PATH)
print("DATASET_YML", DATASET_YML)


# In[ ]:


#![[ -d $DATASET_PATH ]] && rm -r $DATASET_PATH

#!curl -L "https://app.roboflow.com/ds/diuFUpT435?key=I9oJGiknpO" > roboflow.zip; unzip roboflow.zip -d $DATASET_PATH; rm roboflow.zip


# In[ ]:


#import yaml

#with open(DATASET_YML) as f:
#     list_doc = yaml.safe_load(f)

#list_doc['path'] = DATASET_PATH
#list_doc['train'] = "train/images"
#if 'val' in list_doc:
#    list_doc['val'] = "valid/images"
#if 'test' in list_doc:
#    list_doc['test'] = "test/images"


#with open(DATASET_YML, "w") as f:
#    yaml.dump(list_doc, f)
    



# # Train Custom YOLOv5 Detector
# 
# ### Next, we'll fire off training!
# 
# 
# Here, we are able to pass a number of arguments:
# - **img:** define input image size
# - **batch:** determine batch size
# - **epochs:** define the number of training epochs. (Note: often, 3000+ are common here!)
# - **data:** set the path to our yaml file
# - **cfg:** specify our model configuration
# - **weights:** specify a custom path to weights. (Note: you can download weights from the Ultralytics Google Drive [folder](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J))
# - **name:** result names
# - **nosave:** only save the final checkpoint
# - **cache:** cache images for faster training


# In[ ]:



imgsz=640
batch=12
epochs = 1500





# In[ ]:


get_ipython().system('')


# # Evaluate Custom YOLOv5 Detector Performance

# Training losses and performance metrics are saved to Tensorboard and also to a logfile defined above with the **--name** flag when we train. In our case, we named this `yolov5s_results`. (If given no name, it defaults to `results.txt`.) The results file is plotted as a png after training completes.
# 
# Note from Glenn: Partially completed `results.txt` files can be plotted with `from utils.utils import plot_results; plot_results()`.

# In[ ]:


# Start tensorboard
# Launch after you have started training
# logs save in the folder "runs"
get_ipython().system('kill 201')
get_ipython().magic('load_ext tensorboard')

get_ipython().magic('tensorboard --logdir runs --host 0.0.0.0')


# In[ ]:


import os, sys, pathlib
sys.path.append(str(pathlib.Path("../../../.").resolve()))
print(sys.path)


# In[ ]:


from utils.plots import plot_results  # plot results.txt as results.png
# we can also output some older school graphs if the tensor board isn't working for whatever reason... 

results_folder = os.getcwd()+"/runs/train"

last_run = sorted(os.listdir(results_folder))[-1]
print(last_run)
#last_run = "exp13"
run_dir = results_folder+"/"+last_run

Image(filename=run_dir+"/results.png", width=1000)  # view results.png


# In[ ]:


def get_run_file(*fname):
    return "/".join([run_dir, *fname])


# ### Curious? Visualize Our Training Data with Labels
# 
# After training starts, view `train*.jpg` images to see training images, labels and augmentation effects.
# 
# Note a mosaic dataloader is used for training (shown below), a new dataloading concept developed by Glenn Jocher and first featured in [YOLOv4](https://arxiv.org/abs/2004.10934).

# In[ ]:


# first, display our ground truth data
print("GROUND TRUTH TRAINING DATA:")
Image(filename=get_run_file("val_batch0_labels.jpg"), width=900)


# In[ ]:


# print out an augmented training example
print("GROUND TRUTH AUGMENTED TRAINING DATA:")
Image(filename=get_run_file('train_batch0.jpg'), width=900)


# #Run Inference  With Trained Weights
# Run inference with a pretrained checkpoint on contents of `test/images` folder downloaded from Roboflow.

# # Export Trained Weights for Future Inference
# 
# Now that you have trained your custom detector, you can export the trained weights you have made here for inference on your device elsewhere

# In[ ]:


from IPython.display import FileLink
cwd =os.getcwd()

best = get_run_file("weights", "best.pt")

rel_path_best = best.replace(cwd, ".")


print(rel_path_best)
FileLink(rel_path_best)


# ## Congrats!
# 
# Hope you enjoyed this!
# 
# --Team [Roboflow](https://roboflow.ai)
