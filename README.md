These codes were created to insure accurate bottom detection on split-beam EK80 and EK60 echosounder echograms. 
To do so, we used a convolutional neural network with a U-Net architecture. .
The libraries needed to run the scripts are installed with the following command :

__pip install numpy matplotlib scikit-image scikit-learn tensorboard torch torchvision pillow tqdm scipy h5py tifffile dask argparse__

The aim of each of the scripts hereby provided is listed below :
- echograms.py : creation of the Echogram class that charges the data for training, validation and testing purposes.
- metric.py : functions that calculate the Intersection over Union (IoU) and pixel accuracy during validation phases
- unet.py : contains the Unet class of the model architecture
- train.py : training script. It calls all of the previous scripts to proceed with the training with parameters are defined in the parse_args() function
- inference.py : script that is used to test the trained neural network both on both single samples and on a lot of samples
- inference_new_cruise.py : script that is used to test the trained neural network both on a new cruise from Matecho Matlab matrix Echogram.mat
- parametrage.py : script to compute the most fitting parameters for the neural network. 

The last two scripts are the ones that you need to launch to create, train and test your own CNN U-Net. 
Before running them, insure that you have modified the args.rootdir in the main of the code to put the directory in which you put the codes. 
Moreover, modify the root_dir in functions get_trainloader (train.py), get_validationloader (train.py) and get_testloader (inference.py). 
(line : data = Echograms(data_type='validate', root_dir='D:/PFE/Codes_finaux/data/'))

The data furnished is a sample of 200 kHz echograms collected during the SCOPES (https://doi.org/10.17600/18000662), FAROFA3 (https://doi.org/10.17882/71024)
and PIRATA (https://doi.org/10.17882/71379) cruises performed by the IRD.

 
