----------------------------------------------------------------------
----------------------------------------------------------------------
This ReadMe-file has several sections which are enumerated as follows:
----------------------------------------------------------------------
----------------------------------------------------------------------

1) Data Augmentation        
	
	Focuses on the augmentation of the RawData used in the project


2) Mantis Detect using a CNN
	
	Focuses on the CNNs created during this project

-----------------------------------------------------------------------
## 1 ) Data Augmentation
-----------------------------------------------------------------------
-----------------------------------------------------------------------

1. Note that if you want to generate more data with the Augmentaion.py file
we recommend to copie an existing rawdata folder (e.g. RawData_2C or
RawData_17C) and rename it to your own preference.

2. Open the Augmentation.py skript using your editor of choise

3. Run the Code. This will set your working directory to the source file 
location and show you an example augmentation

4. In the console run the function create_augmented_data(folder,number_wanted_files):
for the folder parameter set the name of your folder you want to augment 
data in. The number_wanted_files parameter stands for your wished number
of data in each subfolder at the end of augmentation.

5. You will see that the code is runing when you look at the console and
will be able to see howmany new data was generated using augmentation.

-----------------------------------------------------------------------

## 2) Mantis Detect using a CNN
-----------------------------------------------------------------------
-----------------------------------------------------------------------

1. Open the file MantisDetect.py in your editor of choice 

2. If you want to see how the training of the CNNs is archived in this 
project you can simply run the entire code.

3. NOTE If you wish to save your model please end-comend the lines 267 and 338
in the code.

4. NOTE If you wish to load the models to use the utility functions without 
training you can also load the CNNs by end-commend the lines 268 and 
339. Feel free to play around with the prediction_visualization function.



