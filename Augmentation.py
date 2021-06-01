# -*- coding: utf-8 -*-
"""



"""
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import tensorflow as tf 
import cv2

def create_augmented_data(folder,number_wanted_files):
    """
    This function takes a folder which should contain subfolders and a 
    number of wanted files per subfolder as an input. It reads the images and 
    in the folder and creates augmentations of the original images until the 
    wanted number of files is reached.
    """
    augment_count = 0
    for subdir, dirs, files in os.walk(folder):
        
        for di in dirs:
            #count files in directory
            path = os.path.join(folder,di)
            num_of_files = len(os.listdir(os.path.join(folder,di)))
            print(f'\nThere are {num_of_files} in {di}, creating additional {number_wanted_files-num_of_files} augmented images')
            files = os.listdir(os.path.join(folder,di))
            # print(files)
            #select files to augment
            files_to_augment = np.random.randint(0,num_of_files-1,number_wanted_files-num_of_files)
            #incresse amount of augmentions
            augment_count += number_wanted_files-num_of_files
            files_in_folder = len(os.listdir(os.path.join(folder,di)))
            
            c = 0
            while files_in_folder < number_wanted_files:
                for i in range(num_of_files):
                    c += 1
                    # img = Image.open(os.path.join(path,files[files_to_augment[i]]))
                    img = cv2.imread(os.path.join(path,files[files_to_augment[i]]))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    augmentation_number = np.random.randint(4)
                    new_img = augmentation(img,augmentation_number)
                    new_img.save(f'{os.path.join(path,files[files_to_augment[i]]).strip(".jpg")}_AUG{c}.jpg')
                
                    files_in_folder = len(os.listdir(os.path.join(folder,di)))
                        
                    #break out of forloop
                    if files_in_folder == number_wanted_files:
                        print('Stop for-loop')
                        break
                    
                #break out of while loop   
                if files_in_folder == number_wanted_files:
                    print('Stop while-loop')
                    break
                
    print(f'Created {augment_count} Images with Augmentation !!!')
    return 

def augmentation(imgarray, augmentation_number):
    """
    augments image based on number input
    
     0 = rotate between -5 and 5° and change saturation of the image and flip left to right
     1 = rotate between -5 and 5° and change birghtness of the image and flip up to down
     2 = crop the image and move parts of the image
    >2 = transpose and random rotation of -15 to 15° rotation
    """
    # flip left right and saturation adjust
    if augmentation_number == 0:
        random_angle = np.random.uniform(-5,5)
        new = ndimage.rotate(imgarray,random_angle,mode ='nearest')
        new = tf.image.random_saturation(new,0.5, 1.5)
        new = np.array(new)
        new = np.fliplr(new)
    
    # flip up down and brightness adjust
    elif augmentation_number == 1:
        random_angle = np.random.uniform(-5,5)
        new = ndimage.rotate(imgarray,random_angle,mode ='nearest')
        new = tf.image.random_brightness(new, 0.3)
        new = np.array(new)
        new = np.flipud(new)
        
    elif augmentation_number == 2:
        # cropping out random part and resizing with kNN
        dim1 = len(imgarray)
        dim2 = len(imgarray[0])
        new = tf.image.random_crop(imgarray,(round(dim1*0.6),round(dim2*0.6),3))
        new = tf.image.resize(new,[dim1,dim2], method = 'nearest')
        new = np.array(new)

    else:
        random_angle = np.random.uniform(-15,15)
        new = tf.image.transpose(imgarray)
        new = np.array(new)
        new = ndimage.rotate(new,random_angle,mode ='nearest')
        new = np.array(new)
        
    #create image
    new_image = Image.fromarray(new, 'RGB')
    
    return new_image


"""
Setup the Working Enviornment
"""
#set path to file location
#path to file
abspath = os.path.abspath(__file__)
#path to folder the file is in
dname = os.path.dirname(abspath)
#change working directory to file location
os.chdir(dname)

"""
Example
"""
#read and lookup image
img = Image.open('Test_Images\\Test1.jpg')
img = np.array(img)

#flip ndarray
Aug_1 = augmentation(img,0)

#rotate ndarry
Aug_2 = augmentation(img,1)

# random cropping and resizing
Aug_3 = augmentation(img,2)

# transpose and random rotation
Aug_4 = augmentation(img,3)

plt.figure('Augmentations', figsize=(20, 20))
plt.subplot(151)
plt.imshow(img)
plt.axis('off')
plt.title('Original\n', fontsize=20)
plt.subplot(152)
plt.imshow(Aug_1)
plt.axis('off')
plt.title('Flipped: left, right\nsaturation change', fontsize=20)
plt.subplot(153)
plt.imshow(Aug_2)
plt.title('Flipped: top, down\nbrightness change', fontsize=20)
plt.axis('off')
plt.subplot(154)
plt.imshow(Aug_3)
plt.title('Cropped \nand resized', fontsize=20)
plt.axis('off')
plt.subplot(155)
plt.imshow(Aug_4)
plt.title('Transpose \nand rotated', fontsize=20)
plt.axis('off')
