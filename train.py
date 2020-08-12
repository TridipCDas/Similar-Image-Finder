from imutils import paths
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd

config={"BATCH_SIZE":32, "DATASET_LOCATION":'dataset/', "FEATURES_OUTPUT":"features.csv"}

#Listing the paths of all our images
imagePaths=list(paths.list_images(config["DATASET_LOCATION"]))

#Loading our pretrained model
model=VGG16(weights="imagenet",include_top=False)

#Extracting the labels from the images
labels={}
for (i,imagepath) in enumerate(imagePaths):
    labels[i]=imagepath.split('/')[1]


    
stack_features=[]

#Loop over the images in batches
for (b,i) in enumerate(range(0,len(imagePaths),config["BATCH_SIZE"])):
    
    print("[INFO] processing batch {}/{}".format(b + 1,
          int(np.ceil(len(imagePaths) / float(config["BATCH_SIZE"])))))
    #Extracting the batch of images
    batchPaths=imagePaths[i:i+config["BATCH_SIZE"]]
    
    images=[]
    
    for path in batchPaths:
        
        #Load the image  while ensuring the image is resized to 224x224 pixels
        image=load_img(path,target_size=(224,224))
        image=img_to_array(image)
        
        #Preprocess the image
        image = np.expand_dims(image, axis=0)
        image=preprocess_input(image)
        
        
        images.append(image)

    # pass the images through the network and use the outputs as
    # our actual features, then reshape the features into a
    batchImages = np.vstack(images)
    features=model.predict(batchImages,batch_size=10)
    features = features.reshape((features.shape[0], 7 * 7 * 512))
    stack_features.append(features)
    
features_matrix=np.vstack(stack_features)

#Processing the column names
cols=[]
for i in range(features_matrix.shape[1]):
    col_name="F{}".format(i)
    cols.append(col_name)
    
#Making a Dataframe with our features    
dframe=pd.DataFrame(data=features_matrix,columns=cols)

#Writing the features to a .csv file
dframe.to_csv(config["FEATURES_OUTPUT"],index=False)