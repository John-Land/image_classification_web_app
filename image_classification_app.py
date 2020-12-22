import streamlit as st
import tensorflow as tf #deep learning
from tensorflow import keras #deep learning
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick

def main():
    
    
    #helper functions
    
    #download resnet50 model
    @st.cache(suppress_st_warning=True)
    def download_model():
        model = tf.keras.applications.resnet50.ResNet50()
        return model
    
    #download image from provided user url
    def download_image(image_url):
        # Adding information about user agent
        opener=urllib.request.build_opener()
        opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
        urllib.request.install_opener(opener)
    
        # setting filename and image URL
        filename = 'sample_image.jpg'
        image_url = image_url
    
        # calling urlretrieve function to get resource
        urllib.request.urlretrieve(image_url, filename)

        # load image
        img = image.load_img(filename)
        return img
    
    #preprocess image for model 
    def preprocessed_image(img):
        img_array = image.img_to_array(img) #convert image to array
        img_batch = np.expand_dims(img_array, axis=0)  #add additional dimension for batch size
        img_resized = tf.image.resize(img_batch, [224, 224]) #resize image for the model
        img_preprocessed = preprocess_input(img_resized) #preprocess the image
        return img_preprocessed
    
    #predict class lebel of image
    def predict_image_label(preprocessed_img):
        #predict and decode predictions
        prediction = model.predict(preprocessed_img)
        decoded_prediction = decode_predictions(prediction, top=5)[0]
        
        #prediction list of class name and class probability
        prediction_list = []
        for i in range(5):
            prediction_list.append(decoded_prediction[i][1:3])
        
        return prediction_list

    
    #web app
   
    st.set_page_config(layout="centered")

    st.title("Image Classification Web App")
    st.sidebar.title("Image Classification Web App")
    st.markdown("Image Classifcation with the ResNet50 model")
    st.sidebar.markdown("Image Classifcation with the ResNet50 model")
    
    # user input for image URL
    st.header("Input image URL")
    user_input = st.text_input("Paste Imange Web URL here", "https://images.freeimages.com/images/large-previews/be7/puppy-2-1456421.jpg")

    if user_input:
        image_url = str(user_input)
    
        #download image
        img = download_image(image_url)
    
        # plot downloaded image
        fig, ax = plt.subplots()
        ax = plt.imshow(img)
        plt.axis('off')
        st.pyplot(fig)
    
        # download model
        model = download_model()
    
        #preprocess image for model 
        preprocessed_img = preprocessed_image(img)
    
        st.subheader("Model Prediction")
        #predict class lebel of image
        prediction = predict_image_label(preprocessed_img)
        prediction = pd.DataFrame(prediction)
        prediction.columns = ['Class', 'Probability']
        prediction['Probability'] = prediction['Probability']
        
        #plot predictions
        fig, ax = plt.subplots()
        ax = sns.barplot(x='Probability', y="Class", data=prediction, palette='Blues_r')
        ax.set(xlim=(0, 1), ylabel="")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
     
        sns.despine(left=True, bottom=True)
        ax.grid(False)

        for rect in ax.patches:
            # Get X and Y placement of label from rect.
            y_value = rect.get_y() + rect.get_height() / 2
            x_value = rect.get_width()
    
            label = "{0:.1%}".format(x_value)
            
            ax.annotate(label,(x_value+0.05, y_value),ha='center',va='center')
        
        st.pyplot(fig)
    
    
    
if __name__ == '__main__':
    main()