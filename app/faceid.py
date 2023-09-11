from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

# Build layout
class CamApp(App):
    def build(shelf):
        shelf.img1 = Image(size_hint=(1,.8))
        shelf.button = Button(text="verify",on_press=shelf.verify,size_hint=(1,.1))
        shelf.verification_label = Label(text="Verification Uninitiated",size_hint=(1,.1))
        # load keras model
        shelf.model = tf.keras.models.load_model('siamese_model.h5',
                            custom_objects={
                                'L1Dist':L1Dist,
                                'BinaryCrossentropy':tf.losses.BinaryCrossentropy
                            })
    
    # add items to layout
        layout = BoxLayout(orientation="vertical")
        layout.add_widget(shelf.img1)
        layout.add_widget(shelf.button)
        layout.add_widget(shelf.verification_label)

        shelf.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(shelf.update,1.0/33.0)
        
        return layout
    def update(shelf,*args):
        ret,frame = shelf.capture.read()
        frame = frame[120:120+250,200:200+250,:]
        buf = cv2.flip(frame,0).tostring()
        img_texture = Texture.create(size=(frame.shape[1],frame.shape[0]),colorfmt='bgr')
        img_texture.blit_buffer(buf,colorfmt='bgr',bufferfmt='ubyte')
        shelf.img1.texture = img_texture
    def preprocess(self,file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img,(100,100))
        img /=255.0
        return img
    def verify(self,*args):
        detection_thresold = 0.5
        verification_threshold = 0.5
        
        SAVE_PATH = os.path.join('verification_image','input_image','input_image.jpg')
        ret,frame = self.capture.read()
        frame = frame[120:120+250,200:200+250,:]
        cv2.imwrite(SAVE_PATH,frame)
        results = []
        for image in os.listdir(os.path.join('verification_image','valid_image')):
            input_image = self.preprocess(os.path.join('verification_image','input_image','input_image.jpg'))
            validation_image = self.preprocess(os.path.join('verification_image','valid_image',image))
            
            #make predictions
            result = self.model.predict(list(np.expand_dims([input_image,validation_image],axis=1)))
            results.append(result)
        detection = np.sum(np.array(results) > detection_thresold)
        print("detection ",detection)
        verification = detection/len(os.listdir(os.path.join('verification_image','valid_image')))
        print("verification ",verification)
        verified = verification > verification_threshold
        self.verification_label.text = 'verfied' if verification == True else 'unverified'
        Logger.info(results)
        Logger.info(np.sum(np.array(results)>0.2))
        Logger.info(np.sum(np.array(results)>0.4))
        Logger.info(np.sum(np.array(results)>0.8))
        Logger.info(np.sum(np.array(results)>0.9))
        
        return results,verified 
       
    
if __name__ == '__main__':
    CamApp().run()