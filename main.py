from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
from Face2bmi_ui import Ui_Face2bmi
import sys
import os
import dlib
import cv2
from PIL import Image, ImageQt
import numpy as np
import pandas as pd
from keras.models import load_model
from tensorflow import get_default_graph
import datetime


detector = dlib.get_frontal_face_detector()

def crop_faces(np_img):

    detected = detector(np_img, 1)
    
    d = detected[0]
    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
    xw1 = int(x1 - .1 * w)
    yw1 = int(y1 - .1 * h)
    xw2 = int(x2 + .1 * w)
    yw2 = int(y2 + .1 * h)
    cropped_img = crop_image(np_img, xw1, yw1, xw2, yw2)

    return cropped_img

def crop_image(img, x1, y1, x2, y2):
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2, :]

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                             -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_REPLICATE)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2


Label=[15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27.,
       28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40.,
       41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52., 53.,
       54., 55., 56., 57., 58., 59., 60., 61., 62., 63., 64., 65., 66.,
       67., 68., 69., 70.]
# , 73., 74.,  80., 81., 85.


classes=np.array(Label).astype('str').tolist()


model = load_model('my_model_56.h5')
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

global graph

graph = get_default_graph()

 
class Webcam(QThread):
    changePixmap = pyqtSignal(QImage)
    image = pyqtSignal(np.ndarray)
    def run(self):
        cap = cv2.VideoCapture(0)
        while(self.stop_flag==False):
            ret, frame = cap.read()
            
            if ret:
                y,x,_ = frame.shape
                frame = frame[:,int((x-y)/2+1):int(x-(x-y)/2+1)]
                frame = cv2.flip(frame,1)
                self.image.emit(frame)

                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                self.changePixmap.emit(convertToQtFormat)
        cap.release()


class Model(QThread):
    bmi = pyqtSignal(str)
    normalized_pic = pyqtSignal(QImage)
    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)

    def run(self):
        with graph.as_default():
            image = ImageQt.fromqpixmap(self.pic)
            image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)  
            img = crop_faces(image)
            rgbImage = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
            self.normalized_pic.emit(convertToQtFormat)
            
            img = Image.fromarray(rgbImage)
            img = img.resize((299,299))
            x = np.expand_dims(img, axis=0)
            x = x*1./255
            X = model.predict_classes(x)
            self.bmi.emit(str(Label[X[0]]))


class PredictAll(QThread):

    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)
        self.window=parent
        self.data = pd.DataFrame()
        
    def run(self):                   

        for name in self.window.pics:
            
            self.window.ui.graphicsView.setPixmap(QtGui.QPixmap(os.path.join(self.window.dirName, name)))
            self.window.ui.pic_name.setText('File Name : %s' %name)
            
            self.window.model.pic = self.window.ui.graphicsView.pixmap()
            self.window.model.start()
            self.window.model.bmi.connect(self.window.setBMI)
            self.window.model.normalized_pic.connect(self.window.setNormalizedPic)
            self.window.model.wait()
            
            r = pd.Series({'name':name,'bmi':int(float(self.window.bmi))})
            self.data = self.data.append(r,ignore_index=True)
            
        self.data.to_csv('./prediction.csv',index=False)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_Face2bmi()
        self.ui.setupUi(self)
        self.ui.graphicsView.setScaledContents(True)
        self.ui.normalized.setScaledContents(True)
        self.ui.folder.clicked.connect(self.openFileDialog)
        self.ui.prev.clicked.connect(self.prevPic)
        self.ui.next.clicked.connect(self.nextPic)
        self.ui.camera.clicked.connect(self.webcam_open)
        self.ui.stop.clicked.connect(self.webcam_close)
        self.ui.predict.clicked.connect(self.predict_bmi)
        self.ui.predict_all.clicked.connect(self.predict_all_images)
        self.ui.saveCam.clicked.connect(self.save_Webcam)
        self.webcam = Webcam(self)
        self.model = Model(self)
        self.predict_all = PredictAll(self)
        
    def setImage(self, image):
        
        self.ui.graphicsView.setPixmap(QtGui.QPixmap.fromImage(image))

             
    def nextPic(self):

        if len(self.pics) > 0:

            self.indx = (self.indx + 1) % len(self.pics)

            self.ui.graphicsView.setPixmap(QtGui.QPixmap(os.path.join(self.dirName, self.pics[self.indx])))

            self.ui.pic_name.setText('File Name : %s' %self.pics[self.indx])



    def prevPic(self):

        if len(self.pics) > 0:

            self.indx = (self.indx - 1) % len(self.pics)

            self.ui.graphicsView.setPixmap(QtGui.QPixmap(os.path.join(self.dirName, self.pics[self.indx])))
            
            self.ui.pic_name.setText('File Name : %s' %self.pics[self.indx])



    def openFileDialog(self):

        self.dirName = QtWidgets.QFileDialog.getExistingDirectory(self.ui.graphicsView, 'Choose Directory', os.getcwd())

        self.pics = [pic for pic in os.listdir(self.dirName) if pic.endswith(('jpg','JPG','png','PNG','bmp','jpeg'))]
        
        self.ui.dir_name.setText('Directory : %s' %self.dirName)
        
        
        if len(self.pics) > 0:

            self.indx = 0

            self.ui.graphicsView.setPixmap(QtGui.QPixmap(os.path.join(self.dirName, self.pics[self.indx])))

            self.ui.pic_name.setText('File Name : %s' %self.pics[self.indx])

    def predict_all_images(self):
        
        self.predict_all.start()

            
    def setBMI(self,bmi):
        
        self.bmi = bmi
        self.ui.bmi_Browser.setText(bmi)

    def setNormalizedPic(self,image):
        
        self.ui.normalized.setPixmap(QtGui.QPixmap.fromImage(image))
    
    def predict_bmi(self):
        self.ui.bmi_Browser.setText('Wait')
        self.model.pic = self.ui.graphicsView.pixmap()
        self.model.start()
        self.model.bmi.connect(self.setBMI)
        self.model.normalized_pic.connect(self.setNormalizedPic)


    def webcam_open(self):
        self.ui.dir_name.setText('Open webcam.')
        self.ui.pic_name.setText('')
        self.webcam.stop_flag = False
        self.webcam.start()
        self.webcam.changePixmap.connect(self.setImage)
        
    def webcam_close(self):
        self.webcam.stop_flag = True
        self.webcam.terminate()
        
    def save_Webcam(self):
        if not os.path.exists('saved'):
                os.makedirs('saved')
        ISOTIMEFORMAT = '%Y-%m-%d-%H-%M-%S'
        theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
        image = ImageQt.fromqpixmap(self.ui.graphicsView.pixmap())
        image.save('./saved/'+theTime+'.jpg')
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    return main

if __name__ == '__main__':
    main() 