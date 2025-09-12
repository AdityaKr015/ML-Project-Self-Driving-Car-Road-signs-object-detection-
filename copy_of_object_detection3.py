!pip install ultralytics

from  ultralytics import YOLO

# Commented out IPython magic to ensure Python compatibility.
# %pwd

!mkdir RoadsignDetection

path='/content/RoadsignDetection'

# Commented out IPython magic to ensure Python compatibility.
import os
os.chdir("/content/RoadsignDetection")
# %pwd

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="AALOD3IeCnQtf934sL40")
project = rf.workspace("selfdriving-car-qtywx").project("self-driving-cars-lfjou")
version = project.version(6)
dataset = version.download("yolov8")

!yolo train data=/content/RoadsignDetection/Self-Driving-Cars-6/data.yaml model=yolov8n.pt epochs=10  imgsz=640



# model = YOLO("yolov8n.pt")  # Official pre-trained model



!ls '/content/RoadsignDetection/runs/detect/train'

from IPython.display import Image,display
resultData=('/content/RoadsignDetection/runs/detect/train')
Image(filename=f'{resultData}/confusion_matrix.png')

resultData=('/content/RoadsignDetection/runs/detect/train')
Image(filename=f'{resultData}/results.png')

from google.colab import drive
drive.mount('/content/drive')
# !ls '/content/RoadsignDetection/Self-Driving-Cars-6'

import pandas as pd
resultData = '/content/RoadsignDetection/runs/detect/train/'
df = pd.read_csv(f'{resultData}results.csv')
print(df.head(5))

import pandas as pd
df=pd.read_csv(f'{resultData}results.csv')
print(df.tail(1))

# Commented out IPython magic to ensure Python compatibility.
import os
os.chdir("/content/")
# %pwd

!mkdir ManultestingData

os.chdir("/content/ManultestingData")

weights='/content/ManultestingData/runs/detect/train2/weights'
!yolo task=detect mode= predict model={weights}/best.pt conf=0.1 source ='/content/ManultestingData/new_vedio.mp4'

# !ls '/content/drive/My Drive'

from google.colab import drive
drive.mount('/content/RoadsignDetection/drive')

# Replace 'your_video_name.mp4' with the actual name of your video file in Google Drive
video_path = '/content/drive/My Drive/your_video_name.mp4'
weights_path = '/content/RoadsignDetection/runs/detect/train/weights/best.pt' # Use the correct path to your best weights

!yolo task=detect mode=predict model={weights_path} conf=0.1 source={video_path}