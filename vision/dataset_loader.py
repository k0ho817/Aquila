from roboflow import Roboflow
import os
from dotenv import load_dotenv

# apikey load
load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=api_key)

'''
dataset index
0 : Custom dataset
1 : yolov11-70iwr
'''

# custom dataset
def cust_data():
    project = rf.workspace("korea-national-university-of-transportation-ifxep").project("aquila-sy18i")
    version = project.version(1)
    dataset = version.download("yolov8", location="../data")

# roboflow dataset1
def robo_data1():
    project = rf.workspace("learn-j9wwx").project("yolov11-70iwr")
    version = project.version(1)
    dataset = version.download("yolov8", location="../data")

def load_dataset(dataset_num):
    if dataset_num == 0:
        cust_data()
    elif dataset_num == 1:
        robo_data1()
    else:
        print("Not dataset index")
        return
    
load_dataset(0)