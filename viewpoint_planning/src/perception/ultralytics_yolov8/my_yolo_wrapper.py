from ultralytics import YOLO
from PIL import Image


# This file is VOID now

class MyYoloWrapper:
    """
    Gets data from the camera and performs semantic segmentation and pose estimation.
    """

    def __init__(self):
        # TODO: change this camera to D435i
        self.model = YOLO("weights/myvpp_sim_best.pt")
        # self.source = "images/integration/test_p1.png"
        

    def getResult(self, input_img ):
        results = self.model(input_img)
        return results
    
    def getModel(self):
        return self.model

