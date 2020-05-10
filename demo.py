from Model.Detector import Detector
from PIL import Image
from Model.Adaboost import Adaboost

if __name__ == "__main__":
    img = Image.open('demo.jpg').resize((128, 128))
    detector = Detector("Parameters/Model/")
    print(detector(img))
