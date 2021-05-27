from apscheduler.schedulers.background import BackgroundScheduler

from finder.settings import MEDIA_ROOT
from person.models import MissingPersonPhotos
from PIL import Image, ImageOps
import numpy as np
import cv2 as cv

from recognition import facial_recognition


def train_job():
    data = list(MissingPersonPhotos.objects.all().order_by('missing_person__id'))
    if len(data) > 0:
        images, labels = [], []
        for i in data:
            try:
                #img = Image.open(fp=MEDIA_ROOT + '\\train\\' + i.photo.name)
                img = cv.imread(MEDIA_ROOT + '\\train\\' + i.photo.name)
                #cv.imshow("ola",img)
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                images.append(np.asarray(img, 'uint8'))
                labels.append(i.missing_person.id)
            except:
                pass
        print(np.array(labels))
        recognizer = facial_recognition.FacialRecognition()
        recognizer.train_lbph(images, np.array(labels))