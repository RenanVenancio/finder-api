from PIL import Image, ImageOps
from django.test import TestCase
import os
import numpy as np

from finder.settings import MEDIA_ROOT
from person.models import MissingPersonPhotos
from recognition import facial_recognition
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "finder.settings")
django.setup()


class FacialRecogitionTest(TestCase):
    def setUp(self):
        print("Config")

    def testar_imagens(self):
        dados = list(MissingPersonPhotos.objects.all().order_by('missing_person__id'))
        images, labels = [], []
        for i in dados:
            try:
                img = Image.open(fp=MEDIA_ROOT + '\\train\\' + i.photo.name).convert('L')
                images.append(np.asarray(img, 'uint8'))
                labels.append(i.missing_person.id)
            except:
                print("Falha")
            print('ok')
        print(labels)
        recognizer = facial_recognition.Recognizer()
        recognizer.train_lbph(images, np.array(labels))

    def testar_reconhecimento(self):
        recognizer = facial_recognition.Recognizer()
        img = Image.open(fp=MEDIA_ROOT + '\\foto.jpg').convert('L')
        img = recognizer.preparar_imagem(img)
        recognizer.read_classifier_rec()
        print(recognizer.reconhecer_lbph(np.asarray(img, 'uint8')))
