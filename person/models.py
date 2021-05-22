import uuid

from PIL import Image, ImageOps
from django.db import models

from finder.settings import MEDIA_ROOT
from recognition.facial_recognition import Recognizer, FacialRecognition
import numpy as np

STATES = (
    ('AC', 'Acre'), ('AL', 'Alagoas'), ('AP', 'Amapá'),
    ('AM', 'Amazonas'), ('BA', 'Bahia'), ('CE', 'Ceará'),
    ('DF', 'Distrito Federal'), ('ES', 'Espírito Santo'),
    ('GO', 'Goiás'), ('MA', 'Maranhão'), ('MT', 'Mato Grosso'),
    ('MS', 'Mato Grosso do Sul'), ('MG', 'Minas Gerais'),
    ('PA', 'Pará'), ('PB', 'Paraíba'), ('PR', 'Paraná'),
    ('PE', 'Pernambuco'), ('PI', 'Piauí'), ('RJ', 'Rio de Janeiro'),
    ('RN', 'Rio Grande do Norte'), ('RS', 'Rio Grande do Sul'),
    ('RO', 'Rondônia'), ('RR', 'Roraima'), ('SC', 'Santa Catarina'),
    ('SP', 'São Paulo'), ('SE', 'Sergipe'), ('TO', 'Tocantins')
)


class MissingPerson(models.Model):
    GENDER = (
        ('M', 'MALE'),
        ('F', 'FEMALE'),
        ('O', 'OTHER')
    )
    name = models.CharField(max_length=200, blank=False, null=False)
    birth_date = models.DateField(null=False, blank=False)
    gender = models.CharField(max_length=1, choices=GENDER, blank=False, null=False)
    state = models.CharField(max_length=2, choices=STATES, blank=False, null=False)
    city = models.CharField(max_length=50, blank=False, null=False)
    reason_disappearance = models.CharField(max_length=500, blank=False, null=False)
    with_special_needs = models.BooleanField(null=True, blank=True, default=False)
    facial_recognition = models.BooleanField(null=True, blank=True)
    special_features = models.CharField(max_length=500, blank=False, null=False)
    date_of_disappearance = models.DateField(null=False, blank=False)
    alert_email = models.EmailField(null=False, blank=False)
    insert_date = models.DateTimeField(auto_now_add=True)


class MissingPersonPhotos(models.Model):
    missing_person = models.ForeignKey('person.MissingPerson', on_delete=models.CASCADE, related_name='photos')
    photo = models.ImageField('Foto', blank=False, null=False)
    insert_date = models.DateTimeField(auto_now_add=True)
    train = models.BooleanField()
    is_face_photo = models.BooleanField()

    def save(self, *args, **kwargs):
        if self.photo:
            recognizer = FacialRecognition()
            file = Image.open(self.photo)
            file = ImageOps.exif_transpose(file)
            #file = file.thumbnail((400, 400))
            file = recognizer.resize_image(file, 800, 600)
            file_ext = self.photo.file.content_type
            file_name = str(uuid.uuid4())
            file_ext = file_ext.split('/')[1]
            self.photo.name = file_name + '.' + file_ext
            recognizer.prepare_image(file).save(MEDIA_ROOT + '/train/' + file_name + '.' + file_ext, file_ext)
        super(MissingPersonPhotos, self).save(*args, **kwargs)
