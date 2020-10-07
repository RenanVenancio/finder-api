from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    cpf = models.TextField(max_length=11, null=False)
    address = models.TextField(max_length=255, null=False)
    zipcode = models.TextField(max_length=8, null=False)
    uf = models.TextField(max_length=2, null=False)
    city = models.TextField(max_length=50, null=False)
    neighborhood = models.TextField(max_length=50, null=False)
    phone = models.TextField(max_length=20, null=False)