from django.db import models


class EmailConfig(models.Model):
    subject_name = models.CharField('Sujeito', blank=False, null=False, default='Finder', max_length=100)
    email = models.EmailField('Email', blank=False, null=False)
    password = models.CharField('Senha', null=False, blank=False, max_length=255)
    use_ssl = models.BooleanField('Usar SSL', default=True)
    use_tls = models.BooleanField('Usar TLS', default=True)
    server_address = models.CharField('Endereço do servidor', null=False, blank=False, default='smtp.gmail.com', max_length=255)
    server_port = models.CharField('Porta do servidor', default="587", null=False, blank=False, max_length=10)
    insert_date = models.DateTimeField(auto_now_add=True)

