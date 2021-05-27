import django
from django.test import TestCase
import os

from finder_admin.views import Email


class EmailSendTest(TestCase):
    def setUp(self):
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "finder.settings")
        django.setup()
        print("Config")

    def test_mail_send(self):
        Email.send_email(message_html='<h1>Hi, you like this?</h1><p>Its a only test for mail sending</p>',
                         recipient_list=['mailtest@hotmail.com'], subject='alert')
