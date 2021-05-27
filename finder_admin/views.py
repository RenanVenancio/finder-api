from django.core.mail import send_mail
from django.core.mail.backends.smtp import EmailBackend
from finder_admin.models import EmailConfig


class Email:
    @staticmethod
    def send_email(message_html, recipient_list, subject):
        mail_setting = EmailConfig.objects.last()
        host = mail_setting.server_address
        host_user = mail_setting.email
        host_pass = mail_setting.password
        host_port = mail_setting.server_port
        try:
            connection = EmailBackend(
                host=host,
                port=host_port,
                password=host_pass,
                username=host_user,
                use_tls=True,
                timeout=10
            )
            send_mail(
                subject=subject,
                from_email=mail_setting.email,
                connection=connection,
                recipient_list=recipient_list,
                html_message=message_html,
                message=''
            )
            connection.close()
        except Exception as _error:
            print('Error in sending mail >> {}'.format(_error))
            return False
