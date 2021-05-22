
from django.apps import AppConfig


class FinderConfig(AppConfig):
    name = 'finder'

    def ready(self):
        from recognition.job import train_job
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler()
        scheduler.add_job(train_job, 'interval', minutes=1440)
        scheduler.start()
