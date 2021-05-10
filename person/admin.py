from django.contrib import admin

from person.models import MissingPerson, MissingPersonPhotos


class Person(admin.ModelAdmin):
    list_display = ('id', 'name', 'birth_date', 'gender', 'state', 'city', 'reason_disappearance', 'with_special_needs')
    list_display_links = ('id', 'name')
    search_fields = ('name', 'state')
    list_per_page = 20


admin.site.register(MissingPerson, Person)


class PersonPhotos(admin.ModelAdmin):
    list_display = ('id', 'missing_person', 'insert_date')
    list_display_links = ('id',)
    list_per_page = 20


admin.site.register(MissingPersonPhotos, PersonPhotos)
