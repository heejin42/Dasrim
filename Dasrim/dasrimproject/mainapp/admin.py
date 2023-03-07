from django.contrib import admin
from .models import Account, MyDiagnosis, Record

# Register your models here.
admin.site.register(Account)
admin.site.register(MyDiagnosis)
admin.site.register(Record)