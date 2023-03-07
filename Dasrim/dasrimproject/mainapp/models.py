from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Account(models.Model):
    objects = models.Manager()
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='users') # id
    name = models.CharField(max_length=10)
    nickname = models.CharField(max_length=30)
    email = models.CharField(max_length=48)
    gender = models.CharField(max_length=5)
    age = models.IntegerField()
    created_date = models.DateTimeField(auto_now = True)
    
    def __str__(self):
        return self.name

class MyDiagnosis(models.Model):
    objects = models.Manager()
    diagnosis = models.ForeignKey(User, on_delete=models.CASCADE, related_name='diagnosis')
    image = models.ImageField(blank=True, upload_to='images', null=True)
    summary = models.CharField(max_length=400, null=True)
    self_image = models.CharField(max_length=20, null=True)
    self_select = models.CharField(max_length=50, null=True)
    feel = models.IntegerField(null=True)
    created_date = models.DateTimeField(auto_now = True)
    # result
    score1 = models.IntegerField(null=True)
    score2 = models.IntegerField(null=True)
    score3 = models.IntegerField(null=True)
    total_score = models.FloatField(null=True)
    level = models.IntegerField(null=True)
    # images
    

    def __str__(self):
        return self.diagnosis.username

class Record(models.Model):
    objects = models.Manager()
    record_user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='record_user')
    count = models.IntegerField(null=True)  # 회차
    created_date = models.DateTimeField(auto_now = True)   # 자동으로 생성되는 시간
    situation = models.CharField(max_length=100, null=True)  # 상황
    feel = models.CharField(max_length=100, null=True)  # 감정
    automatic = models.CharField(max_length=100, null=True) # 자동적 사고
    image = models.ImageField(blank=True, upload_to='images', null=True)  # 이미지
    core_belief = models.CharField(max_length=100, null=True) # 핵심신념
    writer = models.CharField(max_length=100, null=True) # 작성자 - 처음 생성시 한번만 'elly'자동 저장
    cognitive_error = models.CharField(max_length=100, null=True) # 인지적오류

    def __str__(self):
        return self.record_user.username