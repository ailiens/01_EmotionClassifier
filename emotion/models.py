from django.db import models

# Create your models here.
class Member(models.Model):
    userid = models.CharField(max_length=50, null=False, blank=False, primary_key=True)
    passwd = models.CharField(max_length=500, null=False, blank=False)
    name = models.CharField(max_length=20, null=False)
    address = models.CharField(max_length=20, null=True)
    tel = models.CharField(max_length=20, null=True)


class Predict(models.Model):
    idx = models.AutoField(primary_key=True)
    userid = models.CharField(max_length=50, null=False)
    query = models.CharField(max_length=500, null=False)
    predict = models.CharField(max_length=50, null=False)