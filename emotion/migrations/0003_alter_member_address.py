# Generated by Django 4.2.4 on 2023-08-14 05:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('emotion', '0002_predict'),
    ]

    operations = [
        migrations.AlterField(
            model_name='member',
            name='address',
            field=models.CharField(max_length=20, null=True),
        ),
    ]