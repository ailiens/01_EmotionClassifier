# Generated by Django 4.2.4 on 2023-08-10 04:14

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Member',
            fields=[
                ('userid', models.CharField(max_length=50, primary_key=True, serialize=False)),
                ('passwd', models.CharField(max_length=500)),
                ('name', models.CharField(max_length=20)),
                ('address', models.CharField(max_length=20)),
                ('tel', models.CharField(max_length=20, null=True)),
            ],
        ),
    ]
