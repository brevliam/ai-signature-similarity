from django.db import models
from django.db.models.signals import pre_delete
from django.dispatch import receiver
import os

class Signature(models.Model):
    nik = models.CharField(max_length=16, default=None)
    img = models.ImageField(upload_to='images/', default=None)
    is_anchor = models.BooleanField(default=True)

    def __str__(self):
        return self.nik
    
@receiver(pre_delete, sender=Signature)
def delete_image_file(sender, instance, **kwargs):
    image_path = instance.img.path

    if os.path.isfile(image_path):
        os.remove(image_path)
