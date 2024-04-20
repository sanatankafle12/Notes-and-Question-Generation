from django.db import models

# Create your models here.

class TextEntry(models.Model):
    text = models.TextField()
    summary = models.TextField(blank=True)
    mcqs = models.TextField(blank=True)