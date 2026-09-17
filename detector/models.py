from django.db import models
from django.contrib.auth.models import User

class DetectionLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    file_name = models.CharField(max_length=255, default="scan_media.png")
    file_type = models.CharField(max_length=50, default="image")
    verdict = models.CharField(max_length=50, default="Real")
    confidence_score = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.file_name} - {self.verdict}"


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    profile_pic = models.ImageField(upload_to='profile_pics/', default='profile_pics/default.png', blank=True)

    def __str__(self):
        return f'{self.user.username} Profile'