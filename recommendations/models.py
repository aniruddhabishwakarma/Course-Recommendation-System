from django.db import models
from django.contrib.auth.models import User

class Instructor(models.Model):
    id = models.AutoField(primary_key=True)  # Using AutoField for unique IDs
    name = models.CharField(max_length=255, unique=True)  # Unique name
    job_title = models.CharField(max_length=255, blank=True)
    photo = models.URLField(blank=True)
    url = models.URLField(blank=True)
    
    def __str__(self):
        return self.name

class Course(models.Model):
    id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=255)
    url = models.URLField()
    thumbnail = models.URLField(blank=True)
    instructor = models.ForeignKey(Instructor, on_delete=models.CASCADE)  # ✅ ForeignKey to Instructor
    num_lectures = models.IntegerField(default=0)
    subscribers = models.IntegerField(default=0)
    price = models.CharField(max_length=50, default="Free")
    currency = models.CharField(max_length=50, default="Unknown")
    duration = models.CharField(max_length=100, blank=True)
    rating = models.FloatField(default=0)
    description = models.TextField(blank=True)
    topic = models.CharField(max_length=100, blank=True)
    
    def __str__(self):
        return self.title
    

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    full_name = models.CharField(max_length=100)
    contact_number = models.CharField(max_length=15)

def __str__(self):
    return self.user.username

class CourseView(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    viewed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} viewed Course ID: {self.course.id}"
    
class SearchQuery(models.Model):
    keyword = models.CharField(max_length=255)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    searched_at = models.DateTimeField(auto_now_add=True)  # Automatically set the timestamp

    def __str__(self):
        return f"{self.keyword}"