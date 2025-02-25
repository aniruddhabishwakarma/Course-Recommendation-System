from django.contrib import admin
from .models import Instructor, Course, UserProfile, CourseView, SearchQuery

class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'full_name', 'contact_number')

# Register your models here.
admin.site.register(Instructor)
admin.site.register(Course)
admin.site.register(UserProfile, UserProfileAdmin)
admin.site.register(CourseView)
admin.site.register(SearchQuery)