from django.urls import path
from .views.courses_view import *
from .views.user_view import *

urlpatterns = [
    path('', home_view, name='home'),
    path('search/', search_view, name='search'),
    path('course/<int:course_id>/', course_detail_view, name='course_detail'),
    path('instructor/<int:instructor_id>/courses/', instructor_courses_view, name='instructor_courses'),
    path('register/', register, name='register'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
]
