from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib import messages
from ..forms import CustomUserCreationForm
from ..models import Course
import random

# Create your views here.

def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(request, 'Account created successfully! You can now log in.')
            return redirect('login')
        else:
            messages.error(request, 'Registration failed. Please check the form for errors.')
    else:
        form = CustomUserCreationForm()
    return render(request, 'recommendations/login.html', {'form': form})

# ✅ User Login
def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f'Welcome back, {user.username}!')
            return redirect('home')  # Redirect to homepage or dashboard
    else:
        form = AuthenticationForm()
    return render(request, 'recommendations/login.html', {'form': form})

# ✅ User Logout
def logout_view(request):
    logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('login')


def home_view(request):
    # Get all course IDs
    course_ids = Course.objects.values_list('id', flat=True)
    # Select 15 random IDs
    random_ids = random.sample(list(course_ids), min(len(course_ids), 15))
    # Query the courses based on the random IDs
    courses = Course.objects.filter(id__in=random_ids)
    
    return render(request, 'recommendations/home.html', {'courses': courses})