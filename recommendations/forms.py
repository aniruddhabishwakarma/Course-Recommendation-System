from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from .models import UserProfile

class CustomUserCreationForm(UserCreationForm):
    full_name = forms.CharField(max_length=100, required=True, label='Full Name')
    contact_number = forms.CharField(max_length=15, required=True, label='Contact Number')

    class Meta:
        model = User
        fields = ['username', 'full_name', 'contact_number', 'email', 'password1', 'password2']

    def save(self, commit=True):
        user = super().save(commit=False)
        if commit:
            user.save()
            # Save additional fields in UserProfile
            UserProfile.objects.create(
                user=user,
                full_name=self.cleaned_data['full_name'],
                contact_number=self.cleaned_data['contact_number']
            )
        return user
