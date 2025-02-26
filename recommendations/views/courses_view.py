from django.shortcuts import render, redirect, get_object_or_404
from ..models import Course, Instructor, CourseView, SearchQuery
import random
from django.db.models import Q, Count
import nltk
from nltk.stem import WordNetLemmatizer
from django.core.paginator import Paginator
from .recommendation_utils import get_similar_courses, get_trending_searches_with_courses, get_user_based_recommendations

# ✅ Home View
def home_view(request):
    # Get all course IDs
    course_ids = Course.objects.values_list('id', flat=True)
    # Select 15 random IDs
    random_ids = random.sample(list(course_ids), min(len(course_ids), 15))
    # Query the courses based on the random IDs
    courses = Course.objects.filter(id__in=random_ids)

    # ✅ Get User-Based Recommendations if User is Logged In
    if request.user.is_authenticated:
        print(f"Fetching recommendations for {request.user.username}")  # ✅ Debugging
        recommended_courses = get_user_based_recommendations(request.user)
        print(f"Recommended Courses: {recommended_courses}")  # ✅ Debugging
    else:
        recommended_courses = None  # ✅ No recommendations for anonymous users


    # ✅ Get Trending Searches with Courses for Display
    trending_data = get_trending_searches_with_courses()
    print(f"Trending Data: {trending_data}")  # ✅ Debugging

    return render(request, 'recommendations/home.html', {
    'courses': courses,
    'trending_data': trending_data,  # ✅ Pass Trending Data to Template
    'recommended_courses': recommended_courses  # ✅ Pass Recommendations to Template
})


# ✅ Search View
def search_view(request):
    query = request.GET.get('q', '')  # Get the search query from the URL parameter
    lemmatizer = WordNetLemmatizer()
    search_results = []

    if query:
        # ✅ Store the Search Keyword ONLY if User is Logged In
        if request.user.is_authenticated:
            SearchQuery.objects.create(keyword=query, user=request.user)

        # Split the query into words and lemmatize each word
        keywords = query.split()
        lemmatized_keywords = [lemmatizer.lemmatize(word.lower()) for word in keywords]

        # Debugging - Print the lemmatized keywords
        print("Original Keywords:", keywords)
        print("Lemmatized Keywords:", lemmatized_keywords)
        
        # Build Q objects for searching in title and description
        q_objects = Q()
        for word in lemmatized_keywords:
            q_objects |= Q(title__icontains=word) | Q(description__icontains=word)

        # Search for courses matching any of the lemmatized keywords
        search_results = Course.objects.filter(q_objects).order_by('-rating').distinct()

        # ✅ Implement Pagination
        paginator = Paginator(search_results, 10)  # Show 10 results per page
        page_number = request.GET.get('page')  # Get the current page number
        page_obj = paginator.get_page(page_number)  # Get the current page of results

    else:
        page_obj = []

    return render(request, 'recommendations/search_results.html', {
        'query': query,
        'page_obj': page_obj
    })


# ✅ Course Detail View
def course_detail_view(request, course_id):
    # Get the course by ID or return 404 if not found
    course = get_object_or_404(Course, id=course_id)

    # Get the instructor details (ForeignKey relation)
    instructor = course.instructor

    # ✅ Track Course View for Logged-In Users
    if request.user.is_authenticated:
        # Check if this course has already been viewed by this user
        view_exists = CourseView.objects.filter(user=request.user, course=course).exists()
        if not view_exists:
            # If not, create a new CourseView entry
            CourseView.objects.create(user=request.user, course=course)

        # ✅ Get Similar Courses using the New Function
        similar_courses = get_similar_courses(course)
        print(f"Similar Courses: {similar_courses}")  # ✅ Debugging
    else:
        similar_courses = None  # ✅ None for Unauthenticated Users

    return render(request, 'recommendations/course_detail.html', {
        'course': course,
        'instructor': instructor,
        'similar_courses': similar_courses  # ✅ Pass Similar Courses to Template
    })


# ✅ Instructor's Courses View
def instructor_courses_view(request, instructor_id):
    # Get the instructor by ID or return 404 if not found
    instructor = get_object_or_404(Instructor, id=instructor_id)

    # Get all courses by this instructor
    courses = Course.objects.filter(instructor=instructor).order_by('-rating')

    # ✅ Implement Pagination for Instructor's Courses
    paginator = Paginator(courses, 9)  # Show 9 courses per page
    page_number = request.GET.get('page')  # Get the current page number
    page_obj = paginator.get_page(page_number)  # Get the current page of results

    return render(request, 'recommendations/instructor_courses.html', {
        'instructor': instructor,
        'page_obj': page_obj
     })
