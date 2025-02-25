from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ..models import Course, SearchQuery, CourseView, Course
from django.db.models import Count, Q

def get_similar_courses(course):
    """
    Get a list of similar courses based on content (title, description, topic).
    Uses TF-IDF Vectorization and Cosine Similarity.
    """
    # ✅ Get all courses from the database
    all_courses = Course.objects.all()
    course_list = list(all_courses)

    # Extract relevant fields (title, description, topic) for vectorization
    course_contents = [f"{c.title} {c.description} {c.topic}" for c in course_list]

    # Vectorize the content using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(course_contents)

    # Calculate Cosine Similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get index of the current course
    course_index = course_list.index(course)

    # Get similarity scores and sort them
    similarity_scores = list(enumerate(cosine_sim[course_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the IDs of the most similar courses (excluding itself)
    similar_course_indices = [i[0] for i in similarity_scores if i[0] != course_index][:5]
    similar_courses = [course_list[i] for i in similar_course_indices]

    return similar_courses

def get_trending_searches_with_courses():
    """
    Get the top 5 trending searches with 5 courses related to each keyword.
    """
    # ✅ Get Top 5 Trending Searches
    trending_keywords = (
        SearchQuery.objects.values('keyword')
        .annotate(keyword_count=Count('keyword'))
        .order_by('-keyword_count')[:5]
    )

    trending_data = []

    for trend in trending_keywords:
        keyword = trend['keyword']

        # ✅ Get 5 Courses Related to the Keyword
        related_courses = Course.objects.filter(
            Q(title__icontains=keyword) | Q(description__icontains=keyword)
        ).order_by('-rating')[:5]

        trending_data.append({
            'keyword': keyword,
            'courses': related_courses
        })
    
    return trending_data

def get_user_based_recommendations(user, num_recommendations=5):
    """
    Get course recommendations based on other users' viewing patterns.
    """
    # ✅ Get all courses the current user has viewed
    viewed_courses = CourseView.objects.filter(user=user).values_list('course_id', flat=True)

    # ✅ Find users who viewed the same courses
    similar_users = CourseView.objects.filter(course_id__in=viewed_courses).exclude(user=user).values('user').distinct()

    # ✅ Get courses viewed by these similar users but not by the current user
    recommended_courses = (
        CourseView.objects.filter(user__in=similar_users)
        .exclude(course_id__in=viewed_courses)
        .values('course')
        .annotate(view_count=Count('course'))
        .order_by('-view_count')[:num_recommendations]
    )

    # ✅ Fetch the Course objects for the recommended course IDs
    recommended_course_ids = [item['course'] for item in recommended_courses]
    recommended_courses = Course.objects.filter(id__in=recommended_course_ids)

    return recommended_courses
