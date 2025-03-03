from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ..models import Course, SearchQuery, CourseView, Course
from django.db.models import Count, Q
import random

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

def get_trending_searches_with_courses(keyword_limit=2, course_limit=3):
    # ✅ Get the Top `keyword_limit` Trending Searches
    trending_searches = (
        SearchQuery.objects.values('keyword')
        .annotate(keyword_count=Count('keyword'))
        .order_by('-keyword_count')[:keyword_limit]
    )

    trending_data = []

    for trend in trending_searches:
        keyword = trend['keyword']
        
        # ✅ Fetch Courses Related to the Keyword
        related_courses = Course.objects.filter(
            Q(title__icontains=keyword) | Q(description__icontains=keyword)
        ).order_by('-rating')[:course_limit]  # ✅ Get Top `course_limit` Courses by Rating

        # ✅ Limit the Number of Courses to Display for Each Keyword
        limited_courses = related_courses[:course_limit]
        
        # ✅ Append Keyword and Its Limited Courses to the Data
        trending_data.append({
            'keyword': keyword,
            'courses': limited_courses
        })
        
        # ✅ Debugging Output
        print(f"Keyword: {keyword} -> Courses: {[course.title for course in limited_courses]}")

    return trending_data


def get_user_based_recommendations(user, num_recommendations=15):
    """
    Get course recommendations based on the user's most frequently searched keywords.
    """
    # ✅ Step 1: Get User’s Most Searched Keywords
    user_searches = (
        SearchQuery.objects.filter(user=user)
        .values('keyword')
        .annotate(search_count=Count('keyword'))
        .order_by('-search_count')
    )

    # ✅ Step 2: If No Search History, Return None
    if not user_searches.exists():
        return None  # ✅ No Recommendations if No Search History

    # ✅ Step 3: Select the Top 1-3 Most Searched Keywords
    top_keywords = [entry['keyword'] for entry in user_searches[:3]]

    # ✅ Step 4: Build a Query for All Top Keywords
    keyword_query = Q()
    for keyword in top_keywords:
        keyword_query |= Q(title__icontains=keyword) | Q(description__icontains=keyword)

    # ✅ Step 5: Get Courses Related to These Keywords
    all_recommended_courses = list(Course.objects.filter(keyword_query))  # Convert to List

    # ✅ Step 6: Shuffle the Courses Randomly
    random.shuffle(all_recommended_courses)

    # ✅ Step 7: Return Only `num_recommendations` Courses
    recommended_courses = all_recommended_courses[:num_recommendations]

    print(f"User's Most Searched Keywords: {top_keywords}")  # ✅ Debugging
    print(f"Recommended Courses (Randomized): {[course.title for course in recommended_courses]}")  # ✅ Debugging

    return recommended_courses
