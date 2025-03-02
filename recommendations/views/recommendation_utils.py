from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ..models import Course, SearchQuery, CourseView, Course
from django.db.models import Count, Q
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('stopwords')
nltk.download('punkt')


TECH_DOMAINS = {
    "Machine Learning", "Deep Learning", "Artificial Intelligence", "Computer Vision",
    "Data Science", "Big Data", "Cybersecurity", "Blockchain", "Cloud Computing",
    "Face Detection", "Natural Language Processing", "Python", "Java", "JavaScript",
    "SQL", "NoSQL", "DevOps", "React", "Angular", "Django", "Flask", "IELTS"
}


def extract_keyword(course_title):
    """
    Extracts the most domain-specific keyword(s) from the course title.
    Prioritizes known tech domains. If none found, returns the most frequent significant word.
    """
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(course_title)
    filtered_words = [word for word in words if word.isalnum() and word.lower() not in stop_words]

    # ✅ Check for Known Tech Domains in Course Title
    detected_keywords = [word for word in filtered_words if word in TECH_DOMAINS]
    
    if detected_keywords:
        return " ".join(detected_keywords[:2])  # ✅ Return first 2 matched keywords
    
    # ✅ If No Domain-Specific Keywords, Return the 2 Most Common Words
    most_common = [word for word, count in Counter(filtered_words).most_common(2)]
    
    return " ".join(most_common) if most_common else course_title.split()[0]  # Default to first word

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


def get_user_based_recommendations(user, num_recommendations=6):
    """
    Get user-based recommendations by finding the most viewed course by the user,
    and fetching similar courses using get_similar_courses.
    """
    # ✅ Step 1: Identify Courses Viewed by the User
    viewed_courses = (
        CourseView.objects
        .filter(user=user)
        .values('course_id')
        .annotate(view_count=Count('course_id'))
        .order_by('-view_count')
    )

    # ✅ Step 2: Find the Most Repeated Course ID
    if viewed_courses.exists():
        most_viewed_course_id = viewed_courses.first()['course_id']
        most_viewed_course = Course.objects.get(id=most_viewed_course_id)
        print(f"User's Most Viewed Course: {most_viewed_course.title} ({most_viewed_course_id})")  # ✅ Debugging

        # ✅ Step 3: Get Similar Courses
        recommended_courses = get_similar_courses(most_viewed_course, num_recommendations)
        print(f"Recommended Courses: {[course.title for course in recommended_courses]}")  # ✅ Debugging
    else:
        recommended_courses = None  # ✅ No recommendations if no history

    return recommended_courses

def extract_keyword(course_title):
    """Extracts the most relevant keyword from the course title."""
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(course_title)
    filtered_words = [word for word in words if word.isalnum() and word.lower() not in stop_words]
    
    if filtered_words:
        return filtered_words[0]  # ✅ Return the first keyword
    return course_title.split()[0]  # ✅ Default to first word if filtering fails
