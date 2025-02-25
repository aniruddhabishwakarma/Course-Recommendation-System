import requests
import time
import json
import os
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Fetches course and instructor data from Udemy API and stores it in JSON files'

    def handle(self, *args, **kwargs):
        self.stdout.write(self.style.SUCCESS('Fetching data from Udemy API...'))
        # Your fetching logic here

CLIENT_ID = "KgAcAKYRPDDJiQuDwLa8ONofIWM81fheeMRIva6U"
CLIENT_SECRET = "PBTIk3ESOKxBMXDU5aIplW3yGb2QpXzxbRq19icb8qhm6bo6nFF9QiMohKtSIchlfcqEIUIAxoYyjfh86h1Qo3WRnxbv07LSa9X3aZqrN0TjVBwOSJSAGYf9vzjUXugD"
UDEMY_API_URL = "https://www.udemy.com/api-2.0/courses/"

topics = [
    "Python", "Java", "JavaScript", "Deep Learning", "Machine Learning", "NLP", "Computer Vision", 
    "Data Science", "Data Analytics", "Mathematics", "Graphic Design", "IELTS"
]

# ✅ Function to Fetch Courses for Each Topic
def fetch_udemy_courses(topic, num_pages=4, page_size=100):
    all_courses = []
    instructors = {}

    for page in range(1, num_pages + 1):
        params = {
            "search": topic,
            "page": page,
            "page_size": page_size,
            "fields[course]": "id,title,url,image_480x270,visible_instructors,"
                              "num_subscribers,price,price_detail,content_info,"
                              "avg_rating,headline,num_published_lectures"
        }
        
        headers = {"Accept": "application/json"}
        response = requests.get(UDEMY_API_URL, params=params, headers=headers, auth=(CLIENT_ID, CLIENT_SECRET))

        if response.status_code == 200:
            courses = response.json().get("results", [])
            for course in courses:
                # ✅ Get first instructor
                instructor_data = course.get("visible_instructors", [{}])[0]
                instructor_id = instructor_data.get("id", "unknown")

                # ✅ Store instructor if not already stored
                if instructor_id not in instructors:
                    instructors[instructor_id] = {
                        "id": instructor_id,
                        "name": instructor_data.get("display_name", "Unknown"),
                        "job_title": instructor_data.get("job_title", ""),
                        "photo": instructor_data.get("image_100x100", ""),
                        "url": instructor_data.get("url", "")
                    }

                # ✅ Handle price safely
                price_detail = course.get("price_detail")
                price = price_detail["price_string"] if price_detail else "Free"
                currency = price_detail["currency"] if price_detail else "Unknown"

                course_data = {
                    "topic": topic,
                    "course_id": course.get("id", ""),
                    "title": course.get("title", ""),
                    "url": f"https://www.udemy.com{course.get('url', '')}",
                    "thumbnail": course.get("image_480x270", ""),
                    "instructor_id": instructor_id,  # ✅ Foreign Key
                    "num_lectures": course.get("num_published_lectures", 0),
                    "subscribers": course.get("num_subscribers", 0),
                    "price": price,
                    "currency": currency,
                    "duration": course.get("content_info", ""),
                    "rating": course.get("avg_rating", 0),
                    "description": course.get("headline", "")
                }
                all_courses.append(course_data)

            print(f"✅ Topic '{topic}' - Page {page}: Fetched {len(courses)} courses")

        else:
            print(f"❌ API Request Failed for '{topic}' on Page {page}: {response.status_code}, {response.text}")

        # ✅ Pause to prevent API rate limits
        time.sleep(1.5)

    return all_courses, instructors

# ✅ Fetch Courses and Store in JSON Files
all_courses_data = []
all_instructors_data = {}

for topic in topics:
    courses, instructors = fetch_udemy_courses(topic, num_pages=4, page_size=100)  
    all_courses_data.extend(courses)
    
    # ✅ Merge Instructors without Duplicates
    all_instructors_data.update(instructors)

# ✅ Convert to JSON and Save
def save_to_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"✅ Successfully saved to '{filename}'")

# ✅ Save Instructors and Courses Separately
save_to_json(list(all_instructors_data.values()), "instructors.json")
save_to_json(all_courses_data, "courses.json")
