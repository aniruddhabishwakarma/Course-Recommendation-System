import pandas as pd
from django.core.management.base import BaseCommand
from recommendations.models import Instructor, Course

class Command(BaseCommand):
    help = 'Import courses from CSV file and link to instructors'

    def handle(self, *args, **kwargs):
        self.stdout.write(self.style.SUCCESS('Reading CSV file...'))

        # Path to your CSV file
        csv_file = 'udemy_courses_data.csv'

        # Read the CSV using pandas
        df = pd.read_csv(csv_file)

        # ✅ Loop through each row and save course data
        for index, row in df.iterrows():
            title = row['title']
            url = row['url']
            thumbnail = row['thumbnail']
            instructor_name = row['instructor']  # Matching by name
            num_lectures = row['num_lectures']
            subscribers = row['subscribers']
            price = row['price']
            currency = row['currency']
            duration = row['duration']
            rating = row['rating']
            description = row['description']
            topic = row['topic']

            # ✅ Get or Create Instructor (ForeignKey Link)
            instructor, created = Instructor.objects.get_or_create(name=instructor_name)
            
            # ✅ Check for Existing Course
            if not Course.objects.filter(title=title, instructor=instructor).exists():
                Course.objects.create(
                    title=title,
                    url=url,
                    thumbnail=thumbnail,
                    instructor=instructor,  # ✅ Link using ForeignKey
                    num_lectures=num_lectures,
                    subscribers=subscribers,
                    price=price,
                    currency=currency,
                    duration=duration,
                    rating=rating,
                    description=description,
                    topic=topic
                )
                self.stdout.write(self.style.SUCCESS(f'Added Course: {title}'))

        self.stdout.write(self.style.SUCCESS('Course Import Complete!'))
