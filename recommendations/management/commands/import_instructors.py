import pandas as pd
from django.core.management.base import BaseCommand
from recommendations.models import Instructor

class Command(BaseCommand):
    help = 'Import instructors from CSV file'

    def handle(self, *args, **kwargs):
        self.stdout.write(self.style.SUCCESS('Reading CSV file...'))

        # Path to your CSV file
        csv_file = 'udemy_courses_data.csv'

        # Read the CSV using pandas
        df = pd.read_csv(csv_file)

        # ✅ Extract Unique Instructors
        unique_instructors = df[['instructor', 'instructor_photo']].drop_duplicates()

        # ✅ Save to Database
        for index, row in unique_instructors.iterrows():
            name = row['instructor']
            photo = row['instructor_photo']

            if not Instructor.objects.filter(name=name).exists():
                Instructor.objects.create(
                    name=name,
                    photo=photo
                )
                self.stdout.write(self.style.SUCCESS(f'Added: {name}'))

        self.stdout.write(self.style.SUCCESS('Import Complete!'))
