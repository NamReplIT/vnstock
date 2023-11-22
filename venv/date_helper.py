from datetime import datetime, timedelta
from dateutil import parser
from random import randint

class DateHelper:
    
    LONG_DATE_FORMAT = "%A, %d %B %Y"
    SHORT_DATE_FORMAT = "%d/%m/%Y"
    TIME_12_HOUR_FORMAT = "%I:%M %p"
    TIME_24_HOUR_FORMAT = "%H:%M:%S"
    ISO_8601_DATE_TIME = "%Y-%m-%dT%H:%M:%S"
    ISO_8601_DATE = "%Y-%m-%d"
    FRIENDLY_DATE_TIME = "%b %d, %Y at %I:%M %p"
    FULL_NUMERIC_DATE_TIME = "%Y-%m-%d %H:%M:%S"
    YEAR_MONTH = "%Y-%m"
    MONTH_DAY_YEAR = "%m/%d/%Y"
    DAY_MONTH_YEAR = "%d/%m/%Y"

    @staticmethod
    def parse_date(date):
        if isinstance(date, str):
            return parser.parse(date)
        elif isinstance(date, datetime):
            return date
        else:
            raise ValueError("The date must be a string or a datetime object.")

    @staticmethod
    def addDays(date, days, exclude=[]):
        date = DateHelper.parse_date(date)
        increment = 1 if days > 0 else -1
        while days != 0:
            date += timedelta(days=increment)
            if DateHelper._should_count_day(date, exclude):
                days -= increment
        return date

    @staticmethod
    def subtractDays(date, days, exclude=[]):
        return DateHelper.addDays(date, -days, exclude)

    @staticmethod
    def compareDates(date1, date2):
        date1 = DateHelper.parse_date(date1)
        date2 = DateHelper.parse_date(date2)
        return (date1 - date2).total_seconds()

    @staticmethod
    def daysBetween(startDate, endDate, exclude=[]):
        start = DateHelper.parse_date(startDate)
        end = DateHelper.parse_date(endDate)
        daysCount = 0
        while start <= end:
            if DateHelper._should_count_day(start, exclude):
                daysCount += 1
            start += timedelta(days=1)
        return daysCount

    @staticmethod
    def format(date, pattern):
        date = DateHelper.parse_date(date)
        return date.strftime(pattern)

    @staticmethod
    def isLeapYear(year):
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    @staticmethod
    def getDaysInMonth(month, year):
        return (datetime(year, month + 1, 1) - datetime(year, month, 1)).days

    @staticmethod
    def toFullDateDetail(date):
        date = DateHelper.parse_date(date)
        return date.strftime('%A, %d %B %Y %H:%M:%S')

    @staticmethod
    def now():
        return datetime.now()

    @staticmethod
    def randomDateTime(date, time, secondInBetween=[]):
        date = DateHelper.parse_date(date).date()
        time = DateHelper.parse_date(time).time()
        seconds = randint(secondInBetween[0], secondInBetween[1])
        return datetime.combine(date, time) + timedelta(seconds=seconds)

    @staticmethod
    def addSeconds(date, seconds):
        date = DateHelper.parse_date(date)
        return date + timedelta(seconds=seconds)

    @staticmethod
    def _should_count_day(date, exclude):
        day_name = date.strftime('%A')
        day_str = date.strftime('%Y-%m-%d')
        if 'weekend' in exclude and day_name in ['Saturday', 'Sunday']:
            return False
        if day_name in exclude or day_str in exclude:
            return False
        return True
