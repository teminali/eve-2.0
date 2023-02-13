import datetime
import pytz

from classes.explainer import Explainer

explainer = Explainer()


class Funcs:
    @staticmethod
    def get_time_of_day(time_zone):
        tz = pytz.timezone(time_zone)
        now = datetime.datetime.now(tz)
        hour = now.hour

        if hour < 12:
            return "Good morning"
        elif 12 <= hour < 17:
            return "Good afternoon"
        else:
            return "Good evening"

    @staticmethod
    def get_today(time_zone):
        tz = pytz.timezone(time_zone)
        now = datetime.datetime.now(tz)
        return now.strftime("%A")

    @staticmethod
    def get_time(time_zone):
        tz = pytz.timezone(time_zone)
        now = datetime.datetime.now(tz)
        return now.strftime("%H:%M:%S")

    @staticmethod
    def get_date(time_zone):
        tz = pytz.timezone(time_zone)
        now = datetime.datetime.now(tz)
        return now.strftime("%d/%m/%Y")

    # creating a message from the Explainer class
    @staticmethod
    def create_message_from_getmessage(conversation, question):
        prompt = f"The following is a conversation with a customer care live support AI Agent. The assistant is " \
                 f"helpful, creative, clever, cheerful, a little funny and very " \
                 f"friendly.\n{conversation}human: {question}\nbot:"

        print(prompt)
        result = explainer.generate_response(prompt=prompt, question=question)

        # remove line break from the result
        result = result.replace("\n", " ")

        return result

        # creating a message from the GetMessage class /

# time_zone = "Africa/Nairobi"
# f = Funcs()
# print(f.get_time_of_day(time_zone))
