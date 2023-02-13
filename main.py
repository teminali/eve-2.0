# libraries
import datetime
import os
import random
import shutil

import numpy as np
import pickle
import json
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import uvicorn

from classes.funcs import Funcs
# from classes.q2a import BertQuestionAnswering

lemmatizer = WordNetLemmatizer()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

time_zone = "Africa/Nairobi"
funcs = Funcs()

# Load the list of tokens from the text file
with open("services/config.json") as f:
    config = json.load(f)
    valid_tokens = config["tokens"]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chatbot_response")
async def chatbot_response(request: Request):
    data = await request.json()
    lang = data["lang"]
    uid = data["uid"]
    bot_name = data["bot_name"]
    user_message = data["user_message"]
    company_name = data["company_name"]
    company_email = data["company_email"]
    company_phone = data["company_phone"]
    company_address = data["company_address"]
    company_website = data["company_website"]
    company_description = data["company_description"]
    industry = data["industry"]

    # chat initialization with real data
    model = load_model(f"models/chatbot/{lang}/1/eve_model.h5")
    intents = json.loads(open(f"datasets/intents/customer_care.{lang}.json").read())
    words = pickle.load(open(f"models/chatbot/{lang}/1/words.pkl", "rb"))
    classes = pickle.load(open(f"models/chatbot/{lang}/1/classes.pkl", "rb"))
    # chat initialization with real data/

    try:
        token = request.headers.get("token")
        if token not in valid_tokens:
            print("Invalid token")
            return {"response": "Technical error, please contact the developer."}

        name = user_message[11:]
        ints = predict_class(user_message, model, words, classes)

        if user_message.startswith('my name is'):
            res1 = get_response(lang, ints, intents, user_message, token, uid, bot_name, company_name, company_email,
                                company_phone, company_address, company_website, company_description, industry)
            res = res1.replace("{n}", name)
        elif user_message.startswith('hi my name is'):
            res1 = get_response(lang, ints, intents, user_message, token, uid, bot_name, company_name, company_email,
                                company_phone, company_address, company_website, company_description, industry)
            res = res1.replace("{n}", name)
        else:
            res = get_response(lang, ints, intents, user_message, token, uid, bot_name, company_name, company_email,
                               company_phone, company_address, company_website, company_description, industry)

        return {"response": res}
    except Exception as e:
        print(e)
        return {"response": ""}


@app.post("/upload_knowledge_base")
async def upload_knowledge_base(request: Request):
    try:
        token = request.headers.get("token")
        if token not in valid_tokens:
            print("Invalid token")
            return "Technical error, please contact the developer."

        file = await request.form()["file"]

        if not file:
            return "No file found in the request."

        file_name = file.filename
        file_content = file.file.read()

        # Save the file to disk or process the content as needed
        with open(f"datasets/knowledge_base/{token}" + file_name, "wb") as f:
            f.write(file_content)

        return {"status": "success", "message": "File uploaded successfully"}
    except Exception as e:
        print(e)
        return "Sorry, my hands are tied right now. Please try again later."


# chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence=None, model=None, words=None, classes=None):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    error_threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(
        lang,
        ints,
        intents_json,
        question,
        token,
        uid,
        bot_name,
        company_name,
        company_email,
        company_phone,
        company_address,
        company_website,
        company_description,
        industry
):
    global result
    # q2a = BertQuestionAnswering(knowledge_base=f'knowledge_base/{token}', uid=uid)

    used_model = "margarita"

    if len(ints) < 1:
        tag = ""
    else:
        tag = ints[0]["intent"]

    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag or tag == "":
            if tag == "":
                result = ""
            else:
                result = random.choice(i["responses"])

            # replace [company_name] with company_name, [company_email] with company_email, etc.
            result = result.replace("[company_name]", company_name)
            result = result.replace("[company_email]", company_email)
            result = result.replace("[company_phone]", company_phone)
            result = result.replace("[company_address]", company_address)
            result = result.replace("[company_website]", company_website)
            result = result.replace("[company_description]", company_description)
            result = result.replace("[industry]", industry)
            result = result.replace("[bot_name]", bot_name)
            result = result.replace("[time_of_day]", funcs.get_time_of_day(time_zone))
            result = result.replace("[day]", funcs.get_today(time_zone))
            result = result.replace("[date]", funcs.get_date(time_zone))
            result = result.replace("[time]", funcs.get_time(time_zone))
            result = result.replace("[time_zone]", time_zone)

            if tag == "knowledge_base":
                used_model = "g2a"
                print("fetching answer from knowledge base...")
                error_msg = "Our servers are currently experiencing some delays. We apologize for the inconvenience " \
                            "and appreciate your patience."

                if lang == "sw":
                    error_msg = "Kwa sasa, seva zetu zina ucheleweshaji. Tunasikitika kwa ucheleweshaji huo na tunashukuru " \
                                "kwa subira yako. "
                try:
                    # result = q2a.answer_question(question)
                    result = error_msg
                    if result == "":
                        result = error_msg
                except Exception as e:
                    print(e)
                    result = error_msg

            if tag == "confused" or tag == "to_do_with_previous" or result == "":
                # let's start by extracting the last 20 messages but keep the line breaks between messages/lines
                with open(f"knowledge_base/{token}/chat_logs/{uid}.txt", "r") as f:
                    lines = f.readlines()
                    if len(lines) > 20:
                        lines = lines[-20:]
                    else:
                        lines = lines[-len(lines):]

                # remove the timestamp and labels human and bot
                for i in range(len(lines)):
                    lines[i] = lines[i].split(" - ")[1].strip()

                # if the last message is from the human, remove it
                if lines[-1].split(":")[0] == "human":
                    lines = lines[:-1]

                # reformat the conversation string
                conversation = ""
                for i in range(len(lines)):
                    if i % 2 == 0:
                        conversation += lines[i] + "\n"
                    else:
                        conversation += lines[i] + "\n"

                used_model = "explainer_model"
                result = funcs.create_message_from_getmessage(conversation, question)

            break
    if tag == "goodbye" or tag == "kwaheri":
        session = "ended"
    else:
        session = "ongoing"

    response = dict(bot_message=result, session=session, model=used_model)

    # create chat logs text file with the name uid and add the question and answer to the file with the timestamp and
    # labels human and bot, respectively. Otherwise, append the question and answer to the file with a line break.
    if os.path.exists(f"knowledge_base/{token}/chat_logs/{uid}.txt"):
        with open(f"knowledge_base/{token}/chat_logs/{uid}.txt", "a") as f:
            f.write(f"\n{datetime.datetime.now()} - human: {question}")
            f.write(f"\n{datetime.datetime.now()} - bot: {result}")
    else:
        # create chat logs directory if it doesn't exist
        if not os.path.exists(f"knowledge_base/{token}/chat_logs"):
            os.makedirs(f"knowledge_base/{token}/chat_logs")

        with open(f"knowledge_base/{token}/chat_logs/{uid}.txt", "w") as f:
            f.write(f"{datetime.datetime.now()} - human: {question}")
            f.write(f"\n{datetime.datetime.now()} - bot: {result}")

    # if session is ended, move the chat logs file to knowledge_base/chat_logs_archive
    if session == "ended":
        if not os.path.exists(f"knowledge_base/{token}/chat_logs_archive"):
            os.makedirs(f"knowledge_base/{token}/chat_logs_archive")

        shutil.move(f"knowledge_base/{token}/chat_logs/{uid}.txt",
                    f"knowledge_base/{token}/chat_logs_archive/{uid}.txt")

    return response


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=True)
