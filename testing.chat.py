import requests
import json

url = "http://localhost:3000/chatbot_response"

while True:
    msg = input("Human: ")
    headers = {
        "token": "sky-36a6ac9b-3bf4-444e-90bf-f68e11589391"
    }

    data = {
        "lang": "en",
        "uid": "29066fa8-abbb-4677-8d21-da5f04bbcd43",
        "bot_name": "Eve",
        "user_message": msg,
        "company_name": "HESLB",
        "company_email": "info@heslb.go.tz",
        "company_phone": "+255 22 286 4643",
        "company_address": "Dar es salaam, Tanzania",
        "company_website": "https://www.heslb.go.tz/",
        "company_description": "The Higher Education Students’ Loans Board (HESLB) is a body corporate established "
                               "under Act No.9 of 2004 (as amended in 2007, 2014 and 2016) with the objective of "
                               "assisting needy and eligible Tanzania students to access loans and grants for higher "
                               "education.",
        "industry": "Loan Board, Tanzania"
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        response_json = response.json()

        if "response" in response_json:
            print("AI: " + response_json["response"]["bot_message"])
        else:
            print("Chatbot: " + json.dumps(response_json))
    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)
    except ValueError as e:
        print(e)
        print("Invalid response from the server:", e)
