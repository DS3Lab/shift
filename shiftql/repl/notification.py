import os

import requests


def send_explain(raw_query, time_elapsed):
    raw_query = raw_query.replace("*", "all")
    bot_token = os.environ["BOT_TOKEN"]
    bot_chatID = os.environ["CHAT_ID"]
    host_name = (
        os.environ["BOT_HOST_NAME"]
        if os.environ["BOT_HOST_NAME"] is not None
        else "Default"
    )
    bot_message = (
        "[{}] Your query ***{}*** has been finished in ***{:.2f}*** seconds.".format(
            host_name, raw_query.replace("_", "-"), time_elapsed
        )
    )
    send_text = (
        "https://api.telegram.org/bot"
        + bot_token
        + "/sendMessage?chat_id="
        + bot_chatID
        + "&parse_mode=Markdown&text="
        + bot_message
    )
    response = requests.get(send_text)
    return response.json()


if __name__ == "__main__":
    send_explain()
