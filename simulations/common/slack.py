import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

notify_channel = 'C01FM29EHEF'  # ハードコードしているが，外部から変えられるようにするべき


def post_message(message: str):
    client = WebClient(token=os.environ['SLACK_BOT_TOKEN'])
    response = client.chat_postMessage(channel=notify_channel, text=message)
    return response


def upload_file(filepath: str, initial_comment=None):
    client = WebClient(token=os.environ['SLACK_BOT_TOKEN'])
    response = client.files_upload(channels=notify_channel, file=filepath, initial_comment=initial_comment)
    return response
