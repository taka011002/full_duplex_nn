import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


def notify_slack(message: str):
    client = WebClient(token=os.environ['SLACK_BOT_TOKEN'])
    response = client.chat_postMessage(channel='C01FM29EHEF', text=message)
    return response


def upload_file(filepath: str,initial_comment=None):
    client = WebClient(token=os.environ['SLACK_BOT_TOKEN'])
    response = client.files_upload(channels='C01FM29EHEF', file=filepath, initial_comment=initial_comment)
    return response
