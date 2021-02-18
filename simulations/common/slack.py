import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

notify_channel = 'C01FM29EHEF'  # ハードコードしているが，外部から変えられるようにするべき


def post_message(message: str):
    api_token = os.getenv('SLACK_BOT_TOKEN')
    if api_token is None:
        print('Cannot find slack api token.')
        return

    client = WebClient(token=api_token)
    response = client.chat_postMessage(channel=notify_channel, text=message)
    return response


def upload_file(filepath: str, initial_comment=None):
    api_token = os.getenv('SLACK_BOT_TOKEN')
    if api_token is None:
        print('Cannot find slack api token.')
        return

    client = WebClient(token=api_token)
    response = client.files_upload(channels=notify_channel, file=filepath, initial_comment=initial_comment)
    return response