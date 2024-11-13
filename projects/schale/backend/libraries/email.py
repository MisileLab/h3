from .initializer import initializer

def send_email(name: str, content: str, to: list[str]):
  print(f"send email to {to}")
  resp = initializer.smtp_client.send_email(
    Destination={"ToAddresses": to},
    Message={
      'Body': {'Html': {'Charset': 'UTF-8', 'Data': content + "<p>From theresa, open letter platform</p>"}},
      'Subject': {'Charset': 'UTF-8', 'Data': f'Sign the open letter, {name}'}
    },
    Source="theresa <theresa@rhodes.misile.xyz>"
  )
  print(resp)

