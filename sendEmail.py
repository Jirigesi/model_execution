import smtplib, ssl

port = 465  # For SSL

smtp_server = "smtp.gmail.com"
sender_email = "runmodel.jiri@gmail.com"  # Enter your address
password = "JIrigesi3355"

receiver_email = "runmodel.jiri@gmail.com"  # Enter receiver address

message =  """\
Subject: Model completed Notification 

Model training is completed! ."""

# Create a secure SSL context
context = ssl.create_default_context()

with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
    server.login("runmodel.jiri@gmail.com", password)
    print("Login successfully...")
    server.sendmail(sender_email, receiver_email, message)
    print("Email sent successfully...")


