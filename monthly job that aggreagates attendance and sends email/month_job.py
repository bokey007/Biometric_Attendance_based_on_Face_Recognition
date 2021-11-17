import pandas as pd

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.mime.application import MIMEApplication

import numpy as np
import matplotlib.pyplot as plt

entry_file = 'C:/Users/bokey/attendace_system/face_recognition_for_diat/video-streamer-master/cv/attendance.xlsx'
exit_file = 'C:/Users/bokey/attendace_system/face_recognition_for_diat/video-streamer-master-exit/cv/attendance_exit.xlsx'


df_entry_file = pd.read_excel(entry_file, sheet_name='entry', header=None)
df_exit_file = pd.read_excel(exit_file, sheet_name='exit', header=None)

df_entry_file.columns = ['name', 'date', 'entry_time']
df_exit_file.columns = ['name', 'date', 'exit_time']



df_entry_min = df_entry_file.groupby(['name', 'date'], as_index=False).min()
df_exit_min = df_exit_file.groupby(['name', 'date'], as_index=False).max()

df_merged = pd.merge(df_entry_min, df_exit_min, on=['name', 'date'], how='outer')

df_final = df_merged.sort_values(by='name')

df_final.to_excel("monthly_attendance2.xlsx")

#sending mail

mail_content = '''Have a nice day 
Thank You
Team DRISHTI
'''

#The mail addresses and password
sender_address = 'bokey.production.team@gmail.com'
sender_pass = 'bokey.production@diat'
receiver_address = 'sudhir.admi.team@gmail.com'

#Setup the MIME
message = MIMEMultipart()
message['From'] = sender_address
message['To'] = receiver_address
message['Subject'] = 'Please find the attached excel sheet containing attendace of this month'

#The subject line
#The body and the attachments for the mail
message.attach(MIMEText(mail_content))
attach_file_name = 'monthly_attendance.xlsx'

xlsxpart = MIMEApplication(open('monthly_attendance.xlsx', 'rb').read())
xlsxpart.add_header('Content-Disposition', 'attachment', filename='1.xlsx')

message.attach(xlsxpart)
#Create SMTP session for sending the mail
session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
session.starttls() #enable security
session.login(sender_address, sender_pass) #login with mail_id and password
text = message.as_string()
session.sendmail(sender_address, receiver_address, text)
session.quit()
print('Mail Sent')
