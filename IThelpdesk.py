import flet as ft
import sys
import speech_recognition as sr
import time
import asyncio
import joblib
import csv
import pandas as pd
import numpy as np
import nltk
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import string
import re
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.corpus import stopwords
import numpy as np
from numpy.linalg import norm


def sent_tokens_func(text):
  return nltk.sent_tokenize(text)

def word_tokens_func(text):
  return nltk.word_tokenize(text)  

def to_lower(text):
  if not isinstance(text,str):
    text = str(text)
  return text.lower()

def number_omit_func(text):
  output = ''.join(c for c in text if not c.isdigit())
  return output

def remove_punctuation(text):
  return ''.join(c for c in text if c not in punctuation) 

def stopword_remove_func(sentence):
  stop_words = stopwords.words('english')
  return ' '.join([w for w in nltk.word_tokenize(sentence) if not w in stop_words])

def lemmatize(text):
          wordnet_lemmatizer = WordNetLemmatizer()
          lemmatized_word = [wordnet_lemmatizer.lemmatize(word)for word in nltk.word_tokenize(text)]
          return " ".join(lemmatized_word)

def preprocess(text):
        lower_text = to_lower(text)
        sentence_tokens = sent_tokens_func(lower_text)
        word_list = []
        for each_sent in sentence_tokens:
            lemmatizzed_sent = lemmatize(each_sent)
            clean_text = number_omit_func(lemmatizzed_sent)
            clean_text = remove_punctuation(clean_text)
            clean_text = stopword_remove_func(clean_text)
            word_tokens = word_tokens_func(clean_text)
            for i in word_tokens:
                word_list.append(i)
        return " ".join(word_list)


class ChatMessage(ft.Row):
    def __init__(self, message,user:"bot"):
        super().__init__()

        self.vertical_alignment="end"
        self.wrap=True
        if user=="bot":
            


            self.controls=[
                    # ft.CircleAvatar(
                    #     content=ft.Text("🤖"),
                    #     color=ft.colors.WHITE,
                    #     bgcolor=ft.colors.BLUE,
                    # ),
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Markdown("", selectable=True,extension_set="gitHubWeb",code_theme="gruvbox-dark",),
                            ],
            
                            tight=True,
                            spacing=5,
                        ),
                        bgcolor=ft.colors.BLUE_GREY_800,
                        border_radius=20,
                        padding=20,
                        animate_opacity=300,
                    ),
                ]
            self.alignment=ft.MainAxisAlignment.START
        else:
            self.controls=[
                    ft.Container(
                        content=ft.Column(
                            [
                                # ft.Text("User", weight="bold",color=ft.colors.BLACK),
                                ft.Text(message,color=ft.colors.BLACK,max_lines=20),
                            ],
                            tight=True,
                            spacing=5,
                        ),
                        bgcolor=ft.colors.BLUE_100,
                        border_radius=20,
                        padding=10
                    ),
                    # ft.CircleAvatar(
                    #     content=ft.Text("😊"),
                    #     color=ft.colors.WHITE,
                    #     bgcolor=ft.colors.GREEN,
                    # ),
                ]
            self.alignment=ft.MainAxisAlignment.END

def main(page: ft.Page):
    page.window_title_bar_hidden = True
    page.window_frameless = True
    page.window_width=400
    page.window_height=100
    page.window_left=50
    page.window_top=50
    # page.window_full_screen=True
    pb = ft.ProgressBar(width=400)
    x=ft.Column([ ft.Text("Loading Model..."), pb],alignment=ft.MainAxisAlignment.CENTER)
    page.add(x)
    global sentence_embeddings,model,preprocess,df
    df = pd.read_excel(r'df_withoutdup_final_origi1.xlsx')
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    for i in range(0, 101):
        pb.value = i * 0.01
        time.sleep(0.1)
        page.update()
    # page.window_title_bar_hidden=True
    page.remove(x)
    page.update()
    page.window_full_screen=True
    page.update()


    def kill(args):
        page.window_destroy()

    ##ALT+F4 disable [Does not work]
    # def on_keyboard(e: ft.KeyboardEvent):
    #     if e.alt==True and e.key=="F4":
    #         open_dlg_modal(e)
    # page.on_keyboard_event = on_keyboard

    def send_click(e):
        if new_message.value=="":
            dlg=ft.AlertDialog(title=ft.Text("Field Cannot Be Empty !"))
            page.dialog=dlg
            dlg.open=True
            page.update()
        else:
            global model,df
            sentence_embeddings =joblib.load('embeddings.joblib')
            intext=new_message.value
            intext=preprocess(intext)
            intext_embedding=np.transpose(np.array(model.encode([intext])))
            cosine=np.dot(sentence_embeddings,intext_embedding)/(norm(sentence_embeddings,axis=1)*norm(intext_embedding))
            index = np.where(cosine == np.amax(cosine))
            index=list(index[0])
            clear_button.disabled=True
            new_message.on_submit=None
            send_b.disabled=True
            chat.controls.append(ChatMessage(new_message.value,"user"))
            new_message.value = ""
            generated_response=df['Solution'][index[0]]
            #print(generated_response)
            chat.controls.append((ft.Container(
            content=ft.Row(
                            [ft.ProgressRing(width=16, height=16),
                                # ft.Text("User", weight="bold",color=ft.colors.BLACK),
                                ft.Text("Getting the response for you...",color=ft.colors.WHITE,max_lines=1),
                            ],                            
                            tight=True,
                            spacing=10,),
            padding=12)
            ))
            page.update()

            time.sleep(3)
            # asyncio.sleep(3)

            chat.controls.pop()
            page.update()

            chat.controls.append(ChatMessage("""empty string""","bot"))
            for i in range(len(generated_response)):
                chat.controls[-1].controls[0].content.controls[0].value += generated_response[i] + "_"
                page.update()
                chat.controls[-1].controls[0].content.controls[0].value = chat.controls[-1].controls[0].content.controls[0].value[:-1]
                time.sleep(0.005)
            send_b.disabled=False
            new_message.on_submit=send_click
            clear_button.disabled=False
            page.update()
            generated_response=""

    def kill(args):
        page.window_destroy()

    #CLEARCHAT
    def clear(e):
        chat.controls=[]
        page.update()

    def close_dlg(e):
        dlg_modal.open = False
        page.update()

    dlg_modal = ft.AlertDialog(
        modal=True,
        title=ft.Text("Quit!!"),
        content=ft.Text("Do you really want to quit?"),
        actions=[
            ft.TextButton("Yes", on_click=kill),
            ft.TextButton("No", on_click=close_dlg),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
        on_dismiss=lambda e: print("Modal dialog dismissed!"),
    )
    def open_dlg_modal(e):
        page.dialog = dlg_modal
        dlg_modal.open = True
        page.update()



    chat = ft.ListView(
        expand=True,
        spacing=10,
        auto_scroll=True,
    )
    
    # A new message entry form
    new_message = ft.TextField(
        hint_text="What would you like to know?",
        autofocus=True,
        shift_enter=True,
        min_lines=1,
        max_lines=100,
        # multiline=True,
        filled=True,
        expand=True,
        border_radius=20,
        on_submit=send_click,
        # label="Type Here..."
    )
    #CLEARBUTTON
    clear_button=ft.IconButton(ft.icons.DELETE_SWEEP_OUTLINED,
                        icon_color=ft.colors.WHITE,on_click=clear,
                        tooltip="Clear Chat") 

    def voice_to_text(event):
        mic_var.disabled=True #Deactiavate MIC
        event.control.selected = not event.control.selected
        event.control.update()
        recognizer = sr.Recognizer()

        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            # chat.controls.append(ChatMessage("SPEAK","bot"))
            page.update()
            audio = recognizer.listen(source)

        
        try:
            text = recognizer.recognize_google(audio)
            new_message.value += text #Appends to the txtbox itself
            new_message.focus()
            page.update()

        except sr.UnknownValueError:
            # new_message.value += "Sorry, I couldn't understand the audio."
            # new_message.focus()
            dlg1=ft.AlertDialog(title=ft.Text("Sorry, I couldn't understand the audio."),content=ft.Text("Try Speaking Clearly!!"))
            page.dialog=dlg1
            dlg1.open=True
            page.update()


        except sr.RequestError as e:
            # new_message.value = "Sorry, I couldn't understand the audio." + str(e)
            # new_message.focus()
            dlg2=ft.AlertDialog(title=ft.Text("Sorry, I couldn't understand the audio."))
            page.dialog=dlg2
            dlg2.open=True
            page.update()
            # c=0
        event.control.selected = not event.control.selected
        event.control.update()
        mic_var.disabled=False #Reactiavate MIC
        page.update()


    page.appbar = ft.AppBar(
        leading=ft.Icon(ft.icons.ASSISTANT,color=ft.colors.PURPLE_200,size=35),
        leading_width=40,
        title=ft.Text("I.T. Help Desk Assistant"),
        center_title=False,
        bgcolor=ft.colors.BLACK12,
        actions=[
            # ft.IconButton(ft.icons.THUMB_UP_OFF_ALT_ROUNDED,icon_color=ft.colors.GREEN,tooltip="Like"),
            # ft.IconButton(ft.icons.THUMB_DOWN_ALT_ROUNDED,icon_color=ft.colors.RED,tooltip="Dislike"),
            clear_button,
            ft.IconButton(ft.icons.POWER_SETTINGS_NEW,icon_color=ft.colors.BLUE_50,on_click=open_dlg_modal,tooltip="Close"),
        ],
    )


    mic_var=ft.IconButton(
                    icon=ft.icons.MIC_OFF,
                    tooltip="Voice message",
                    on_click=voice_to_text,
                    selected_icon=ft.icons.MIC,
                    data=0
                    # disabled=True,
                    )

    send_b=ft.IconButton(
                    icon=ft.icons.SEND_ROUNDED,
                    tooltip="Send message",
                    on_click=send_click,
                    icon_color=ft.colors.DEEP_PURPLE_200
                )

    # Add everything to the page
    page.add(
        ft.Container(
            content=chat,
            border=ft.border.all(1, ft.colors.OUTLINE),
            border_radius=15,
            padding=10,
            expand=True,
        ),
        ft.Row(
            [
                new_message,
                # ft.IconButton(
                #     icon=ft.icons.MIC_OFF,
                #     tooltip="Voice message",
                #     on_click=voice_to_text,
                #     selected_icon=ft.icons.MIC,
                #     # disabled=True,
                    
                #     ),
                mic_var,
                send_b,
            ]
        ),
    )



ft.app(main)