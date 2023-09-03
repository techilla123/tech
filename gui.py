import customtkinter
import cv2
from PIL import Image, ImageTk
from tkinter_webcam import webcam

customtkinter.set_appearance_mode("dark")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("dark-blue")  # Themes: blue (default), dark-blue, green

app = customtkinter.CTk()  # create CTk window like you do with the Tk window
app.geometry("1920x1080")
cap = cv2.VideoCapture(0)
def button_function():
    print("button pressed")

def show_video():
   ret, frame = cap.read()
   
   if ret:
       # cv2 uses `BGR` but `GUI` needs `RGB`
       frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

       # convert to PIL image
       img = Image.fromarray(frame)

       # convert to Tkinter image
       photo = ImageTk.PhotoImage(image=img)
       
       # solution for bug in `PhotoImage`
       label.photo = photo
       
       # replace image in label
       label.configure(image=photo)  
   
   # run again after 20ms (0.02s)
   app.after(20, show_video)

def enable_video():
    browse.forget()
    upload.forget()
    snap.pack()
    show_video()


def take_snap():
    print("snap taken")


# Use CTkButton instead of tkinter Button
label = customtkinter.CTkLabel(app)
browse = customtkinter.CTkButton(master=app, text="Browse", command=button_function)
upload = customtkinter.CTkButton(master=app, text="Capture", command=enable_video)
snap= customtkinter.CTkButton(master=app, text="snap", command=take_snap)
snap.forget()
browse.pack(padx=10, pady=10,anchor=customtkinter.SW)
upload.pack(padx=10, pady=10,anchor=customtkinter.SW)
label.pack(fill='both', expand=True)





app.mainloop()

cap.release()