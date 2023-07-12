import tkinter as tk
from tkinter import filedialog
import cv2
from Generator import UNetResNet18
import config
import torch
import numpy as np
from torchvision import transforms
import uuid
from tkinter import *
from PIL import Image,ImageTk

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Colorizer AI v0.0.1")
        self.master.geometry("1920x1080")
        self.master.iconbitmap("gui_images/icon.ico")
        self.master.configure(bg='#151826')

        width = self.master.winfo_width()
        height = self.master.winfo_height()

        self.canvas_width = (width*20/100)
        self.canvas_img_width = (width - self.canvas_width) // 2
        self.canvas_img_height = 990
        self.canvas_height = 90

        self.canvas_img1 = Canvas(self.master,width=self.canvas_img_width,height=self.canvas_img_height,bg='#111419')
        self.canvas_img1.config(highlightbackground="#20252b", highlightcolor="#20252b")
        self.canvas_img1.place(width=self.canvas_img_width,height=self.canvas_img_height,anchor=W,relx=0.0,rely=0.5)

        self.canvas_img2 = Canvas(self.master,width=self.canvas_img_width,height=self.canvas_img_height,bg='#111419')
        self.canvas_img2.config(highlightbackground="#20252b", highlightcolor="#20252b")
        self.canvas_img2.place(width=self.canvas_img_width+(self.canvas_img_width/2),height=self.canvas_img_height,anchor=E,relx=1,rely=0.5)

        img1 = Image.open("gui_images/gray.jpg").resize((int(self.canvas_img_width), self.canvas_img_height))
        img2 = Image.open("gui_images/colored.jpg").resize((int(self.canvas_img_width), self.canvas_img_height))
        self.photo1 = ImageTk.PhotoImage(img1)
        self.photo2 = ImageTk.PhotoImage(img2)
        self.canvas_img1.create_image(0, 0, image=self.photo1, anchor=NW)
        self.canvas_img2.create_image(0, 0, image=self.photo2, anchor=NW)

        self.canvas = Canvas(self.master,width=self.canvas_width,height=height,bg='#111419')
        self.canvas.config(highlightbackground="#20252b", highlightcolor="#20252b")
        self.canvas.place(width=self.canvas_width,height=height,anchor=E,relx=1.0,rely=0.5)


        self.canvas1 = Canvas(self.master,width=width,height=self.canvas_height,bg='#111419')
        self.canvas1.config(highlightbackground="#20252b", highlightcolor="#20252b")
        self.canvas1.place(width=width-self.canvas_width,height=self.canvas_height)


        img = Image.open("gui_images/icon.png").resize((70,70)).convert("RGBA")
        background = Image.new("RGBA", img.size, "#111419")
        background.paste(img, (0,0), img)
        self.icon = ImageTk.PhotoImage(background)
        self.label = Label(self.master, image=self.icon, bd=0,bg='#111419')
        self.label.config(highlightbackground="#111419", highlightcolor="#111419")
        self.label.place(width=80,height=80)


        self.label1 = Label(self.master, text="Colorizer AI", font=("Arial", 20),fg='#e7e0db',bg='#111419')
        self.label1.config(highlightbackground="#111419", highlightcolor="#111419")
        self.label1.place(x=self.canvas_height,y=10)

        self.label1 = Label(self.master, text="v0.0.1", font=("Arial", 13),fg='#818487',bg='#111419')
        self.label1.config(highlightbackground="#111419", highlightcolor="#111419")
        self.label1.place(x=240,y=22)

        self.label3 = Label(self.master, text="Colorize your black and white photos with AI", font=("Arial", 15),fg='#e7e0db',bg='#111419')
        self.label3.config(highlightbackground="#111419", highlightcolor="#111419")
        self.label3.place(x=90,y=50)


        self.save_button = Button(self.master, width=30, height=2,command=self.save)
        self.save_button.config(highlightbackground="#111419", highlightcolor="#111419")
        self.save_button.config(text="Save Image", font=("Arial", 15), fg='#e7e0db', bg='#5d6bff')
        self.save_button.config(bd=0, highlightthickness=0)
        self.save_button.place(relx=0.81, rely=0.813+0.11)

        self.loadimg = Button(self.master, width=17, height=2,command=self.load_image)
        self.loadimg.config(highlightbackground="#20252b", highlightcolor="#20252b")
        self.loadimg.config(text="Load Image", font=("Arial", 11), fg='#e7e0db', bg='#20252b')
        self.loadimg.config(bd=0, highlightthickness=0)
        self.loadimg.place(relx=0.81, rely=0.68+0.11)

        self.loadvideo = Button(self.master, width=17, height=2,command=self.load_video)
        self.loadvideo.config(highlightbackground="#20252b", highlightcolor="#20252b")
        self.loadvideo.config(text="Load Video", font=("Arial", 11), fg='#e7e0db', bg='#20252b')
        self.loadvideo.config(bd=0, highlightthickness=0)
        self.loadvideo.place(relx=0.901, rely=0.68+0.11)

        self.colorize_button = Button(self.master, width=30, height=2,command=self.colorize)
        self.colorize_button.config(highlightbackground="#111419", highlightcolor="#111419")
        self.colorize_button.config(text="Colorize", font=("Arial", 15), fg='#e7e0db', bg='#3e47aa')
        self.colorize_button.config(bd=0, highlightthickness=0)
        self.colorize_button.place(relx=0.81, rely=0.744+0.11)

        self.model = UNetResNet18(1,2).to(config.DEVICE)
        self.model.load_state_dict(torch.load(config.CHECKPOINT_GEN))

        self.img_path = None
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.bgr = None
        self.saturation_level = 1


    def play_video(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (512, 512))
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, (512, 512))
                frame = frame.astype('float32')/255.0
                input_tensor = self.transform(frame)
                input_tensor = input_tensor.to(config.DEVICE)
                input_tensor = input_tensor.unsqueeze(0)
                gen_output = self.model(input_tensor)
                gen_output = gen_output.detach().cpu().squeeze().permute(1, 2, 0).numpy()
                gen_output = np.clip(gen_output,0,1)
                frame = cv2.resize(frame, (768, 990))
                frame = np.expand_dims(frame,axis=2)
                gen_output = cv2.resize(gen_output, (768, 990))

                lab_frame = np.concatenate([frame,gen_output],axis=2)
                lab_frame = np.clip(lab_frame,-1,1)
                self.bgr = cv2.cvtColor(np.uint8(lab_frame*255),cv2.COLOR_LAB2BGR)
                self.frame = cv2.cvtColor(np.uint8(frame*255),cv2.COLOR_GRAY2BGR)

                # Convert the image from RGB to HSV
                hsv = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2HSV)

                # Increase the saturation of the image
                hsv[:,:,1] = np.clip(hsv[:,:,1]*self.saturation_level, 0, 255)

                # Convert the image back to RGB
                self.bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                self.photo = tk.PhotoImage(data=cv2.imencode('.png', self.frame)[1].tobytes())
                self.canvas_img1.create_image(0, 0, image=self.photo, anchor=tk.NW)
                self.photo1 = tk.PhotoImage(data=cv2.imencode('.png', self.bgr)[1].tobytes())
                self.canvas_img2.create_image(0, 0, image=self.photo1, anchor=tk.NW)
                self.master.update()
                self.master.update_idletasks()
            else:
                self.master.update()
                self.master.update_idletasks()
                break

        self.master.update()
        self.master.update_idletasks()


    def load_video(self):
        video_path = filedialog.askopenfilename(initialdir=".", title="Select Video", filetypes=(("video files", "*.mp4"), ("video files", "*.avi")))
        self.cap = cv2.VideoCapture(video_path)
        self.play_video()

    def load_image(self):
        self.img_path = filedialog.askopenfilename(initialdir=".", title="Select Image", filetypes=(("jpeg files", "*.jpg"),("jpeg files", "*.jpeg"), ("png files", "*.png")))
        img = cv2.imread(self.img_path,0)
        img = cv2.resize(img, (768, 990))
        self.img = img
        self.photo = tk.PhotoImage(data=cv2.imencode('.png', img)[1].tobytes())
        self.canvas_img1.create_image(0, 0, image=self.photo, anchor=tk.NW)


    
    def colorize(self):
        img = cv2.imread(self.img_path,0)
        img = cv2.resize(img, (512, 512))
        img = img.astype('float32')/255.0
        input_tensor = self.transform(img)
        input_tensor = input_tensor.to(config.DEVICE)
        input_tensor = input_tensor.unsqueeze(0)
        gen_output = self.model(input_tensor)
        gen_output = gen_output.detach().cpu().squeeze().permute(1, 2, 0).numpy()
        gen_output = np.clip(gen_output,0,1)
        img = cv2.resize(img, (768, 990))
        img = np.expand_dims(img,axis=2)
        gen_output = cv2.resize(gen_output, (768, 990))
        lab_img = np.concatenate([img,gen_output],axis=2)
        lab_img = np.clip(lab_img,-1,1)
        self.bgr = cv2.cvtColor(np.uint8(lab_img*255),cv2.COLOR_LAB2BGR)
        # Convert the image from RGB to HSV
        hsv = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2HSV)

        # Increase the saturation of the image
        hsv[:,:,1] = np.clip(hsv[:,:,1]*self.saturation_level, 0, 255)

        # Convert the image back to RGB
        self.bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.photo1 = tk.PhotoImage(data=cv2.imencode('.png', self.bgr)[1].tobytes())
        self.canvas_img2.create_image(0, 0, image=self.photo1, anchor=tk.NW)

    def save(self):
        if self.bgr is not None:
            cv2.imwrite(f'saved/{uuid.uuid4()}.png',self.bgr)


root = tk.Tk()
app = App(root)
root.mainloop()
