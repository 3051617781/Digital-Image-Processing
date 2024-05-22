from tkinter import *
from tkinter import filedialog
from tkinter import simpledialog
from PIL import Image, ImageTk
from main import  *

def popup_input_box():
    # 弹出输入框并获取输入值
    user_input = simpledialog.askstring("参数设置", "参数设置格式：\n"
                                                "噪声：\n"
                                                "--椒盐:prob\n"
                                                "--高斯:mean var\n"
                                                "--均匀:mean sigma\n\n"
                                                "滤波模型\n"
                                                "--中值:size\n"
                                                "--算数平均:(t1, t2)\n"
                                                "--反调和:size\n"
                                                "--最大/最小:size\n"
                                                "请输入参数:")
    if user_input:
        return user_input
    else:
        return
    
def button1():#上传图片（不用动）
    global lab1, path
    path= filedialog.askopenfilename()

    if len(path) > 0:
        img = cv.imread(path)
        
        output(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.thumbnail((150,100))
        photo = ImageTk.PhotoImage(img)
        if lab1 == None:
            lab1 = Label(window, image=photo)
            lab1.image = photo
            lab1.grid(row=1,column=0,rowspan=2,sticky=NSEW)
            
        else:
            lab1.configure(image=photo)
            lab1.image = photo
            
def output(img):  #显示图片
    global lab2
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img.thumbnail((600,400))
    photo = ImageTk.PhotoImage(img)
    if lab2 == None:
        lab2 = Label(window, image=photo)
        lab2.image = photo
        lab2.grid(row=3,column=1,rowspan=8,sticky=NSEW)         
    else:
        lab2.configure(image=photo)
        lab2.image = photo
    
          
def init():#读取图片
    global path
    img=cv.imread(path)
    return img
              
def buttonsave():

    None #保存图片
       

def button2():#椒盐噪声
    # 设置参数
    para = popup_input_box()
    if para:
        prob = float(para.strip())
        output(sp_noise(init(), prob))
    output(sp_noise(init()))

def button3():   #高斯噪声
    para = popup_input_box()
    if para:
        mean = float(para.strip().split()[0])
        var = float(para.strip().split()[1])
        output(gaussian_noise(init(), mean, var))
    output(gaussian_noise(init()))


def button4():   #均匀噪声
    para = popup_input_box()
    if para:
        mean = float(para.strip().split()[0])
        sigma = float(para.strip().split()[1])
        output(un_noise(init(), mean, sigma))
    output(un_noise(init()))

def button5():   #中值滤波
    para = popup_input_box()
    if para:
        ksize = float(para.strip())
        output(median_blur(init(), ksize))
    output(median_blur(init()))


def button6():   #算术平均滤波
    para = popup_input_box()
    if para:
        tup = tuple(para.strip().split())
        output(avg_blur(init(), tup))
    output(avg_blur(init()))

def button7():   #反调和滤波
    para = popup_input_box()
    if para:
        size = float(para.strip())
        output(harmonic_mean_filter(init(), size))
    output(harmonic_mean_filter(init()))

def button8():   #最大滤波
    para = popup_input_box()
    if para:
        size = float(para.strip())
        output(max_filter(init(), size))
    output(max_filter(init()))

def button9():   #最小滤波
    para = popup_input_box()
    if para:
        size = float(para.strip())
        output(min_filter(init(), size))
    output(min_filter(init()))

window = Tk()
window.title('图像去噪实验')
window.geometry('950x600')
global lab1
global lab2

global path
background1=Image.new('RGB', (150, 100), color='gray')
photo1=ImageTk.PhotoImage(background1)
lab1 = Label(window, image=photo1)
lab1.grid(row=1, column=0, rowspan=2, sticky=NSEW)
background2=Image.new('RGB', (600, 400), color='gray')
photo2=ImageTk.PhotoImage(background2)
lab2 = Label(window, image=photo2)
lab2.grid(row=3, column=1, columnspan=8, rowspan=8, sticky=NSEW) 

b1 = Button(window, text='上传图片', command=button1, width=20).grid(row=0, column=0) 
bsave=Button(window,text='保存图片',command=buttonsave,width=20).grid(row=0,column=1)
b2 = Button(window, text='椒盐噪声', command=button2, width=20).grid(row=3, column=9) 
b3 = Button(window, text='高斯噪声', command=button3, width=20).grid(row=4, column=9) 
b4 = Button(window, text='均匀噪声', command=button4, width=20).grid(row=5, column=9) 
b5 = Button(window, text='中值滤波', command=button5, width=20).grid(row=6, column=9) 
b6 = Button(window, text='算术平均滤波', command=button6, width=20).grid(row=7, column=9) 
b7 = Button(window, text='反调和滤波', command=button7, width=20).grid(row=8, column=9) 
b8 = Button(window, text='最大值滤波', command=button8, width=20).grid(row=9, column=9)
b9 = Button(window, text='最小值滤波', command=button9, width=20).grid(row=10, column=9)

window.mainloop()

