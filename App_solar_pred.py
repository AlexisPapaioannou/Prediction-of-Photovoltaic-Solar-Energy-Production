
#IMPORTS################################################################################################################

from tkinter import *
from tkinter import ttk
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import requests
from keras.models import model_from_json
#import json
from datetime import datetime,timezone
#import image
from PIL import Image,ImageTk

root = Tk()

#FUNCTIONS##############################################################################################################

def window(root):
    root.title("Thesis")
    root.geometry("1366x768")


def raise_frame(frame):
    frame.tkraise()


def First_Page_View():
    global example

    myLabel = Label(PageOne,text="Hello!"+ Entry1.get())
    myLabel.pack(side = TOP)


def Algorithms(ex,dataset):
    for i in range(len(dataset)):
        if (dataset.Month[i] == int(Entry_month.get())) and (dataset.Day[i] == int(Entry_day.get())):
            #print("stop",i)
            shmeio = i      # Arxh ths hmeras (00:00)
            break

    dataset_cut= dataset[shmeio : shmeio + 24]
    X = np.array(dataset_cut[['Temperature', 'Sunshine_Duration', 'Relative_Humidity', 'Wind_Speed_80m','Total_Cloud_Cover', 'Calc_radiation']])
    Y = np.array(dataset_cut[['Shortwave_Radiation']])
    Kwh = np.array(dataset_cut[['Kwh']])
    Scaler_X = preprocessing.StandardScaler().fit(X)
    Scaler_Y = preprocessing.StandardScaler().fit(Y)
    X_scaled = Scaler_X.transform(X)
    Y_scaled = Scaler_Y.transform(Y)

    count = 0
    sum_cloud =0
    for j in range(len(X)):
        if (count >= 8 and count <= 20):
            sum_cloud = sum_cloud + X[j][4]
        count = count + 1

    if(sum_cloud>1100):                     #TOTAL CLOUD
        if(ex==1):
            filename = 'models\\SVR\\model_svr_total_cloud.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
        if (ex == 2):
            filename = 'models\\Boosting\\model_boosting_total_cloud.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
        if(ex==3):
            json_file = open('models\\NN\\model_NN_total_cloud.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("models\\NN\\model_total_cloud.h5")
            loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        print(sum_cloud)
        print("Total cloud")
    elif (sum_cloud>650):                   #CLOUD
        if(ex==1):
            filename = 'models\\SVR\\model_svr_cloud.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
        if (ex == 2):
            filename = 'models\\Boosting\\model_boosting_cloud.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
        if (ex == 3):
            json_file = open('models\\NN\\model_NN_cloud.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("models\\NN\\model_cloud.h5")
            loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        if (ex == 4):
            y1 = dataset.Shortwave_Radiation
        print(sum_cloud)
        print("cloud")
    elif (sum_cloud < 650):                 #CLEAR
        if (ex == 1):
            filename = 'models\\SVR\\model_svr_clear.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
        if (ex == 2):
            filename = 'models\\Boosting\\model_boosting_clear.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
        if (ex == 3):
            json_file = open('models\\NN\\model_NN_clear.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("models\\NN\\model_clear.h5")
            loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        print(sum_cloud)
        print("clear")

    y_pred = loaded_model.predict(X_scaled)
    Y_pred_unscaled = Scaler_Y.inverse_transform(y_pred)
    for k in range(len(X)):
        if (X[k][5] == 0 or Y_pred_unscaled[k] < 0):
            Y_pred_unscaled[k] = 0

    Kwh_pred = ((Y_pred_unscaled / 1000) * 1020 * 0.18)
    fig = Figure(figsize=(6, 4), dpi=100)
    shmeia = np.arange(0, 24, 1)

    ax1 = fig.add_subplot(111).plot(shmeia, Kwh_pred,Kwh , marker='o')
    #fig.add_subplot(111).plot(shmeia, Y, label = 'Pred')
    canvas = FigureCanvasTkAgg(fig, master=PageTwo)  # A tk.DrawingArea.
    canvas.get_tk_widget().place(x=250,y=50)

    axes_names= Label(PageTwo, image=image_name_axes)
    axes_names.place(x=330, y=100)

    diafora = Diafora_Kwh(Kwh_pred, Kwh)
    Sum_Real_Kwh = Sum_Kwh(Kwh)
    Sum_Pred_Kwh = Sum_Kwh(Kwh_pred)

    Label_Text_Real_Kwh = Label(PageTwo, text="Real Kwh:")
    Label_Text_Real_Kwh.place(x=300, y=450)
    Label_Sum_Real_Kwh = Label(PageTwo, text=Sum_Real_Kwh)
    Label_Sum_Real_Kwh.place(x=355, y=450)

    Label_Text_Pred_Kwh = Label(PageTwo, text="Pred Kwh:")
    Label_Text_Pred_Kwh.place(x=420, y=450)
    Label_Sum_Pred_Kwh = Label(PageTwo, text=Sum_Pred_Kwh)
    Label_Sum_Pred_Kwh.place(x=475, y=450)

    Label_text_diaforas = Label(PageTwo, text= "Dif:" )
    Label_text_diaforas.place(x=600, y =450)
    Label_diafora = Label(PageTwo, text = diafora )
    Label_diafora.place(x=620, y=450)


def Diafora_Kwh(Kwh_pred,Kwh):
    diafora = 0

    for i in range(len(Kwh)):
        temp_diafora = Kwh_pred[i] - Kwh[i]
        diafora = diafora + temp_diafora
    return(diafora)


def Sum_Kwh(Kwh):
    sum=0
    for i in range(len(Kwh)):
        sum = sum + Kwh[i]
    return(sum)


def Map():
    #lat = 21.642000
    #lon = 39.671167
    #url = "https://static-maps.yandex.ru/1.x/?lang=en_US&ll=" + str(lat) + "," + str(lon) + "&z=18&size=350,350&l=sat,skl"
    #r = requests.get(url, stream=True)
    #downloaded_file = open("map_solar_panel.png", "wb")
    #for chunk in r.iter_content(chunk_size=256):
    #    if chunk:
    #        downloaded_file.write(chunk)

    photo_map = PhotoImage(file="sun-hat.gif")
    map_photo = Label(PageOne, image=photo_map)
    map_photo.pack()


def Real_time_dataminig():
    currentDay = datetime.now().day
    currentMonth = datetime.now().month
    currentYear = datetime.now().year

    hour = 0

    dt = datetime(currentYear, currentMonth, currentDay, hour )
    timestamp = dt.replace(tzinfo=timezone.utc).timestamp()
    timestamp = int(timestamp)
    print(timestamp)

    lat = 39.654380
    lon = 21.674988

    url = "https://api.darksky.net/forecast/8d4262c6707c537641e9e00da98a546a/"+str(lat)+","+str(lon)+","+str(timestamp)+"?exclude=currently,minutely,daily,alerts,flags"
    response = requests.get(url)

    datas = pd.read_json(response.url)
    hourl_obs=datas.hourly['data']
    hourly_1 = hourl_obs[0]
    teliko = pd.DataFrame.from_dict(hourly_1,orient='index')
    teliko = teliko.T

    for i in range(23):
        hourly_1 =hourl_obs[i+1]
        prosthiki_data = pd.DataFrame.from_dict(hourly_1,orient='index')
        prosthiki_data = prosthiki_data.T
        teliko = teliko.append(prosthiki_data)
    return(teliko)


def Real_Time_Pred(dataset,final_forecast_dataset):

    currentDay = datetime.now().day
    currentMonth = datetime.now().month

    for i in range(len(dataset)):
        if (dataset.Month[i] == currentMonth and dataset.Day[i] == currentDay ):
            shmeio = i      # Arxh ths hmeras (00:00)
            break

    dataset_cut= dataset[shmeio : shmeio + 24]

    Y = np.array(dataset_cut[['Shortwave_Radiation']])

    Scaler_Y = preprocessing.StandardScaler().fit(Y)
    Y_scaled = Scaler_Y.transform(Y)

    X_Calc_rad = np.array(dataset_cut[['Calc_radiation']])

    X_forecast_dataset = np.array(final_forecast_dataset[['temperature','icon','humidity', 'windSpeed', 'cloudCover' ]])
    for i in range(24):
        # TEMPERATURE
        X_forecast_dataset[i][0] = (X_forecast_dataset[i][0] - 32) / 1.8

        # SUNSHINE DURATION
        if (X_forecast_dataset[i][1] == "clear-day"):
            X_forecast_dataset[i][1] = 60
        if (X_forecast_dataset[i][1] == "clear-night"):
            X_forecast_dataset[i][1] = 0
        if (X_forecast_dataset[i][1] == "rain"):
            X_forecast_dataset[i][1] = 0
        if (X_forecast_dataset[i][1] == "snow"):
            X_forecast_dataset[i][1] = 0
        if (X_forecast_dataset[i][1] == "sleet"):
            X_forecast_dataset[i][1] = 0
        if (X_forecast_dataset[i][1] == "wind"):
            X_forecast_dataset[i][1] = 45
        if (X_forecast_dataset[i][1] == "fog"):
            X_forecast_dataset[i][1] = 15
        if (X_forecast_dataset[i][1] == "cloudy"):
            X_forecast_dataset[i][1] = 10
        if (X_forecast_dataset[i][1] == "partly-cloudy-day"):
            X_forecast_dataset[i][1] = 25
        if (X_forecast_dataset[i][1] == "partly-cloudy-night"):
            X_forecast_dataset[i][1] = 0

        #HUMIDITY
        X_forecast_dataset[i][2] = X_forecast_dataset[i][2] * 100

        #CLOUD COVER
        X_forecast_dataset[i][4] = X_forecast_dataset[i][4] * 100

    Final_X_Forecasting= np.append(X_forecast_dataset, X_Calc_rad , axis=1)

    Scaler_X = preprocessing.StandardScaler().fit(Final_X_Forecasting)

    Final_X_Forecasting_scaled = Scaler_X.transform(Final_X_Forecasting)

    count = 0
    sum_cloud = 0
    for j in range(len(Final_X_Forecasting)):
        if (count >= 8 and count <= 20):
            sum_cloud = sum_cloud + Final_X_Forecasting[j][4]

        count = count + 1

    if (sum_cloud > 1500):
        filename = 'models\\SVR\\model_svr_total_cloud.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
    elif (sum_cloud > 800):
        filename = 'models\\SVR\\model_svr_cloud.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
    elif (sum_cloud < 800):
        filename = 'models\\SVR\\model_svr_clear.sav'
        loaded_model = pickle.load(open(filename, 'rb'))

    y_pred = loaded_model.predict(Final_X_Forecasting_scaled)

    Y_pred_unscaled = Scaler_Y.inverse_transform(y_pred)
    for k in range(len(Final_X_Forecasting)):
        if (Final_X_Forecasting[k][5] == 0 or Y_pred_unscaled[k] < 0):
            Y_pred_unscaled[k] = 0

    Kwh_pred = ((Y_pred_unscaled / 1000) * 1020 * 0.18)

    for q in range(len(Kwh_pred)):
        if (Kwh_pred[q]> 150):
            Kwh_pred[q] = Kwh_pred[q] -20

    print(Kwh_pred)

    saveFile = open('Kwh_predict.txt', 'w')
    saveFile.write(str(Kwh_pred))
    saveFile.close()

    fig_real_pred = Figure(figsize=(5, 3), dpi=90)
    shmeia = np.arange(0, 24, 1)

    ax1 = fig_real_pred.add_subplot(111)
    ax1 = ax1.plot(shmeia, Kwh_pred, marker='o' )

    canvas_real = FigureCanvasTkAgg(fig_real_pred, master=PageThree)  # A tk.DrawingArea.
    canvas_real.get_tk_widget().place(x=800, y=150)


def Weather_signal(weather):
    if (weather == "clear-day"):
        return(1)
    if (weather == "clear-night"):
        return(2)
    if (weather == "rain"):
        return(3)
    if (weather == "snow"):
        return(3)
    if (weather == "sleet"):
        return (3)
    if (weather == "wind"):
        return (6)
    if (weather == "fog"):
        return (4)
    if (weather == "cloudy"):
        return (5)
    if (weather == "partly-cloudy-day"):
        return (6)
    if (weather == "partly-cloudy-night"):
        return (7)

#START APP##############################################################################################################

window(root)

dataset = pd.read_csv('1-10-17#1-10-18.csv', delimiter=";", header=11)

Home = Frame(root)
PageOne = Frame(root)
PageTwo = Frame(root)
PageThree = Frame(root)

#CREATE BACKGROUND######################################################################################################

background_image=PhotoImage(file="photos\\fonto_1.png")
background_label = Label(Home, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

background_image_2=PhotoImage(file="photos\\fonto_1.png")
background_labe2 = Label(PageOne, image=background_image)
background_labe2.place(x=0, y=0, relwidth=1, relheight=1)

background_image_3=PhotoImage(file="photos\\fonto_1.png")
background_labe3 = Label(PageTwo, image=background_image)
background_labe3.place(x=0, y=0, relwidth=1, relheight=1)

background_image_4=PhotoImage(file="photos\\fonto_1.png")
background_labe4 = Label(PageThree, image=background_image)
background_labe4.place(x=0, y=0, relwidth=1, relheight=1)

for frame in(Home,PageOne,PageTwo,PageThree):
    frame.grid(row=0,column=0, sticky ='news',padx=50,pady= 50)




#BUTTON 1###############################################################################################################
b1 = ttk.Button(Home,text= "left",command = lambda:raise_frame(PageOne))
photo1 = PhotoImage(file="photos\\page_one.png")
small_photo1 = photo1.subsample(4,4)
b1.config(image = small_photo1)

#BUTTON BACK HOME
b_back = ttk.Button(PageOne,text = "Home",command = lambda :raise_frame(Home))
b_back.pack(side = BOTTOM)

#NEXT BUTTON PAGETWO
b_back = ttk.Button(PageOne,text = "Next",command = lambda :raise_frame(PageTwo))
b_back.place(x=850, y= 222)

#INSIDE PAGEONE
Google_map_Button =ttk.Button(PageOne,text = "Done",command = lambda:First_Page_View() )
Google_map_Button.pack(side = BOTTOM)

Label_1 = Label(PageOne,text="Full Name:")
Label_1.place(x=180,y=10)
Entry1 = Entry(PageOne, width = 50)
Entry1.place(x=270, y= 10)

Label_2 = Label(PageOne,text="Region:")
Label_2.place(x=180,y=40)
Entry2 = Entry(PageOne, width = 20)
Entry2.place(x=270, y= 40)

var_Lang = StringVar()
var_Lon = StringVar()

Label_3 = Label(PageOne,text="Langitude:")
Label_3.place(x=180,y=70)
Entry3 = Entry(PageOne, textvariable = var_Lang, width = 10)
var_Lang.set('21.674988')
Entry3.place(x=270, y= 70)

Label_4 = Label(PageOne,text="Longitude:")
Label_4.place(x=180,y=100)
Entry4 = Entry(PageOne,textvariable = var_Lon, width = 10)
var_Lon.set('39.654380')
Entry4.place(x=270, y= 100)

#CREATE SATELLITE IMAGE
lat = 21.674988
lon = 39.654380

url = "https://static-maps.yandex.ru/1.x/?lang=en_US&ll=" + str(lat) + "," + str(lon) + "&z=18&size=350,350&l=sat"
r = requests.get(url, stream=True)
if r.status_code == 200:
    with open("1.jpg", 'wb') as f:
        f.write(r.content)

example = Image.open("photos\\1.jpg")
example = example.resize((250, 250), Image.ANTIALIAS)
tkimage = ImageTk.PhotoImage(example)
Label(PageOne, image=tkimage).place(x=400, y=300)


#BUTTON 2###############################################################################################################
b2 = ttk.Button(Home,text= "Center" ,command = lambda:raise_frame(PageTwo))
photo2 = PhotoImage(file="photos\\page_two.png")
small_photo2 = photo2.subsample(4,4)
b2.config(image = small_photo2)

#BUTTON BACK HOME
b_back = ttk.Button(PageTwo,text = "Home",command = lambda :raise_frame(Home))
b_back.pack(side = BOTTOM)

#NEXT BUTTON PAGETHREE
b_back = ttk.Button(PageTwo,text = "Next",command = lambda :raise_frame(PageThree))
b_back.place(x=850, y= 222)

#CALENDAR
Label_day = Label(PageTwo,text="Day:")
Label_day.place(x=600,y=0)
Entry_day = Entry(PageTwo, width = 3)
Entry_day.place(x=630, y=2)

Label_month = Label(PageTwo,text="Month:")
Label_month.place(x=655,y=0)
Entry_month = Entry(PageTwo, width = 3)
Entry_month.place(x=705, y=2)

Label_year = Label(PageTwo,text="Year:")
Label_year.place(x=730,y=0)
Entry_year = Entry(PageTwo, width = 4)
Entry_year.place(x=760, y=2)

b_svr = ttk.Button(PageTwo,text= "OK!", command = lambda :Algorithms(1, dataset))
b_svr.place(x=800, y= 0)

#BUTTON ALL MACHINE ALGORITHM

image_name_axes = PhotoImage(file="photos\\pred_real.png")

b_svr = ttk.Button(PageTwo,text= "SVR", command = lambda :Algorithms(1,dataset) )
b_svr.place(x=30, y= 50)
b_Boosting = ttk.Button(PageTwo,text= "Boosting", command = lambda :Algorithms(2,dataset) )
b_Boosting.place(x=30, y= 70)
b_neural = ttk.Button(PageTwo,text= "Neural", command = lambda :Algorithms(3,dataset) )
b_neural.place(x=30, y= 90)


#BUTTON 3###############################################################################################################
b3 = ttk.Button(Home,text= "left" ,command = lambda:raise_frame(PageThree))
photo3 = PhotoImage(file="photos\\page_three.png")
small_photo3 = photo3.subsample(4,4)
b3.config(image = small_photo3)

Label_name_of_Region = Label(PageThree,text="Agioi Theodoroi, Trikala" )
Label_name_of_Region.config(font=("street light italic", 30))
Label_name_of_Region.place(x= 70, y = 50)

#WEATHER DATA LIVE
currentSecond= datetime.now().second
currentMinute = datetime.now().minute
currentHour = datetime.now().hour
currentDay = datetime.now().day
currentMonth = datetime.now().month
currentYear = datetime.now().year

#DAY MONTH YEAR HOUR MINUTE
day = Label(PageThree, text=currentDay )
day.config(font=("street light italic", 13))
day.place(x= 520, y = 120)
diaxwrist1 = Label(PageThree, text="/" )
diaxwrist1.config(font=("street light italic", 13))
diaxwrist1.place(x= 540, y = 120)

month = Label(PageThree, text=currentMonth)
month.config(font=("street light italic", 13))
month.place(x= 560, y = 120)
diaxwrist2 = Label(PageThree, text="/" )
diaxwrist2.config(font=("street light italic", 13))
diaxwrist2.place(x= 580, y = 120)

year = Label(PageThree, text=currentYear )
year.config(font=("street light italic", 13))
year.place(x= 600, y = 120)

hour = Label(PageThree, text=currentHour)
hour.config(font=("street light italic", 13))
hour.place(x= 540, y = 150)

diaxwrist3 = Label(PageThree, text=":" )
diaxwrist3.config(font=("street light italic", 13))
diaxwrist3.place(x= 570, y = 150)

minute = Label(PageThree, text=currentMinute )
minute.config(font=("street light italic", 13))
minute.place(x= 590, y = 150)

final_forecast_dataset = Real_time_dataminig()

#WEATHER FORECAST
now_weather = (final_forecast_dataset.icon[currentHour -1  : currentHour])
h_3_ahead_weather = (final_forecast_dataset.icon[currentHour +2  : currentHour+3])
h_6_ahead_weather= (final_forecast_dataset.icon[currentHour +5  : currentHour+6])

#WEATHER IMAGES
clear_day = PhotoImage(file="weather_icon\\clear_day.png")
clear_night = PhotoImage(file="weather_icon\\clear_night.png")
partly_cloudy_day = PhotoImage(file="weather_icon\\partly_cloudy_day.png")
partly_cloudy_night = PhotoImage(file="weather_icon\\partly_cloudy_night.png")
cloudy_day = PhotoImage(file="weather_icon\\cloudy_day.png")
rainy_day = PhotoImage(file="weather_icon\\rainy_day.png")
foggy_day = PhotoImage(file="weather_icon\\foggy_day.png")

small_partly_cloudy_day = partly_cloudy_day.subsample(5,5)
small_clear_day = clear_day.subsample(5,5)
small_clear_night = clear_night.subsample(5,5)
small_partly_cloudy_night = partly_cloudy_night.subsample(5,5)
small_cloudy_day = cloudy_day.subsample(5,5)
small_rainy_day = rainy_day.subsample(5,5)
small_foggy_day = foggy_day.subsample(5,5)

big_partly_cloudy_day = partly_cloudy_day.subsample(3,3)
big_clear_day = clear_day.subsample(3,3)
big_clear_night = clear_night.subsample(3,3)
big_partly_cloudy_night = partly_cloudy_night.subsample(3,3)
big_cloudy_day = cloudy_day.subsample(3,3)
big_rainy_day = rainy_day.subsample(3,3)
big_foggy_day = foggy_day.subsample(3,3)

Signal_weather_icon_now = Weather_signal(now_weather[0])
Signal_weather_icon_3h_ahead = Weather_signal(h_3_ahead_weather[0])
Signal_weather_icon_6h_ahead = Weather_signal(h_6_ahead_weather[0])

if (Signal_weather_icon_now == 1 ):
    weather_now = ttk.Label(PageThree, image = big_clear_day)
    weather_now.place(x= 120 , y =170)
    celsius = (final_forecast_dataset.temperature[currentHour - 1: currentHour] - 32) / 1.8
    degree_now = ttk.Label(PageThree, text=int(celsius))
    degree_now.config(font=("street light italic", 18))
    degree_now.place(x=280, y=280)
    celsius_sumbol = Label(PageThree, text="°C")
    celsius_sumbol.config(font=("street light italic", 18))
    celsius_sumbol.place(x=320, y=280)

if (Signal_weather_icon_now == 2 ):
    weather_now = ttk.Label(PageThree, image = big_clear_night)
    weather_now.place(x= 120 , y =170)
    celsius = (final_forecast_dataset.temperature[currentHour - 1: currentHour] - 32) / 1.8
    degree_now = ttk.Label(PageThree, text=int(celsius))
    degree_now.config(font=("street light italic", 18))
    degree_now.place(x=280, y=280)
    celsius_sumbol = Label(PageThree, text="°C")
    celsius_sumbol.config(font=("street light italic", 18))
    celsius_sumbol.place(x=320, y=280)

if (Signal_weather_icon_now == 3 ):
    weather_now = ttk.Label(PageThree, image = big_rainy_day)
    weather_now.place(x= 120 , y =170)
    celsius = (final_forecast_dataset.temperature[currentHour - 1: currentHour] - 32) / 1.8
    degree_now = ttk.Label(PageThree, text=int(celsius))
    degree_now.config(font=("street light italic", 18))
    degree_now.place(x=280, y=280)
    celsius_sumbol = Label(PageThree, text="°C")
    celsius_sumbol.config(font=("street light italic", 18))
    celsius_sumbol.place(x=320, y=280)

if (Signal_weather_icon_now == 4 ):
    weather_now = ttk.Label(PageThree, image = big_foggy_day)
    weather_now.place(x= 120 , y =170)
    celsius = (final_forecast_dataset.temperature[currentHour - 1: currentHour] - 32) / 1.8
    degree_now = ttk.Label(PageThree, text=int(celsius))
    degree_now.config(font=("street light italic", 18))
    degree_now.place(x=280, y=280)
    celsius_sumbol = Label(PageThree, text="°C")
    celsius_sumbol.config(font=("street light italic", 18))
    celsius_sumbol.place(x=320, y=280)

if (Signal_weather_icon_now == 5 ):
    weather_now = ttk.Label(PageThree, image = big_cloudy_day)
    weather_now.place(x= 120 , y =170)
    celsius = (final_forecast_dataset.temperature[currentHour - 1: currentHour] - 32) / 1.8
    degree_now = ttk.Label(PageThree, text=int(celsius))
    degree_now.config(font=("street light italic", 18))
    degree_now.place(x=280, y=280)
    celsius_sumbol = Label(PageThree, text="°C")
    celsius_sumbol.config(font=("street light italic", 18))
    celsius_sumbol.place(x=320, y=280)

if (Signal_weather_icon_now == 6 ):
    weather_now = ttk.Label(PageThree, image = big_partly_cloudy_day)
    weather_now.place(x= 120 , y =170)
    celsius = (final_forecast_dataset.temperature[currentHour - 1: currentHour] - 32) / 1.8
    degree_now = ttk.Label(PageThree, text=int(celsius))
    degree_now.config(font=("street light italic", 18))
    degree_now.place(x=280, y=280)
    celsius_sumbol = Label(PageThree, text="°C")
    celsius_sumbol.config(font=("street light italic", 18))
    celsius_sumbol.place(x=320, y=280)

if (Signal_weather_icon_now == 7 ):
    weather_now = ttk.Label(PageThree, image = big_partly_cloudy_night)
    weather_now.place(x= 120 , y =170)
    celsius = (final_forecast_dataset.temperature[currentHour - 1: currentHour] - 32) / 1.8
    degree_now = ttk.Label(PageThree, text=int(celsius))
    degree_now.config(font=("street light italic", 18))
    degree_now.place(x=280, y=280)
    celsius_sumbol = Label(PageThree, text="°C")
    celsius_sumbol.config(font=("street light italic", 18))
    celsius_sumbol.place(x=320, y=280)

#PRINT WEATHER IMAGE
hour_3_time = ttk.Label(PageThree, text= " +3h")
hour_3_time.config(font=("street light italic", 15))
hour_3_time.place(x=150, y=410)

if (Signal_weather_icon_3h_ahead == 1 ):
    weather_now = ttk.Label(PageThree, image = small_clear_day)
    weather_now.place(x= 120 , y =450)
    celsius = (final_forecast_dataset.temperature[currentHour +2  : currentHour+3] - 32) / 1.8
    degree_now = ttk.Label(PageThree, text=int(celsius))
    degree_now.config(font=("street light italic", 18))
    degree_now.place(x=240, y=500)
    celsius_sumbol = Label(PageThree, text="°C")
    celsius_sumbol.config(font=("street light italic", 18))
    celsius_sumbol.place(x=280, y=500)

if (Signal_weather_icon_3h_ahead == 2 ):
    weather_now = ttk.Label(PageThree, image = small_clear_night)
    weather_now.place(x= 120 , y =450)
    celsius = (final_forecast_dataset.temperature[currentHour +2  : currentHour+3] - 32) / 1.8
    degree_now = ttk.Label(PageThree, text=int(celsius))
    degree_now.config(font=("street light italic", 18))
    degree_now.place(x=240, y=500)
    celsius_sumbol = Label(PageThree, text="°C")
    celsius_sumbol.config(font=("street light italic", 18))
    celsius_sumbol.place(x=280, y=500)

if (Signal_weather_icon_3h_ahead == 3 ):
    weather_now = ttk.Label(PageThree, image = small_rainy_day)
    weather_now.place(x= 120 , y =450)
    celsius = (final_forecast_dataset.temperature[currentHour +2  : currentHour+3] - 32) / 1.8
    degree_now = ttk.Label(PageThree, text=int(celsius))
    degree_now.config(font=("street light italic", 18))
    degree_now.place(x=240, y=500)
    celsius_sumbol = Label(PageThree, text="°C")
    celsius_sumbol.config(font=("street light italic", 18))
    celsius_sumbol.place(x=280, y=500)

if (Signal_weather_icon_3h_ahead == 4 ):
    weather_now = ttk.Label(PageThree, image = small_foggy_day)
    weather_now.place(x= 120 , y =450)
    celsius = (final_forecast_dataset.temperature[currentHour +2  : currentHour+3] - 32) / 1.8
    degree_now = ttk.Label(PageThree, text=int(celsius))
    degree_now.config(font=("street light italic", 18))
    degree_now.place(x=240, y=500)
    celsius_sumbol = Label(PageThree, text="°C")
    celsius_sumbol.config(font=("street light italic", 18))
    celsius_sumbol.place(x=280, y=500)

if (Signal_weather_icon_3h_ahead == 5 ):
    weather_now = ttk.Label(PageThree, image = small_cloudy_day)
    weather_now.place(x= 120 , y =450)
    celsius = (final_forecast_dataset.temperature[currentHour +2  : currentHour+3] - 32) / 1.8
    degree_now = ttk.Label(PageThree, text=int(celsius))
    degree_now.config(font=("street light italic", 18))
    degree_now.place(x=240, y=500)
    celsius_sumbol = Label(PageThree, text="°C")
    celsius_sumbol.config(font=("street light italic", 18))
    celsius_sumbol.place(x=280, y=500)

if (Signal_weather_icon_3h_ahead == 6 ):
    weather_now = ttk.Label(PageThree, image = small_partly_cloudy_day)
    weather_now.place(x= 120 , y =450)
    celsius = (final_forecast_dataset.temperature[currentHour +2  : currentHour+3] - 32) / 1.8
    degree_now = ttk.Label(PageThree, text=int(celsius))
    degree_now.config(font=("street light italic", 18))
    degree_now.place(x=240, y=500)
    celsius_sumbol = Label(PageThree, text="°C")
    celsius_sumbol.config(font=("street light italic", 18))
    celsius_sumbol.place(x=280, y=500)

if (Signal_weather_icon_3h_ahead == 7 ):
    weather_now = ttk.Label(PageThree, image = small_partly_cloudy_night)
    weather_now.place(x= 120 , y =450)
    celsius = (final_forecast_dataset.temperature[currentHour +2  : currentHour+3] - 32) / 1.8
    degree_now = ttk.Label(PageThree, text=int(celsius))
    degree_now.config(font=("street light italic", 16))
    degree_now.place(x=240, y=500)
    celsius_sumbol = Label(PageThree, text="°C")
    celsius_sumbol.config(font=("street light italic", 16))
    celsius_sumbol.place(x=280, y=500)

#PRINT WEATHER IMAGE
hour_6_time = ttk.Label(PageThree, text= " +6h")
hour_6_time.config(font=("street light italic", 15))
hour_6_time.place(x=450, y=410)

if (Signal_weather_icon_6h_ahead == 1 ):
    weather_now = ttk.Label(PageThree, image = small_clear_day)
    weather_now.place(x= 420 , y =450)
    celsius = (final_forecast_dataset.temperature[currentHour +2  : currentHour+3] - 32) / 1.8
    degree_now = ttk.Label(PageThree, text=int(celsius))
    degree_now.config(font=("street light italic", 16))
    degree_now.place(x=540, y=500)
    celsius_sumbol = Label(PageThree, text="°C")
    celsius_sumbol.config(font=("street light italic", 16))
    celsius_sumbol.place(x=580, y=500)

if (Signal_weather_icon_6h_ahead == 2 ):
    weather_now = ttk.Label(PageThree, image = small_clear_night)
    weather_now.place(x= 420 , y =450)
    celsius = (final_forecast_dataset.temperature[currentHour +2  : currentHour+3] - 32) / 1.8
    degree_now = ttk.Label(PageThree, text=int(celsius))
    degree_now.config(font=("street light italic", 16))
    degree_now.place(x=540, y=500)
    celsius_sumbol = Label(PageThree, text="°C")
    celsius_sumbol.config(font=("street light italic", 16))
    celsius_sumbol.place(x=580, y=500)

if (Signal_weather_icon_6h_ahead == 3 ):
    weather_now = ttk.Label(PageThree, image = small_rainy_day)
    weather_now.place(x= 420 , y =450)
    celsius = (final_forecast_dataset.temperature[currentHour +2  : currentHour+3] - 32) / 1.8
    degree_now = ttk.Label(PageThree, text=int(celsius))
    degree_now.config(font=("street light italic", 16))
    degree_now.place(x=540, y=500)
    celsius_sumbol = Label(PageThree, text="°C")
    celsius_sumbol.config(font=("street light italic", 16))
    celsius_sumbol.place(x=580, y=500)

if (Signal_weather_icon_6h_ahead == 4 ):
    weather_now = ttk.Label(PageThree, image = small_foggy_day)
    weather_now.place(x= 420 , y =450)
    celsius = (final_forecast_dataset.temperature[currentHour +2  : currentHour+3] - 32) / 1.8
    degree_now = ttk.Label(PageThree, text=int(celsius))
    degree_now.config(font=("street light italic", 16))
    degree_now.place(x=540, y=500)
    celsius_sumbol = Label(PageThree, text="°C")
    celsius_sumbol.config(font=("street light italic", 16))
    celsius_sumbol.place(x=580, y=500)

if (Signal_weather_icon_6h_ahead == 5 ):
    weather_now = ttk.Label(PageThree, image = small_cloudy_day)
    weather_now.place(x= 420 , y =450)
    celsius = (final_forecast_dataset.temperature[currentHour +2  : currentHour+3] - 32) / 1.8
    degree_now = ttk.Label(PageThree, text=int(celsius))
    degree_now.config(font=("street light italic", 16))
    degree_now.place(x=540, y=500)
    celsius_sumbol = Label(PageThree, text="°C")
    celsius_sumbol.config(font=("street light italic", 16))
    celsius_sumbol.place(x=580, y=500)

if (Signal_weather_icon_6h_ahead == 6 ):
    weather_now = ttk.Label(PageThree, image = small_partly_cloudy_day)
    weather_now.place(x= 420 , y =450)
    celsius = (final_forecast_dataset.temperature[currentHour +2  : currentHour+3] - 32) / 1.8
    degree_now = ttk.Label(PageThree, text=int(celsius))
    degree_now.config(font=("street light italic", 16))
    degree_now.place(x=540, y=500)
    celsius_sumbol = Label(PageThree, text="°C")
    celsius_sumbol.config(font=("street light italic", 16))
    celsius_sumbol.place(x=580, y=500)

if (Signal_weather_icon_6h_ahead == 7 ):
    weather_now = ttk.Label(PageThree, image = small_partly_cloudy_night)
    weather_now.place(x= 420 , y =450)
    celsius = (final_forecast_dataset.temperature[currentHour +2  : currentHour+3] - 32) / 1.8
    degree_now = ttk.Label(PageThree, text=int(celsius))
    degree_now.config(font=("street light italic", 16))
    degree_now.place(x=540, y=500)
    celsius_sumbol = Label(PageThree, text="°C")
    celsius_sumbol.config(font=("street light italic", 16))
    celsius_sumbol.place(x=580, y=500)

#PREDICTION BUTTON
button_predict_real_time = ttk.Button(PageThree,text = "Forecast",command = lambda :Real_Time_Pred(dataset,final_forecast_dataset))
button_predict_real_time.place(x= 700,y=200)

#BUTTON BACK HOME
b_back = ttk.Button(PageThree,text = "Home",command = lambda :raise_frame(Home))
b_back.pack(side = BOTTOM)

raise_frame(Home)

b1.pack(side=LEFT, expand=YES,fill=BOTH,padx=60,pady= 150)
b2.pack(side=LEFT, expand=YES,padx=60,pady= 150)
b3.pack(side=LEFT, expand=YES,padx=60,pady= 150)

root.mainloop()


