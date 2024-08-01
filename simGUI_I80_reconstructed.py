'''
Simulation GUI to observe (given recording csv) the original I80 scenarios and 
simulated versions where an agent replaces an ego vehicle in the dataset
'''
from tkinter import Tk, Canvas, Frame, BOTH, BOTTOM, TOP, StringVar, Label
from tkinter.ttk import Button
import pandas as pd
import time
import numpy as np
import os
from Params import Params

class SimGUI(Frame):
    def __init__(self, *args):
        super(SimGUI,self).__init__()
        self.fps = 1
        self.Time_Step = 0.1 #sec

        self.proportion_x = 1;
        self.proportion_y = 1;
        self.lane_w = 12
        self.car_w = 6

        self.offset = 0 *self.proportion_x
        self.start_x = 20
        self.start_y = 50
        self.end_road = self.start_x + 1000*self.proportion_x - self.offset
        self.start_merging = self.start_x + 385 * self.proportion_x - self.offset
        self.end_merging = self.start_x + 720 * self.proportion_x - self.offset
        self.mid_y = self.start_y + self.lane_w * self.proportion_y
        self.end_y = self.start_y + self.lane_w * 2 * self.proportion_y

        self.canvas_h = self.start_y + 100*self.proportion_y + self.start_y
        self.canvas_w = self.end_road + self.start_x
        
        self.cars = []
        self.episode_df = args[0]
        self.Frame_ID = int(self.episode_df.iloc[0].Frame_ID)
        self.ego_car_ID = args[1]
        self.root = args[2]
        self.root.geometry(""+str(int(self.canvas_w)) + "x" + str(int(self.canvas_h)+100) + "+50+200")
        self.canvas = Canvas(self)

        self.initUI()

    def initUI(self):
        self.master.title("Ego Car_ID: " + str(self.ego_car_ID))
        self.pack(fill=BOTH, expand=1)

        self.canvas.create_line(self.start_x, self.start_y, self.end_road, self.start_y,width=2) # first line
        self.canvas.create_line(self.start_x, self.mid_y, self.start_merging, self.mid_y,width=2) # second line: start to start_merging
        self.canvas.create_line(self.start_x, self.end_y, self.end_merging, self.end_y,width=2) # third line

        self.canvas.create_line(self.start_merging, self.mid_y,self.end_merging+5*self.proportion_x, self.mid_y,dash=(4, 4),width=2) # second line: start_merging to end_merging 
        self.canvas.create_line(self.end_merging,self.end_y,self.end_merging+5*self.proportion_x,self.mid_y,width=2)
        self.canvas.create_line(self.end_merging+5*self.proportion_x,self.mid_y,self.end_road,self.mid_y,width=2)

        self.button = Button(self,text="Start Sim",command=self.start_sim)
        self.Frame_ID_text = StringVar()
        self.Frame_ID_text.set("Frame_ID: " + str(self.Frame_ID))
        self.Frame_ID_label = Label(self, textvariable=self.Frame_ID_text)

        self.Frame_ID_label.pack(side=TOP)
        self.button.pack(side=BOTTOM)
        self.canvas.pack(fill=BOTH, expand=1)

    def start_sim(self):
        self.frame_no = 0
        self.create_cars()
        self.update_sim()

    def create_cars(self):
        #print("Create Cars - Frame_ID " + str(self.Frame_ID))
        self.cars = []
        for dummy,car in self.episode_df[(self.episode_df['Frame_ID']==self.Frame_ID)].iterrows():
            if int(car['Vehicle_ID']) == self.ego_car_ID:
                colour = "#fb0"
            else:
                colour = "#f50"
            car_x = self.start_x + (car['Local_Y'] - car['Vehicle_Length'])*self.proportion_x - self.offset
            car_y = self.start_y + ((car['Lane_ID']-6)*self.lane_w + (self.lane_w-self.car_w)/2)*self.proportion_y
            delta_x = 0
            delta_y = 0
            car_rectangle = self.canvas.create_rectangle(car_x, car_y, car_x + car['Vehicle_Length']*self.proportion_x, car_y + self.car_w*self.proportion_y, outline=colour, fill=colour)
            car_rectangle_text = self.canvas.create_text(car_x+int(car['Vehicle_Length'])*self.proportion_x*0.5, car_y+int(self.car_w)*self.proportion_y*0.5, font=("Arial", 10), text=str(int(car['Vehicle_ID'])))
            self.cars.append([car_rectangle,car_rectangle_text,car['Vehicle_ID'],delta_x,delta_y,car['Local_Y'],car['Lane_ID']])
    
    def add_car(self,car):
        colour = "#f50"
        car_x = self.start_x + (car['Local_Y'] - car['Vehicle_Length'])*self.proportion_x - self.offset
        car_y = self.start_y + ((car['Lane_ID']-6)*self.lane_w+ (self.lane_w-self.car_w)/2)*self.proportion_y
        delta_x = 0
        delta_y = 0
        car_rectangle = self.canvas.create_rectangle(car_x, car_y, car_x + car['Vehicle_Length']*self.proportion_x, car_y + self.car_w*self.proportion_y, outline=colour, fill=colour)
        car_rectangle_text = self.canvas.create_text(car_x+int(car['Vehicle_Length'])*self.proportion_x*0.5, car_y+int(self.car_w)*self.proportion_y*0.5, font=("Arial", 10), text=str(int(car['Vehicle_ID'])))
        self.cars.append([car_rectangle,car_rectangle_text,car['Vehicle_ID'],delta_x,delta_y,car['Local_Y'],car['Lane_ID']])

    def update_sim(self):
        if not (self.Frame_ID+1) > self.episode_df.iloc[-1].Frame_ID:
            self.Frame_ID += 1
            #print("Frame_ID: " + str(self.Frame_ID))
            self.Frame_ID_text.set("Frame_ID " + str(self.Frame_ID))
            
            cars_exist = np.zeros((len(self.cars),1));
            for dummy,car in self.episode_df.loc[(self.episode_df['Frame_ID']==self.Frame_ID)].iterrows():
                
                in_cars = False
                for i,temp in enumerate(self.cars):
                    if temp[2] == car['Vehicle_ID']:
                        in_cars = True
                        cars_exist[i,0] = 1;
                        self.cars[i][3] = (car['Local_Y'] - temp[-2])*self.proportion_x/self.fps
                        self.cars[i][4] = (car['Lane_ID'] - temp[-1])*self.lane_w*self.proportion_y/self.fps
                        self.cars[i][-2] = car['Local_Y']
                        self.cars[i][-1] = car['Lane_ID']
                        break
                
                if not in_cars:
                    self.add_car(car)

            # if any car in self.cars is not in df, remove it
            for i,exist in enumerate(cars_exist):
                if exist == 0:
                    #print("Delete Car " + str(self.cars[i][2]))           
                    self.canvas.delete(self.cars[i][0])
                    self.canvas.delete(self.cars[i][1])
                    self.cars.pop(i)

            self.move_frame()
        else:
            #print("Frame_ID: " + str(self.Frame_ID) + " - End")
            time.sleep(1)
            self.quit()
            self.destroy()

    def move_frame(self):
        if self.frame_no < self.fps:
            for car in self.cars:
                self.canvas.move(car[0],car[3],car[4])
                self.canvas.move(car[1],car[3],car[4])

            self.canvas.update()
            self.frame_no += 1
            self.after(int(1000*self.Time_Step/self.fps), self.move_frame)
        else:
            self.frame_no = 0 
            self.update_sim()
def main():
    DATA_PATH = os.path.split(os.getcwd())[0]+"data/"
    PATH_VS = DATA_PATH+"NGSIM_I80_quadruplets/"
    RESULTS_PATH = DATA_PATH+"NGSIM_I80_sim_results/"
    EXPERIMENT_PATH = os.path.split(os.getcwd())[0]+"/experiments/" # Path of the experiment where level-k training files are located
    PATH_DYNs = ["some_title1/dynM56/",
                "some_title1/dynM58/",
                "some_title2/dynM59/",
                "some_title3/dynM57/"]
                
    EGO_ID = 1323
    ego_status = "vs_startsat7/"
    
    for follower_status in ["no_followers/","with_followers/"]:
        print("\n"+follower_status)    
        filename = os.listdir(PATH_VS+follower_status+ego_status+str(EGO_ID))[0]
        sim_df = pd.read_csv(PATH_VS+follower_status+ego_status+str(EGO_ID)+"/"+filename)
        ego_df = sim_df.loc[sim_df['Vehicle_ID']==EGO_ID]
        last_frame = ego_df.loc[ego_df['Local_Y']<=(Params.end_for_car0/0.3048)].Frame_ID.values[-1]
        sim_df = sim_df.loc[sim_df["Frame_ID"]<=last_frame]
        root = Tk()
        SimGUI(sim_df,EGO_ID,root)
        root.mainloop()
        root.destroy()      
       
    for PATH_DYN in PATH_DYNs:
        print("\n"+PATH_DYN)
        for follower_status in ["no_followers/","with_followers/"]:
            print("\n"+follower_status)
            filename = RESULTS_PATH+PATH_DYN+follower_status+ego_status+"egoID"+str(EGO_ID)+".csv"
            sim_df = pd.read_csv(filename)
            sim_df.Local_Y *= 1/0.3048
            sim_df.Mean_Speed *= 1/0.3048
            sim_df.Mean_Accel *= 1/0.3048
            sim_df.Lane_ID = 7-sim_df.Lane_ID
            sim_df.Vehicle_Length *= 1/0.3048
            root = Tk()
            SimGUI(sim_df,EGO_ID,root)
            root.mainloop()
            root.destroy()            
if __name__ == '__main__':
    main()