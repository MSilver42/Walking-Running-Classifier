#note: tested through the usage of every walking and jumping .csv file available, 
#as well as monitoring directory of output files. 
#also added error messages and exceptions: file not selected error, wrong file format exception, 
#prediction already ran exception to ensure smooth user operation with no room for confusion. 
#Kept UI clean and precise.

import tkinter as tk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# UI model for application using Tkinter
class UI(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Walking and Jumping!")
        self.pack()
        self.create_widgets()

    # defining button widgets
    def create_widgets(self):
        self.select_file_button = tk.Button(self)
        self.select_file_button["text"] = "Select '.csv' File"
        self.select_file_button["command"] = self.select_file
        self.select_file_button.pack(side="top")

        self.predict_button = tk.Button(self)
        self.predict_button["text"] = "Run Prediction"
        self.predict_button["command"] = self.predict
        self.predict_button.pack(side="top")
        
        self.quit_button = tk.Button(self, text="Exit", fg="red", command=self.master.destroy)
        self.quit_button.pack(side="bottom")
        
        self.quit_button = tk.Label(self, text="Project Developed by: Amy Li, Mason Silver, Carter Conboy")
        self.quit_button.pack(side="bottom")

    # finding input files
    def select_file(self):
        self.file_path = filedialog.askopenfilename()
        print(self.file_path)
    
    # calling predict function
    def predict(self):
        if not hasattr(self, 'file_path'):
            print("Please select a file first")
            messagebox.showerror("Error", "Please select a file!")
            return
        try:
            if os.path.splitext(self.file_path)[1] != '.csv':
                raise Exception('File extension is not csv!')

            # check for duplicate runs
            if hasattr(self, 'already_ren') and self.already_ran:
                raise Exception('Prediction has already been run!')

            # Load the model
            model = pickle.load(open('model.pkl', 'rb'))

            # Read the input file using pandas
            data = pd.read_csv(self.file_path)

            rows_per_chunk = 1010 # calculated from training data (5 seconds * 202 Hz)

            # remove outliers that are more than 3 standard deviations from the mean
            data = data[(np.abs(data['Absolute acceleration (m/s^2)']) - 
                        np.mean(data['Absolute acceleration (m/s^2)'])) / 
                        np.std(data['Absolute acceleration (m/s^2)']) < 3]

            # Split the data into chunks
            temp = [data[i:i+rows_per_chunk] for i in range(0, data.shape[0], rows_per_chunk)]

            #remove last chunk if it is not 5 seconds long
            if len(temp[len(temp) - 1]) < rows_per_chunk:
                temp.pop()

            #converting the list to a numpy array
            chunks = np.array([temp.values for temp in temp])

            #time column set as 0 through 5
            for chunk in chunks:
                chunk[:,0] = np.linspace(0, 5, chunk.shape[0])

            #flatten
            chunks = np.array([i.flatten() for i in chunks])

            #predicting output
            predictions = model.predict(chunks)

            #creating new np array
            temp = np.repeat(predictions, rows_per_chunk)

            original_data =np.genfromtxt(self.file_path, delimiter=',')

            if temp.shape[0] < original_data.shape[0]:
                 #remove last chunk if it's not 5s long
                temp = np.append(temp, np.repeat(predictions[len(predictions) - 1], original_data.shape[0] - temp.shape[0]))
                
              #output to CSV file
            file_name = os.path.splitext(self.file_path)[0] + '_predictions.csv'
            pd.DataFrame(temp).to_csv(file_name, index=False) 
            
            accuracy = model.score(chunks, predictions)
            accuracy_str = f"Accuracy: {accuracy:.4f}"

            #show accuracy in UI
            if hasattr(self, 'accuracy_label'):
                self.accuracy_label.config(text=accuracy_str)
            else:
                self.accuracy_label = tk.Label(self, text=accuracy_str)
                self.accuracy_label.pack()

                print("Accuracy: ", model.score(chunks, predictions))
                print(temp)
            
            #plotting figure of predictions vs. time
            fig = plt.Figure(figsize=(8, 4), dpi=150)
            ax = fig.add_subplot(111)
            ax.scatter(original_data[:, 0], temp, s=1)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Output')
            ax.set_title('Predictions Vs Time')
            canvas = FigureCanvasTkAgg(fig, master=self)
            canvas.draw()
            canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
            
            #already ran equal to true
            self.already_ran = True

        except Exception as e:
            print(e)
            
root = tk.Tk()
app = UI(master=root)
app.mainloop()
