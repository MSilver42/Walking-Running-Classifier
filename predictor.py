import pickle
import os
import numpy as np
import pandas as pd

def predict(csv_file):

    if not os.path.exists(csv_file):
        raise Exception('File not found')
    elif os.path.splitext(csv_file)[1] != '.csv':
        raise Exception('File extension is not csv')

    # Load the model
    model = pickle.load(open('model.pkl', 'rb'))
    
    # Read the input file using pandas
    data = pd.read_csv(csv_file)

    rows_per_chunk = 1010 # calculated from training data (5 seconds * 202 Hz)


    # remove outliers that are more than 3 standard deviations from the mean
    data = data[(np.abs(data['Absolute acceleration (m/s^2)']) - np.mean(data['Absolute acceleration (m/s^2)'])) / np.std(data['Absolute acceleration (m/s^2)']) < 3]

    # Apply moving average filter
    data['Absolute acceleration (m/s^2)'] = data['Absolute acceleration (m/s^2)'].rolling(window=10).mean()
    data['Linear Acceleration x (m/s^2)'] = data['Linear Acceleration x (m/s^2)'].rolling(window=10).mean()
    data['Linear Acceleration y (m/s^2)'] = data['Linear Acceleration y (m/s^2)'].rolling(window=10).mean()
    data['Linear Acceleration z (m/s^2)'] = data['Linear Acceleration z (m/s^2)'].rolling(window=10).mean()

    # drop rows with NaN values
    data = data.dropna()

    # Split the data into chunks
    temp = [data[i:i+rows_per_chunk] for i in range(0, data.shape[0], rows_per_chunk)]

    # remove last chunk if it is not 5 seconds long
    if len(temp[len(temp) - 1]) < rows_per_chunk:
        temp.pop()

    # Convert the list to a numpy array
    chunks = np.array([temp.values for temp in temp])
    
    # Set time column to 0 through 5
    for chunk in chunks:
        chunk[:,0] = np.linspace(0, 5, chunk.shape[0])
    
    # extract features
    features = feature_extraction(chunks)

    # flatten 
    chunks = np.array([i.flatten() for i in chunks])

    # add features to chunks
    chunks = np.concatenate((chunks, features), axis=1)
    
    # Predict the output
    predictions = model.predict(chunks)

    # create new np array where each prediction is repeated rows_per_chunk times
    temp = np.repeat(predictions, rows_per_chunk)

    original_data =np.genfromtxt(csv_file, delimiter=',')

    if temp.shape[0] < original_data.shape[0]:
        # if the last chunk is not 5 seconds long, fill the rest of the array with the last prediction
        temp = np.append(temp, np.repeat(predictions[len(predictions) - 1], original_data.shape[0] - temp.shape[0]))

    print("Accuracy: ", (len(temp)-np.count_nonzero(temp == "jumping"))/len(temp)*100, "%" )

    # add temp as new column to data
    output = np.column_stack((original_data, temp))

    # delete first row
    output = np.delete(output, 0, 0)

    # save data to csv
    np.savetxt('output.csv', output, delimiter=',', fmt='%s')


def predict_pca(csv_file):

    if not os.path.exists(csv_file):
        raise Exception('File not found')
    elif os.path.splitext(csv_file)[1] != '.csv':
        raise Exception('File extension is not csv')

    # Load the model
    model = pickle.load(open('model_pca.pkl', 'rb'))
    pca = pickle.load(open('pca.pkl', 'rb'))
    
    # Read the input file using pandas
    data = pd.read_csv(csv_file)

    rows_per_chunk = 1010 # calculated from training data (5 seconds * 202 Hz)


    # remove outliers that are more than 3 standard deviations from the mean
    data = data[(np.abs(data['Absolute acceleration (m/s^2)']) - np.mean(data['Absolute acceleration (m/s^2)'])) / np.std(data['Absolute acceleration (m/s^2)']) < 3]

    # Apply moving average filter
    data['Absolute acceleration (m/s^2)'] = data['Absolute acceleration (m/s^2)'].rolling(window=10).mean()
    data['Linear Acceleration x (m/s^2)'] = data['Linear Acceleration x (m/s^2)'].rolling(window=10).mean()
    data['Linear Acceleration y (m/s^2)'] = data['Linear Acceleration y (m/s^2)'].rolling(window=10).mean()
    data['Linear Acceleration z (m/s^2)'] = data['Linear Acceleration z (m/s^2)'].rolling(window=10).mean()

    # drop rows with NaN values
    data = data.dropna()
    
    # Split the data into chunks
    temp = [data[i:i+rows_per_chunk] for i in range(0, data.shape[0], rows_per_chunk)]

    # remove last chunk if it is not 5 seconds long
    if len(temp[len(temp) - 1]) < rows_per_chunk:
        temp.pop()

    # Convert the list to a numpy array
    chunks = np.array([temp.values for temp in temp])
    
    # Set time column to 0 through 5
    for chunk in chunks:
        chunk[:,0] = np.linspace(0, 5, chunk.shape[0])

    # extract features
    features = feature_extraction(chunks)
    
    # flatten features
    features = np.array([i.flatten() for i in features])


    # apply pca
    features = pca.transform(features)

    
    # Predict the output
    predictions = model.predict(features)

    # create new np array where each prediction is repeated rows_per_chunk times
    temp = np.repeat(predictions, rows_per_chunk)

    original_data =np.genfromtxt(csv_file, delimiter=',')

    if temp.shape[0] < original_data.shape[0]:
        # if the last chunk is not 5 seconds long, fill the rest of the array with the last prediction
        temp = np.append(temp, np.repeat(predictions[len(predictions) - 1], original_data.shape[0] - temp.shape[0]))

    print("Accuracy: ", (len(temp)-np.count_nonzero((temp == "jumping") | (temp == "Jumping")))/len(temp)*100, "%")

    # add temp as new column to data
    output = np.column_stack((original_data, temp))

    # delete first row
    output = np.delete(output, 0, 0)

    # save data to csv
    np.savetxt('output.csv', output, delimiter=',', fmt='%s')


def feature_extraction(data):
    temp = []
    for i in data:
        for j in range(1,5):
            temp.append([np.max(i[:, j]), np.min(i[:, j]), np.ptp(i[:, j]), np.mean(i[:, j]), np.median(i[:, j]), np.var(i[:, j]), np.std(i[:, j])])
    temp = np.array(temp)
    temp = temp.reshape(data.shape[0], 28)
    return temp


predict('test (2).csv')