import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from tensorflow.keras.models  import Sequential
from sklearn.preprocessing import StandardScaler 
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2,l1
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from keras import Model
from keras.layers import Input, Dense, Dropout, LSTM
def B_ebola(B0, t, lmbda):
    """
    New function for the time-dependent transmission rate
    Input : B0 : initial transmission rate
            t : time
            lmbda : decay rate of the transmission rate
    Output : B : transmission rate at time t
    """
    return B0 * np.exp(-lmbda * t)
def f_ebola(y,t,B0,lmbda):
    """ 
    Function to solve the model for ebola data
    """
    S,E,Z,R=y
    N=S+Z+E+R
    gamma=1/7
    sigma=1/9.7
    betha=B_ebola(B0,t,lmbda) # B0,t,lmbda
    dSdt=-betha*S*Z /N
    dEdt=betha*S*Z / N - (sigma*E)
    dZdt=(sigma*E) - gamma*Z
    dRdt=gamma*Z
    return np.array([dSdt,dEdt,dZdt,dRdt])
def calculate_precise(name="Guinea"):
    t,Z,Ct=read_data(dataset=name)
    # Use the non linear function to find betha and lambda optimized 
    bds = ((0.0, 0.0), (np.inf, np.inf))
    popt, pcov = curve_fit(non_linear_func, t, Ct, bounds=bds)
    print('Curve-fit: beta0={}, lambda={}.'.format(popt[0], popt[1]))
    U0_ebola=np.array([N_ebola-1,0,1,0])
    # Call odeint to see the solution with the lambda and betha optimized
    sol_ebola = odeint(f_ebola, U0_ebola, t, args=(popt[0], popt[1]))
    S_ebola, E_ebola, Z_ebola, R_ebola = sol_ebola.T
    return Z_ebola
def read_data(dataset=""):
    """ 
    Use panda library to take the wanted data from each dataset. 
    Input : dataset -> Name of the dataset that we want to use 
        Output : t -> Time value of the dataset 
                 Z -> Number of disease at each time t of the dataset 
                 Ct -> Cumulative value of disease from dataset 
    """
    if dataset=="Guinea":
        colonnes_2_3 = Guinea_set.iloc[:, 1:3]
    elif dataset=="Liberia":
        colonnes_2_3 = Liberia_set.iloc[:, 1:3]
    elif dataset=="Sierra Leone":
        colonnes_2_3 = Sierra_set.iloc[:, 1:3]
    else : 
        print("Wrong dataset name : Guinea - Liberia - Sierra Leone")
        return
    valeurs=colonnes_2_3.values
    t=valeurs[:,0]
    Z=valeurs[:,1]
    Ct=np.cumsum(Z)
    return t,Z, Ct
Guinea_set=pd.read_csv("data/ebola_cases_guinea.dat",sep="\s+")
Liberia_set=pd.read_csv("data/ebola_cases_liberia.dat",sep="\s+")
Sierra_set=pd.read_csv("data/ebola_cases_sierra_leone.dat",sep="\s+")
N_ebola=10**7

def non_linear_func(t, beta0, lam):
    """ 
    Function to find the optimize Betha and lambda for each dataset
        Input : t -> Time 
                beta0 -> First parameter that we are looking for 
                lam -> Second parameter that we are looking for
        Output : -> Cumulative number of outbreak
    """
    def f_ebola_fit(y, t_):
        """ 
        Function to know the behavior of the ebola disease with each parameter 
        """
        S, E, Z, R = y
        N = S + Z + E + R
        gamma = 1/7
        sigma = 1/9.7
        betha = B_ebola(beta0, t_, lam)
        dSdt = -betha * S * Z / N
        dEdt = betha * S * Z / N - (sigma * E)
        dZdt = (sigma * E) - gamma * Z
        dRdt = gamma * Z
        return np.array([dSdt, dEdt, dZdt, dRdt])

    U0_ebola = np.array([N_ebola-1, 0, 1, 0])
    sol=odeint(f_ebola_fit,U0_ebola,t)
    Z_model = sol[:, 2]
    return np.cumsum(Z_model)
# Function Created or changed for the Project_4 
def neural_network(t,Z):
    X_data = t.reshape(-1, 1).astype(np.float32)
    y_data = Z.reshape(-1, 1).astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)
    split_point = int(len(t) * 0.7)
    #Train
    X_train = X_scaled[:split_point]
    y_train = y_data[:split_point]
    # Test
    X_test = X_scaled[split_point:]
    y_test = y_data[split_point:]
    print(f"\nTrain size: {len(X_train)}   Test size: {len(X_test)}")
    model = Sequential(
        [
            Dense(units=128,activation='relu',input_shape=(1,)),
            Dense(units=64,activation='relu'),
            Dense(units=32, activation='relu'),
            Dense(1)
        ]
    )
    model.compile(optimizer='adam',loss='mse',metrics=['mae'])
    print("Begining of training ...")
    model.fit(X_train,y_train,
              epochs=100,
              validation_data=(X_test,y_test),
              verbose=0)
    Z_pred_nn = model.predict(X_scaled)
    return Z_pred_nn
def polynomial_regression(t,Z,order=2):
    X=t.reshape(-1,1)
    poly = PolynomialFeatures(order,include_bias=False)
    X_poly = poly.fit_transform(X)
    model_poly = LinearRegression()
    model_poly.fit(X_poly, Z)
    y_pred = model_poly.predict(X_poly)
    return y_pred
def lstm(dataset=""):
    """
    Function to apply an LSTM on the dataset
    Input -> dataset: Name of the dataset that we want to use,the window parameter of the lstm are changing 
                    because the dataset are not the same size so I use different parametrization for each
    """
    if dataset=="Guinea":
        CSV_PATH = "data/ebola_cases_guinea.dat" 
        WINDOW_SIZE  = 15
        EPOCHS       = 75 # réduire ça cest bien 
        BATCH_SIZE   = 32 # Tester avec batch size
    elif dataset=="Sierra Leone":
        CSV_PATH = "data/ebola_cases_sierra_leone.dat" 
        WINDOW_SIZE  = 10
        EPOCHS       = 150 # réduire ça cest bien 
        BATCH_SIZE   = 16 # Tester avec batch size
    elif dataset=="Liberia":
        CSV_PATH = "data/ebola_cases_liberia.dat" 
        WINDOW_SIZE  = 10
        EPOCHS       = 150 # réduire ça cest bien 
        BATCH_SIZE   = 16 # Tester avec batch size
    else : 
        print("Wrong dataset name ")
        return
    df= pd.read_csv(CSV_PATH,sep='\s+')
    df.drop('Days', axis=1, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    num_cols = [c for c in df.columns if c != "Date"]
    df[num_cols] = df[num_cols].replace({",": ""}, regex=True).astype(float)
    test_size = max(WINDOW_SIZE + 1, int(len(df) * 0.2))  # fallback to last 20%
    train_df = df.iloc[:-test_size].copy()
    test_df  = df.iloc[-test_size:].copy()
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    scaler = MinMaxScaler()
    scaler.fit(train_df[["NumOutbreaks"]])
    train_scaled = scaler.transform(train_df[["NumOutbreaks"]])
    test_scaled  = scaler.transform(df[["NumOutbreaks"]].iloc[len(train_df) - WINDOW_SIZE:])
    X_train, y_train = [], []
    for i in range(WINDOW_SIZE, len(train_scaled)):
        X_train.append(train_scaled[i - WINDOW_SIZE:i, 0])
        y_train.append(train_scaled[i, 0])
    X_test, y_test = [], []
    for i in range(WINDOW_SIZE, len(test_scaled)):
        X_test.append(test_scaled[i - WINDOW_SIZE:i, 0])
        y_test.append(test_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test   = np.array(X_test), np.array(y_test)
    # Reshape to fit LSTM input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    y_train = y_train.reshape(-1, 1)
    y_test  = y_test.reshape(-1, 1)
    #Debug print
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)
    # Build
    input_layer = Input(shape=(WINDOW_SIZE, 1))
    x = LSTM(128, return_sequences=True)(input_layer)
    x = Dropout(0.2)(x) # Test avec 0.1 au lieux de 0.2
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    # x = Dense(32, activation="relu")(x)
    output_layer = Dense(1)(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.summary()
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )
    y_pred_scaled = model.predict(X_test)
    y_true = scaler.inverse_transform(y_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    # Use the Meanabsolute error to have an idea of the precision of the model
    mae  = mean_absolute_error(y_true, y_pred)
    print("\n=== Test Results ===")
    print(f"MAE: ${mae:.2f}")
    # Plot of the spliting between train and test and the LSTM prediciton
    plt.figure(figsize=(15, 6), dpi=150)
    plt.plot(train_df["Date"], train_df["NumOutbreaks"], color="black", lw=2, label="Training set")
    plt.plot(test_df["Date"], test_df["NumOutbreaks"], color="blue", lw=2, label="Test set")
    plt.plot(test_df["Date"],y_pred,color="cyan")
    plt.gca().set_facecolor("white")
    plt.title("Ebola outbreak Training for "+ dataset, fontsize=15, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("NumOutbreaks (People)", fontsize=12)
    plt.legend(loc="upper left", fontsize=12)
    plt.grid(color="gray", alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_dataset(dataset="",linear=False,odeint=False,polynomial=False,order_polynomial=2,nn=False):
    """ 
    A function to avoid code repetition for the 3 datasets
    Input : dataset : Name of the dataset to use
            linear : Boolean parameter to allow the plot of the linear Regression
            odeint : Boolean parameter to allow the plot of the Optimization regression
            polynomial : Boolean parameter to allow the plot of the Polynomial regression
            order_polynomial : The order of the polynomial regression
            nn : Boolean parameter to allow the plot of the Neural Network prediction
    Output : A plot with incidence and cumulative incidence and the curve wished
    """
    t,Z,Ct=read_data(dataset=dataset)
    fig, ax1 = plt.subplots(figsize=(10,6))
    if linear :
        X = t.reshape(-1, 1)  # sklearn expects a 2D array for the features
        model = LinearRegression()
        model.fit(X, Z)
        y_pred = model.predict(X)
        ax1.plot(t, y_pred, color='darkgreen', linestyle='--', label='Linear Regression')
    if odeint : 
        y_odeint=calculate_precise(name=dataset)
        ax1.plot(t, y_odeint, label='Modèle Z(t)')
    if polynomial :
        y_polynomial= polynomial_regression(t,Z,order=order_polynomial)
        ax1.plot(t,y_polynomial,color='brown',label="Polynomial Regression")
    if nn :
        Z_nn=neural_network(t,Z)
        ax1.plot(t,Z_nn,color='cyan',label="Neural Network")
    ax1.scatter(t, Z, color="red", marker="o", label="Incidence (observed)")
    ax1.set_xlabel('Day since first outbreak (days)')
    ax1.set_ylabel('Number of outbreak ')
    ax1.tick_params(axis="y")
    ax2 = ax1.twinx()
    ax2.scatter(t, Ct, color="black", marker="s", label="Cumulative outbreaks")
    ax2.set_xlabel('Day since first outbreak (days)')
    ax2.set_ylabel('Cumulative number of outbreak ')
    plt.grid(axis="both")
    plt.title("Ebola outbreaks in "+dataset)
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='upper left')
    plt.show()
