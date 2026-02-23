import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ---------------- LOGIN ---------------- #

ADMIN_USER = "admin"
ADMIN_PASS = "csdavanthi2026"


def login():

    st.title("INV Technologies - Retail AI Dashboard")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):

        if user == ADMIN_USER and pwd == ADMIN_PASS:
            st.session_state["login"] = True
            st.success("Login Successful")
            st.rerun()
        else:
            st.error("Invalid Credentials")


if "login" not in st.session_state:
    st.session_state["login"] = False


if not st.session_state["login"]:
    login()
    st.stop()


# ---------------- SIDEBAR ---------------- #

st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "Data Input", "Basket Recommendation", "Logout"]
)


# ---------------- DATA STORAGE ---------------- #

if "data" not in st.session_state:
    st.session_state["data"] = None

# ---------------- DASHBOARD ---------------- #

if menu == "Dashboard":

    st.title("📊 Sales Analytics Dashboard")

    if st.session_state["data"] is None:
        st.warning("Please enter data first in Data Input section.")
        st.stop()

    data = st.session_state["data"]

    months = [f"M{i}" for i in range(1,13)]
    sales = data["Sales"]

    # PIE CHART
    st.subheader("Monthly Sales Distribution")

    fig1, ax1 = plt.subplots()
    ax1.pie(sales, labels=months, autopct='%1.1f%%')
    st.pyplot(fig1)

    # LSTM MODEL
    df = pd.DataFrame(data)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X = []
    y = []

    for i in range(8):
        X.append(scaled[i:i+4])
        y.append(scaled[i+4][0])

    X = np.array(X)
    y = np.array(y)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(4,9)),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=100, verbose=0)

    # PREDICTION
    last = scaled[-4:]
    future = []

    current = last.reshape(1,4,9)

    for i in range(12):

        pred = model.predict(current, verbose=0)
        future.append(pred[0][0])

        next_row = current[0,-1].copy()
        next_row[0] = pred[0][0]

        next_row = next_row.reshape(1,1,9)

        current = np.concatenate((current[:,1:,:], next_row), axis=1)

    dummy = np.zeros((12,9))
    dummy[:,0] = future

    predicted = scaler.inverse_transform(dummy)[:,0]

    # LINE GRAPH
    st.subheader("Actual vs Predicted Sales")

    fig2, ax2 = plt.subplots()

    ax2.plot(range(1,13), sales, label="Actual Sales", marker="o")
    ax2.plot(range(13,25), predicted, label="Predicted Sales", marker="o")

    ax2.set_xlabel("Month")
    ax2.set_ylabel("Sales")
    ax2.legend()

    st.pyplot(fig2)


# ---------------- DATA INPUT ---------------- #

elif menu == "Data Input":

    st.title("📝 Enter Monthly Data")

    def get_values(msg):
        return list(map(float, st.text_input(msg).split(",")))


    st.info("Enter 12 values separated by commas")

    sales = get_values("Sales")
    profit = get_values("Profit")
    customers = get_values("Customers")
    discount = get_values("Discount")

    rice = get_values("Rice Stock")
    oil = get_values("Oil Stock")
    sugar = get_values("Sugar Stock")
    milk = get_values("Milk Stock")
    soap = get_values("Soap Stock")


    if st.button("Save Data"):

        columns = [sales,profit,customers,discount,
                   rice,oil,sugar,milk,soap]

        for col in columns:
            if len(col)!=12:
                st.error("Each field must have exactly 12 values")
                st.stop()

        df = pd.DataFrame({
            "Sales":sales,
            "Profit":profit,
            "Customers":customers,
            "Discount":discount,
            "Rice":rice,
            "Oil":oil,
            "Sugar":sugar,
            "Milk":milk,
            "Soap":soap
        })

        st.session_state["data"] = df

        st.success("Data Saved Successfully")

        st.dataframe(df)


# ---------------- BASKET RECOMMENDATION ---------------- #

elif menu == "Basket Recommendation":

    st.title("🛒 Basket Recommendation System")

    if st.session_state["data"] is None:
        st.warning("Enter data first.")
        st.stop()

    data = st.session_state["data"]

    products = ["Rice","Oil","Sugar","Milk","Soap"]

    movement = {}

    for p in products:
        movement[p] = data[p].iloc[0] - data[p].iloc[-1]

    st.subheader("Fast Moving Products")

    sorted_move = sorted(movement.items(),
                         key=lambda x:x[1],
                         reverse=True)

    for item in sorted_move:
        st.write(f"{item[0]} → Sold Units: {int(item[1])}")


    st.subheader("Recommended Combos")

    corr = data[products].corr()

    found = False

    for i in range(len(products)):
        for j in range(i+1,len(products)):

            if corr.iloc[i,j] > 0.6:
                st.success(f"{products[i]} + {products[j]}")
                found = True

    if not found:
        st.info("No strong product combinations detected.")


# ---------------- LOGOUT ---------------- #

elif menu == "Logout":

    st.session_state["login"] = False
    st.success("Logged out successfully")
    st.rerun()