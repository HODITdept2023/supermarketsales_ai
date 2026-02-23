import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# ================= PAGE CONFIG ================= #

st.set_page_config(
    page_title="Retail AI Dashboard",
    layout="wide",
    page_icon="📊"
)

# ================= LOGIN ================= #

ADMIN_USER = "admin"
ADMIN_PASS = "csdavanthi2026"

if "login" not in st.session_state:
    st.session_state["login"] = False

def login():
    st.title("🔐 Retail AI Analytics System")
    st.subheader("Login to Continue")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == ADMIN_USER and pwd == ADMIN_PASS:
            st.session_state["login"] = True
            st.success("Login Successful")
            st.rerun()
        else:
            st.error("Invalid Credentials")

if not st.session_state["login"]:
    login()
    st.stop()


# ================= SIDEBAR ================= #

st.sidebar.title("📌 Navigation")

menu = st.sidebar.radio(
    "Go To",
    ["Dashboard", "Data Input", "Basket Recommendation", "Logout"]
)

if "data" not in st.session_state:
    st.session_state["data"] = None


# ================= DASHBOARD ================= #

if menu == "Dashboard":

    st.title("📊 AI Sales Analytics Dashboard")

    if st.session_state["data"] is None:
        st.warning("Please enter data first.")
        st.stop()

    data = st.session_state["data"]

    # ---------------- KPI METRICS ---------------- #

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Sales", int(sum(data["Sales"])))
    col2.metric("Avg Profit", round(np.mean(data["Profit"]), 2))
    col3.metric("Total Customers", int(sum(data["Customers"])))
    col4.metric("Avg Discount", round(np.mean(data["Discount"]), 2))


    # ---------------- PIE CHART ---------------- #

    st.subheader("📌 Monthly Sales Distribution")

    months = [f"M{i}" for i in range(1,13)]

    fig_pie = px.pie(
        names=months,
        values=data["Sales"],
        title="Sales Contribution per Month",
        hole=0.4
    )

    st.plotly_chart(fig_pie, use_container_width=True)


    # ---------------- LSTM FORECAST ---------------- #

    df = pd.DataFrame(data)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []

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


    # ---------------- FORECAST GRAPH ---------------- #

    st.subheader("📈 Sales Forecast (Actual vs Predicted)")

    fig_line = go.Figure()

    fig_line.add_trace(go.Scatter(
        x=list(range(1,13)),
        y=data["Sales"],
        mode='lines+markers',
        name="Actual Sales"
    ))

    fig_line.add_trace(go.Scatter(
        x=list(range(13,25)),
        y=predicted,
        mode='lines+markers',
        name="Predicted Sales"
    ))

    fig_line.update_layout(
        xaxis_title="Month",
        yaxis_title="Sales",
        template="plotly_white"
    )

    st.plotly_chart(fig_line, use_container_width=True)


# ================= DATA INPUT ================= #

elif menu == "Data Input":

    st.title("📝 Enter 12-Month Data")

    def get_values(label):
        text = st.text_input(label)
        if text.strip() == "":
            return []
        try:
            return list(map(float, text.split(",")))
        except:
            st.error(f"Invalid input in {label}")
            return []

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

        fields = [sales,profit,customers,discount,rice,oil,sugar,milk,soap]
        names = ["Sales","Profit","Customers","Discount",
                 "Rice","Oil","Sugar","Milk","Soap"]

        for n,f in zip(names,fields):
            if len(f)!=12:
                st.error(f"{n} must have 12 values")
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


# ================= BASKET ================= #

elif menu == "Basket Recommendation":

    st.title("🛒 Basket Intelligence")

    if st.session_state["data"] is None:
        st.warning("Enter data first.")
        st.stop()

    data = st.session_state["data"]

    products = ["Rice","Oil","Sugar","Milk","Soap"]

    movement = {
        p: data[p].iloc[0] - data[p].iloc[-1]
        for p in products
    }

    df_move = pd.DataFrame({
        "Product": list(movement.keys()),
        "Sold Units": list(movement.values())
    })

    st.subheader("📦 Fast Moving Products")

    fig_bar = px.bar(
        df_move,
        x="Product",
        y="Sold Units",
        color="Sold Units",
        title="Stock Movement Analysis"
    )

    st.plotly_chart(fig_bar, use_container_width=True)


# ================= LOGOUT ================= #

elif menu == "Logout":
    st.session_state["login"] = False
    st.rerun()