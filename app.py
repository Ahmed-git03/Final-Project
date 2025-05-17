import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import pickle
import base64
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, classification_report, confusion_matrix
from io import BytesIO

# --- THEME SETUP ---
THEMES = {
    "Zombie": """
        body {
            background-color: #121212;
            color: #39ff14;
            font-family: 'Creepster', cursive, monospace;
        }
        .stButton>button {
            background-color: #39ff14;
            color: #121212;
            font-weight: bold;
            border-radius: 5px;
        }
        .stSlider>div>div>input[type=range]::-webkit-slider-thumb {
            background-color: #39ff14;
        }
        .stSelectbox>div>div>div {
            background-color: #121212;
            color: #39ff14;
        }
    """,
    "Futuristic": """
        body {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: #00f0ff;
            font-family: 'Orbitron', sans-serif;
        }
        .stButton>button {
            background-color: #00f0ff;
            color: #002f34;
            font-weight: 600;
            border-radius: 4px;
            box-shadow: 0 0 10px #00f0ff;
        }
        .stSlider>div>div>input[type=range]::-webkit-slider-thumb {
            background-color: #00f0ff;
        }
        .stSelectbox>div>div>div {
            background-color: #203a43;
            color: #00f0ff;
        }
    """,
    "Game of Thrones": """
        body {
            background-color: #1c1c1c;
            color: #c0b283;
            font-family: 'Cinzel', serif;
        }
        .stButton>button {
            background-color: #c0b283;
            color: #1c1c1c;
            font-weight: 700;
            border-radius: 0;
            border: 2px solid #c0b283;
            text-transform: uppercase;
        }
        .stSlider>div>div>input[type=range]::-webkit-slider-thumb {
            background-color: #c0b283;
        }
        .stSelectbox>div>div>div {
            background-color: #1c1c1c;
            color: #c0b283;
            border: 1px solid #c0b283;
        }
    """,
    "Gaming": """
        body {
            background: #0f0f0f;
            color: #ff0054;
            font-family: 'Press Start 2P', cursive, monospace;
            text-shadow: 0 0 8px #ff0054;
        }
        .stButton>button {
            background-color: #ff0054;
            color: #0f0f0f;
            font-weight: 700;
            border-radius: 12px;
            box-shadow: 0 0 12px #ff0054;
        }
        .stSlider>div>div>input[type=range]::-webkit-slider-thumb {
            background-color: #ff0054;
        }
        .stSelectbox>div>div>div {
            background-color: #1a1a1a;
            color: #ff0054;
            border: 1px solid #ff0054;
        } 
    """
}

def apply_theme(theme_name):
    css = THEMES.get(theme_name, "")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    # Load Google Fonts for all themes
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Creepster&family=Orbitron&family=Cinzel&family=Press+Start+2P&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

# --- PAGE CONFIG ---
st.set_page_config(page_title="📊 Financial ML App", layout="wide")
st.sidebar.title("🧭 App Navigation")
page = st.sidebar.radio("Go to", ["Welcome", "Upload Dataset", "Yahoo Finance", "EDA", "ML Models"])
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Visual Theme")
theme = st.sidebar.selectbox("Choose a Visual Theme", list(THEMES.keys()))
apply_theme(theme)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Quick Links")
st.sidebar.markdown(
    """
    - [💻 GitHub](https://github.com/Ahmed-git03/Final-Project)
    - [🔗 Linkedin](www.linkedin.com/in/ahmed-islam-625209318)
    """
)

st.sidebar.markdown("---")
st.sidebar.caption("© 2025 Financial ML App.")

# --- HELPER FUNCTIONS ---
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_data
def fetch_yfinance_data(ticker):
    return yf.download(ticker, period="6mo", interval="1d")

def generate_download_link(model, filename):
    buffer = BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    return f'<a href="data:file/output.pkl;base64,{b64}" download="{filename}">📥 Download Trained Model</a>'

# --- PAGES ---
if page == "Welcome":
    st.title("💰 Financial ML App")
    st.markdown("""
    Welcome to our Financial Machine Learning App!  
    Explore financial data, perform visual analysis, and apply ML models including:
    - 📈 Linear Regression
    - 🔍 Logistic Regression
    - 🔗 K-Means Clustering
    """)
    st.image(r"money-8763.gif", use_container_width=True)

elif page == "Upload Dataset":
    st.header("📁 Upload a Financial Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.session_state['df'] = df
        st.write("### Data Preview", df.head())

        if st.checkbox("📊 Show Data Summary"):
            st.write("### Description", df.describe())
            st.write("### Data Types", df.dtypes)

elif page == "Yahoo Finance":
    st.header("📈 Real-time Stock Data")
    
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)")
    today = datetime.date.today()   
    default_start = today - datetime.timedelta(days=180)

    start_date, end_date = st.date_input(
        "Select Date Range",
        value=(default_start, today),
        min_value=datetime.date(2000, 1, 1),
        max_value=today
    )
    interval = st.selectbox("Select Interval", ['1d', '1wk', '1mo'])

    if st.button("Fetch Data") and ticker and start_date < end_date:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

        # Flatten MultiIndex columns if they exist
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(filter(None, col)).strip() for col in data.columns]
            # Map back standard column names if needed
            for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                matches = [c for c in data.columns if col in c]
                if matches:
                    data[col] = data[matches[0]]

        if not data.empty:
            st.write(f"### Price Data for {ticker.upper()} ({start_date} → {end_date}, Interval: {interval})", data.tail())

            # Closing price slider
            min_close = float(data['Close'].min())
            max_close = float(data['Close'].max())
            price_range = st.slider(
                "Filter by Closing Price",
                min_value=round(min_close, 2),
                max_value=round(max_close, 2),
                value=(round(min_close, 2), round(max_close, 2))
            )

            # Filter data by closing price range
            data_filtered = data[(data['Close'] >= price_range[0]) & (data['Close'] <= price_range[1])]

            if data_filtered.empty:
                st.warning("No data available for the selected closing price range.")
            else:
                # Indicators on filtered data
                data_filtered['SMA_20'] = data_filtered['Close'].rolling(window=20).mean()
                data_filtered['SMA_50'] = data_filtered['Close'].rolling(window=50).mean()
                data_filtered['EMA_20'] = data_filtered['Close'].ewm(span=20, adjust=False).mean()
                data_filtered['EMA_50'] = data_filtered['Close'].ewm(span=50, adjust=False).mean()

                delta = data_filtered['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(14).mean()
                avg_loss = loss.rolling(14).mean()
                rs = avg_gain / avg_loss
                data_filtered['RSI'] = 100 - (100 / (1 + rs))

                sma = data_filtered['Close'].rolling(window=20).mean()
                std = data_filtered['Close'].rolling(window=20).std()
                data_filtered['Upper_BB'] = sma + 2 * std
                data_filtered['Lower_BB'] = sma - 2 * std

                # Williams %R indicator
                period = 14
                high_roll = data_filtered['High'].rolling(window=period).max()
                low_roll = data_filtered['Low'].rolling(window=period).min()
                data_filtered['Williams_%R'] = -100 * ((high_roll - data_filtered['Close']) / (high_roll - low_roll))

                # Calculate Fibonacci retracement levels
                max_price = data_filtered['High'].max()
                min_price = data_filtered['Low'].min()
                diff = max_price - min_price
                fib_levels = {
                    '0%': max_price,
                    '23.6%': max_price - 0.236 * diff,
                    '38.2%': max_price - 0.382 * diff,
                    '50%': max_price - 0.5 * diff,
                    '61.8%': max_price - 0.618 * diff,
                    '100%': min_price
                }

                # Candlestick chart with colored candles + Fibonacci lines
                import plotly.graph_objects as go
                fig = go.Figure()

                fig.add_trace(go.Candlestick(
                    x=data_filtered.index,
                    open=data_filtered['Open'], high=data_filtered['High'],
                    low=data_filtered['Low'], close=data_filtered['Close'],
                    increasing_line_color='green',
                    decreasing_line_color='red',
                    name="Candlestick"
                ))

                # Add SMAs and Bollinger Bands
                fig.add_trace(go.Scatter(x=data_filtered.index, y=data_filtered['SMA_20'], line=dict(color='blue'), name='SMA 20'))
                fig.add_trace(go.Scatter(x=data_filtered.index, y=data_filtered['SMA_50'], line=dict(color='red'), name='SMA 50'))
                fig.add_trace(go.Scatter(x=data_filtered.index, y=data_filtered['Upper_BB'], line=dict(color='green', dash='dot'), name='Upper BB'))
                fig.add_trace(go.Scatter(x=data_filtered.index, y=data_filtered['Lower_BB'], line=dict(color='green', dash='dot'), name='Lower BB'))

                # Add Fibonacci retracement horizontal lines
                for level_name, level_price in fib_levels.items():
                    fig.add_hline(
                        y=level_price,
                        line=dict(color='orange', width=1, dash='dash'),
                        annotation_text=f"Fib {level_name}: {level_price:.2f}",
                        annotation_position="top left",
                        opacity=0.6
                    )

                fig.update_layout(
                    title=f'{ticker.upper()} Candlestick with SMA, Bollinger Bands & Fibonacci',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    xaxis_rangeslider_visible=False,
                    template='plotly_dark',
                    height=600
                )
                st.plotly_chart(fig)

                # RSI Plot
                st.subheader("📉 RSI (Relative Strength Index)")
                if 'RSI' in data_filtered.columns:
                    import plotly.express as px
                    fig_rsi = px.line(data_filtered, x=data_filtered.index, y='RSI', title="RSI Indicator")
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    st.plotly_chart(fig_rsi)
                else:
                    st.warning("RSI column not found in the data.")

                # Williams %R Plot
                st.subheader("📊 Williams %R Indicator")
                if 'Williams_%R' in data_filtered.columns:
                    fig_willr = px.line(data_filtered, x=data_filtered.index, y='Williams_%R', title="Williams %R Indicator")
                    fig_willr.add_hline(y=-20, line_dash="dash", line_color="red")
                    fig_willr.add_hline(y=-80, line_dash="dash", line_color="green")
                    st.plotly_chart(fig_willr)
                else:
                    st.warning("Williams %R column not found in the data.")

                st.session_state['df'] = data_filtered.reset_index()
        else:
            st.warning("No data found. Try different dates or check the ticker symbol.")
    elif start_date >= end_date:
        st.error("Start date must be before end date.")

elif page == "EDA":
    st.header("📊 Exploratory Data Analysis")
    df = st.session_state.get('df')
    
    if df is None:
        st.warning("Upload or fetch data first.")
    else:
        vis_type = st.selectbox("Select Visualization", [
            "Histogram", 
            "Boxplot", 
            "Correlation Heatmap", 
            "Bar Plot (Categorical)", 
            "Count Plot (Categorical)"
        ])

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if vis_type in ["Histogram", "Boxplot"]:
            if not numeric_cols:
                st.warning("No numeric columns found in the dataset.")
            else:
                col = st.selectbox("Select Numeric Column", numeric_cols)
                
                # Summary statistics
                stats = df[col].describe()
                st.subheader(f"Summary Statistics for {col}")
                st.write(stats)

                # Simple insights
                st.markdown(f"**Insights:**")
                mean = stats['mean']
                median = df[col].median()
                std = stats['std']
                min_val = stats['min']
                max_val = stats['max']
                st.write(f"- Mean: {mean:.2f}")
                st.write(f"- Median: {median:.2f}")
                st.write(f"- Standard Deviation: {std:.2f}")
                st.write(f"- Range: {min_val:.2f} to {max_val:.2f}")
                st.write(f"- Skewness: {df[col].skew():.2f}")
                st.write(f"- Kurtosis: {df[col].kurtosis():.2f}")

                # Plot
                fig, ax = plt.subplots()
                if vis_type == "Histogram":
                    bins = st.slider("Number of bins", 5, 100, 30)
                    ax.hist(df[col].dropna(), bins=bins, color='orange', edgecolor='black')
                    ax.set_title(f'Histogram of {col}')
                elif vis_type == "Boxplot":
                    sns.boxplot(x=df[col], ax=ax)
                    ax.set_title(f'Boxplot of {col}')
                st.pyplot(fig)

        elif vis_type == "Correlation Heatmap":
            if not numeric_cols:
                st.warning("No numeric columns found for correlation heatmap.")
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)

        elif vis_type == "Bar Plot (Categorical)":
            if not categorical_cols:
                st.warning("No categorical columns found for bar plot.")
            else:
                col = st.selectbox("Select Categorical Column", categorical_cols)
                
                # Summary statistics for categorical
                st.subheader(f"Value Counts for {col}")
                counts = df[col].value_counts()
                st.write(counts)

                # Insights for categorical
                st.markdown(f"**Insights:**")
                st.write(f"- Number of Unique Categories: {df[col].nunique()}")
                st.write(f"- Most Frequent Category: {counts.idxmax()} ({counts.max()} occurrences)")

                # Plot
                fig, ax = plt.subplots()
                df[col].value_counts().plot(kind='bar', color='skyblue', ax=ax)
                ax.set_ylabel("Count")
                ax.set_title(f'Bar Plot of {col}')
                st.pyplot(fig)

        elif vis_type == "Count Plot (Categorical)":
            if not categorical_cols:
                st.warning("No categorical columns found for count plot.")
            else:
                col = st.selectbox("Select Categorical Column", categorical_cols)

                # Summary statistics for categorical
                st.subheader(f"Value Counts for {col}")
                counts = df[col].value_counts()
                st.write(counts)

                # Insights for categorical
                st.markdown(f"**Insights:**")
                st.write(f"- Number of Unique Categories: {df[col].nunique()}")
                st.write(f"- Most Frequent Category: {counts.idxmax()} ({counts.max()} occurrences)")

                # Plot
                fig, ax = plt.subplots()
                sns.countplot(y=col, data=df, order=counts.index, ax=ax)
                ax.set_title(f'Count Plot of {col}')
                st.pyplot(fig)

elif page == "ML Models":
    st.header("🧠 Machine Learning Models")
    df = st.session_state.get('df')

    if df is None:
        st.warning("Please upload or fetch data first.")
    else:
        model_type = st.selectbox("Select Model", ["Linear Regression", "Logistic Regression", "K-Means Clustering"])
        st.write("### Data Preview", df.head())
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if model_type == "Linear Regression":
            x_col = st.selectbox("Feature (X)", numeric_cols)
            y_col = st.selectbox("Target (Y)", numeric_cols)

            X = df[[x_col]].dropna()
            y = df[y_col].dropna()
            y = y[:len(X)]

            model = LinearRegression().fit(X, y)
            pred = model.predict(X)

            st.subheader("📈 Regression Fit with Confidence Interval")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X[x_col], y=y, mode='markers', name='Actual'))
            fig.add_trace(go.Scatter(x=X[x_col], y=pred, mode='lines', name='Prediction', line=dict(color='red')))

            residuals = y - pred
            std_error = np.std(residuals)
            ci_upper = pred + 1.96 * std_error
            ci_lower = pred - 1.96 * std_error
            fig.add_trace(go.Scatter(x=X[x_col], y=ci_upper, line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=X[x_col], y=ci_lower, line=dict(width=0), fill='tonexty', fillcolor='rgba(255,0,0,0.2)', name='95% CI'))

            fig.update_layout(title="Linear Regression Fit with 95% Confidence Interval")
            st.plotly_chart(fig)

            st.subheader("📉 Residual Plot")
            fig_res = px.scatter(x=pred, y=residuals, labels={'x': "Predicted", 'y': "Residuals"})
            fig_res.update_layout(title="Residual Plot")
            st.plotly_chart(fig_res)

            st.subheader("📊 Metrics")
            st.write(f"**MSE:** {mean_squared_error(y, pred):.2f}")
            st.write(f"**R² Score:** {r2_score(y, pred):.2f}")
            st.markdown(generate_download_link(model, "linear_model.pkl"), unsafe_allow_html=True)

        elif model_type == "Logistic Regression":
            y_col = st.selectbox("Target (Binary Y)", numeric_cols)
            x_cols = st.multiselect("Features (X)", [col for col in numeric_cols if col != y_col])

            if x_cols:
                X = df[x_cols].dropna()
                y = (df[y_col].dropna() > df[y_col].median()).astype(int)[:len(X)]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)[:, 1]

                # Confusion Matrix
                st.subheader("📊 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                labels=dict(x="Predicted", y="Actual"))
                st.plotly_chart(fig_cm)

                # ROC Curve with AUC
                st.subheader("📈 ROC Curve")
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC={roc_auc:.2f})'))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                            line=dict(dash='dash'), name='Random'))
                fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
                st.plotly_chart(fig_roc)

                # Precision-Recall Curve
                st.subheader("📈 Precision-Recall Curve")
                from sklearn.metrics import precision_recall_curve, average_precision_score
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                avg_precision = average_precision_score(y_test, y_prob)

                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines',
                                            name=f'Precision-Recall Curve (AP={avg_precision:.2f})'))
                fig_pr.update_layout(title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision')
                st.plotly_chart(fig_pr)

                # Classification Report
                st.subheader("📄 Classification Report")
                st.text(classification_report(y_test, y_pred))

                # Download link for the model
                st.markdown(generate_download_link(clf, "logistic_model.pkl"), unsafe_allow_html=True)

        elif model_type == "K-Means Clustering":
            def flatten_columns(df):
                df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
                return df

            k = st.slider("Number of Clusters (K)", 2, 10, 3)
            df = flatten_columns(df)
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            x_cols = st.multiselect("Features for Clustering", numeric_cols)

            if len(x_cols) >= 2:
                X = df[x_cols].dropna()

                # Optional: Standardize features for better clustering
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X_scaled)
                df['Cluster'] = np.nan
                df.loc[X.index, 'Cluster'] = km.labels_

                st.write("### Cluster Centroids (scaled)")
                centroids = pd.DataFrame(km.cluster_centers_, columns=x_cols)
                st.dataframe(centroids)

                st.bar_chart(pd.Series(km.labels_).value_counts().sort_index())

                if len(x_cols) == 2:
                    fig = px.scatter(df.loc[X.index], x=x_cols[0], y=x_cols[1],
                                    color=df.loc[X.index]['Cluster'].astype(str),
                                    title="K-Means Clustering (2D)")
                elif len(x_cols) >= 3:
                    fig = px.scatter_3d(df.loc[X.index], x=x_cols[0], y=x_cols[1], z=x_cols[2],
                                        color=df.loc[X.index]['Cluster'].astype(str),
                                        title="K-Means Clustering (3D)")
                st.plotly_chart(fig)

                # Elbow plot for inertia
                st.subheader("📉 Elbow Plot")
                inertias = []
                k_range = range(1, 11)
                for i in k_range:
                    km_temp = KMeans(n_clusters=i, n_init="auto", random_state=42).fit(X_scaled)
                    inertias.append(km_temp.inertia_)

                fig_elbow = go.Figure()
                fig_elbow.add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers'))
                fig_elbow.update_layout(title='Elbow Plot for Optimal K',
                                        xaxis_title='Number of Clusters (K)',
                                        yaxis_title='Inertia')
                st.plotly_chart(fig_elbow)

                st.markdown(generate_download_link(km, "kmeans_model.pkl"), unsafe_allow_html=True)
