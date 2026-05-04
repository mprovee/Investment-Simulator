#Installed streamlit, yfinance, gspread, Credentials, numpy, and matplotlib for purposes listed below
import streamlit as st #Web app UI (replaces tkinter)
import yfinance as yf #Uses real stock data and history to make simulation "predictions"
import gspread #To connect to and record data in Google Sheets
from google.oauth2.service_account import Credentials #To log into Google Sheets
import numpy as np #To run simulation math
import matplotlib.pyplot as plt #To build the results chart
from datetime import datetime, timedelta #To record and filter timestamps of simulations
import os, json #To read credentials from environment variable

#Constant Simulation Parameters (Written by Group)
app_title = "Investment Simulator"
starting_balance = 25000
max_stocks = 10
simulation_runs = 1000

#Google Sheets OAuth Connection (Written by Group with roughly 50% AI assistance)
scopes = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

# Load credentials from environment variable instead of a local file.
# On the host machine, set GOOGLE_CREDENTIALS to the full contents of credentials.json.
# On Streamlit Community Cloud, add it under Settings > Secrets as GOOGLE_CREDENTIALS.
@st.cache_resource
def get_sheets():
    _creds_json = os.environ.get("GOOGLE_CREDENTIALS")
    if not _creds_json:
        st.error("Missing GOOGLE_CREDENTIALS environment variable. Set it to the full contents of credentials.json before running.")
        st.stop()
    creds = Credentials.from_service_account_info(json.loads(_creds_json), scopes=scopes)
    client = gspread.authorize(creds)
    sheet = client.open("InvestmentSimulator")
    users_sheet = sheet.worksheet("Users")
    simulations_sheet = sheet.worksheet("Simulations")
    return users_sheet, simulations_sheet

users_sheet, simulations_sheet = get_sheets()

#Session state initialization — replaces global variables (Written by Group)
if "current_user" not in st.session_state:
    st.session_state.current_user = None #This means nobody is logged in
if "stocks_list" not in st.session_state:
    st.session_state.stocks_list = []
if "screen" not in st.session_state:
    st.session_state.screen = "welcome"
if "sim_years" not in st.session_state:
    st.session_state.sim_years = None
if "sim_results" not in st.session_state:
    st.session_state.sim_results = None

#Helper to navigate between screens (replaces show_X_screen calls)
def go_to(screen):
    st.session_state.screen = screen

#Stock Symbol Validation (Necessary ticker information sourced from https://www.geeksforgeeks.org/python/getting-stock-data-using-yfinance-in-python/)
def validate_ticker(ticker_symbol): #Check if ticker exists in yfinance and return company name if found
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        name = info.get("longName") or info.get("shortName")
        if name:
            return True, name
        else:
            return False, None
    except Exception:
        return False, None

#Simulation Engine with Random Events (Written by Group)
def run_monte_carlo(years):
    trading_days = years * 252
    portfolio_results = np.zeros((simulation_runs, trading_days))

    for stock in st.session_state.stocks_list:
        ticker = stock["ticker"]
        amount = stock["amount"]

        #Download 5 years of historical data (Information sourced from https://pypi.org/project/yfinance/)
        data = yf.download(ticker, period="5y", progress=False)
        #Calculate daily log returns (Information sourced from https://numpy.org/doc/stable/reference/generated/numpy.log.html)
        daily_returns = data["Close"].pct_change().dropna()

        #Calculate mean and std for normal distribution (Information sourced from previous numpy citations)
        mean_return = float(daily_returns.mean().iloc[0])
        std_return = float(daily_returns.std().iloc[0])

        #Run 1000 simulations for this stock (Used above Numpy Source for this section)
        stock_sims = np.zeros((simulation_runs, trading_days))
        for i in range(simulation_runs):
            #Random daily returns based on historical mean and volatility
            rand_returns = np.random.normal(mean_return, std_return, trading_days)
            #Compound the returns over time starting from allocated amount
            stock_sims[i] = amount * np.cumprod(1 + rand_returns)

        #Add stock results into total portfolio
        portfolio_results += stock_sims

    return portfolio_results

#Build Chart (Written by Group using https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html)
def build_chart(results, years, avg_final, min_final, max_final, p10, p90):
    fig, ax = plt.subplots(figsize=(9, 4.5))

    trading_days = years * 252
    x = np.linspace(0, years, trading_days)

    #Plot sample paths as faint background lines (This section from https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html)
    for i in range(0, 200, 5):
        ax.plot(x, results[i], color="lightblue", alpha=0.2, linewidth=0.5)

    #Calculate summary lines (Written by Group with help of existing numpy citations)
    mean_path = np.mean(results, axis=0)
    min_path = np.min(results, axis=0)
    max_path = np.max(results, axis=0)
    p10_path = np.percentile(results, 10, axis=0)
    p90_path = np.percentile(results, 90, axis=0)

    #Plot key lines (Written by Group)
    ax.plot(x, mean_path, color="blue", linewidth=2, label="Average")
    ax.plot(x, max_path, color="green", linewidth=1.5, linestyle="--", label="Best Case")
    ax.plot(x, min_path, color="red", linewidth=1.5, linestyle="--", label="Worst Case")
    ax.plot(x, p90_path, color="orange", linewidth=1.5, linestyle=":", label="Top 10%")
    ax.plot(x, p10_path, color="pink", linewidth=1.5, linestyle=":", label="Bottom 10%")

    #Shade likely range (Written by Group)
    ax.fill_between(x, p10_path, p90_path, alpha=0.1, color="blue")

    #Starting balance line (Following 2 use information from https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axhline.html)
    ax.axhline(y=starting_balance, color="black", linewidth=1,
               linestyle="--", alpha=0.5, label="Starting Balance")
    #Following 2 lines written by Group using previous project experience
    ax.set_xlabel("Years", fontsize=14)
    ax.set_ylabel("Portfolio Value ($)", fontsize=14)

    #Title with tickers and stats (First 2 lines of the section use info from https://www.w3schools.com/python/ref_string_join.asp, rest is written by Group)
    tickers = ", ".join([s["ticker"] for s in st.session_state.stocks_list])
    avg_return_pct = ((avg_final - starting_balance) / starting_balance) * 100
    ax.set_title(
        f"Monte Carlo Simulation — {tickers} — {years} Year(s)\n"
        f"Worst: ${min_final:,.0f}  |  Bottom 10%: ${p10:,.0f}  |  "
        f"Avg: ${avg_final:,.0f}  |  Top 10%: ${p90:,.0f}  |  "
        f"Best: ${max_final:,.0f}  |  Avg Return: {avg_return_pct:.1f}%",
        fontsize=12)
    #Following 4 lines are 50/25/25 breakdown between Group, https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.FuncFormatter, and https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.tight_layout.html
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"${val:,.0f}"))
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig

#Save simulation to leaderboard and show placement (Written by Group with roughly 40% AI assistance)
def save_to_leaderboard(years, avg_final, min_final, max_final):
    date_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    #Write simulation row to simulations sheet (Written by Group)
    simulations_sheet.append_row([
        "", st.session_state.current_user, date_run, starting_balance,
        years, round(avg_final, 2), round(min_final, 2), round(max_final, 2)
    ])

    #Update total_simulations count for user (Information sourced from https://docs.gspread.org/en/latest/user-guide.html)
    records = users_sheet.get_all_records()
    for i, row in enumerate(records, 2): #Row 2 is first data row since row 1 is headers
        if row["username"] == st.session_state.current_user:
            users_sheet.update_cell(i, 4, int(row["total_simulations"]) + 1)
            break

    #Pull all scores and calculate placement (Written by Group)
    all_records = simulations_sheet.get_all_records()
    all_maxes = [float(r.get("final_max", 0)) for r in all_records]
    placement = sum(1 for score in all_maxes if score > max_final) + 1

    st.success(
        f"Simulation saved! Your simulation ranked #{placement} out of {len(all_maxes)} all-time by best case value!\n\n"
        f"Best Case: ${max_final:,.0f} | Average: ${avg_final:,.0f} | Worst Case: ${min_final:,.0f}"
    )

# ─────────────────────────────────────────────
# SCREENS
# ─────────────────────────────────────────────

#Show welcome screen when app runs (Written by Group with roughly 25% AI assistance on leaderboard table section)
def show_welcome_screen():
    st.title("Investment Simulator")
    st.markdown("*Select and simulate your own portfolio. Can you beat Wall Street?*")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Login", use_container_width=True):
            go_to("login")
    with col2:
        if st.button("Create Account", use_container_width=True):
            go_to("create_account")
    with col3:
        if st.button("Continue as Guest", use_container_width=True):
            st.info("Guest Mode: Simulations cannot be saved to the leaderboard.")
            st.session_state.current_user = None
            go_to("stock_input")

    #Leaderboard section (Written by Group with roughly 25% AI assistance)
    st.markdown("---")
    st.subheader("Leaderboard")

    #Fetch all simulation records once for both tables (Information sourced from https://docs.gspread.org/en/latest/user-guide.html)
    try:
        all_records = simulations_sheet.get_all_records()
    except Exception:
        all_records = []

    #Filter to this week for weekly table (Written by Group)
    one_week_ago = datetime.now() - timedelta(days=7)
    weekly_records = []
    for r in all_records:
        try:
            if datetime.strptime(r["date_run"], "%Y-%m-%d %H:%M:%S") >= one_week_ago:
                weekly_records.append(r)
        except Exception:
            pass

    #Sort both lists by final_max descending and take top 10 (Written by Group)
    top_alltime = sorted(all_records, key=lambda r: float(r.get("final_max", 0)), reverse=True)[:10]
    top_weekly = sorted(weekly_records, key=lambda r: float(r.get("final_max", 0)), reverse=True)[:10]

    #Helper to build a leaderboard table (Written by Group)
    def make_table(title, rows):
        st.markdown(f"**{title}**")
        if rows:
            table_data = {
                "Rank": [f"#{i+1}" for i in range(len(rows))],
                "User": [r.get("username", "—") for r in rows],
                "Years": [r.get("years", "—") for r in rows],
                "Best Case": [f"${float(r.get('final_max', 0)):,.0f}" for r in rows]
            }
        else:
            table_data = {"Rank": ["—"], "User": ["No entries yet"], "Years": ["—"], "Best Case": ["—"]}
        st.table(table_data)

    col_l, col_r = st.columns(2)
    with col_l:
        make_table("Top 10 All-Time", top_alltime)
    with col_r:
        make_table("Top 10 This Week", top_weekly)

#Login Screen (Written by Group with information sourced from https://www.w3schools.com/python/python_mongodb_find.asp)
def show_login_screen():
    st.title("Login")

    #Input fields (Written by Group)
    uname = st.text_input("Username")
    pword = st.text_input("Password", type="password")

    #Check credentials against users sheet then redirect to allocation screen (Written by Group, roughly 25% AI assistance)
    if st.button("Login"):
        if not uname or not pword:
            st.error("Please enter both fields.")
        else:
            records = users_sheet.get_all_records()
            for row in records:
                if row["username"] == uname and row["password"] == pword:
                    st.session_state.current_user = uname
                    st.success(f"Welcome, {uname}!")
                    go_to("stock_input")
                    st.rerun()
            st.error("Invalid username or password.")

    if st.button("Back to Home"):
        go_to("welcome")

#Create Account Screen (Written by Group with information sourced from https://www.w3schools.com/python/python_mongodb_find.asp)
def show_create_account_screen():
    st.title("Create Account")

    #Input fields (Written by Group)
    uname = st.text_input("Username")
    pword = st.text_input("Password", type="password")

    #Check username not taken then write new row to sheet (Written by Group)
    if st.button("Create Account"):
        if not uname or not pword:
            st.error("Please enter both fields.")
        else:
            records = users_sheet.get_all_records()
            for row in records:
                if row["username"] == uname:
                    st.error("Username already taken.")
                    return
            created_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            users_sheet.append_row([uname, pword, created_date, 0])
            st.session_state.current_user = uname
            st.success(f"Account created! Welcome, {uname}!")
            go_to("stock_input")
            st.rerun()

    if st.button("Back to Home"):
        go_to("welcome")

# Stock Input & Allocation SCREEN (Written by Group, though second line of section uses information from https://www.w3schools.com/python/python_variables_global.asp)
def show_stock_input_screen():
    st.title("Build Your Portfolio")

    #Show login status so user knows if they are logged in or playing as guest (Written by Group)
    status_text = f"Logged in as: {st.session_state.current_user}" if st.session_state.current_user else "Playing as Guest"
    st.markdown(f"<span style='color:red'>{status_text}</span>", unsafe_allow_html=True)

    #Remaining Allocation Balance
    total_allocated = sum(s["amount"] for s in st.session_state.stocks_list)
    remaining = starting_balance - total_allocated
    st.markdown(f"**Remaining Balance: ${remaining:,.2f}**")

    #Input row for ticker and amount
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        ticker_input = st.text_input("Stock Ticker (e.g. AAPL)").strip().upper()
    with col2:
        amount_input = st.text_input("Amount ($)").strip()
    with col3:
        st.markdown("<br>", unsafe_allow_html=True) #Spacing to align button
        add_clicked = st.button("Add Stock")

    #Add stock logic (Written by Group, AI consulted for input handling)
    if add_clicked:
        if not ticker_input or not amount_input:
            st.error("Please enter both a ticker and an amount")
        else:
            try:
                amount = float(amount_input)
            except ValueError:
                st.error("Amount must be a number")
                amount = None

            if amount is not None:
                if amount <= 0:
                    st.error("Amount must be greater than zero")
                elif len(st.session_state.stocks_list) >= max_stocks:
                    st.error(f"Maximum {max_stocks} stocks allowed")
                elif amount > remaining:
                    st.error(f"Amount exceeds remaining balance of ${remaining:,.2f}")
                else:
                    #Check for duplicates (Written by Group)
                    if any(s["ticker"] == ticker_input for s in st.session_state.stocks_list):
                        st.error(f"{ticker_input} is already in your portfolio")
                    else:
                        #Validate ticker exists (Written by Group)
                        with st.spinner(f"Validating {ticker_input}..."):
                            valid, company_name = validate_ticker(ticker_input)
                        if not valid:
                            st.error(f"Ticker not found: {ticker_input}. Try a valid symbol like AAPL or TSLA")
                        else:
                            st.session_state.stocks_list.append({
                                "ticker": ticker_input,
                                "company": company_name,
                                "amount": amount
                            })
                            st.rerun()

    #Stock table (this section was sourced from https://www.pythontutorial.net/tkinter/tkinter-treeview/)
    if st.session_state.stocks_list:
        st.markdown("### Your Portfolio")
        table_data = {
            "Ticker": [s["ticker"] for s in st.session_state.stocks_list],
            "Company": [s["company"] for s in st.session_state.stocks_list],
            "Amount": [f"${s['amount']:,.2f}" for s in st.session_state.stocks_list],
            "Allocation %": [f"{(s['amount'] / starting_balance) * 100:.1f}%" for s in st.session_state.stocks_list]
        }
        st.table(table_data)

        #Remove stock option
        remove_ticker = st.selectbox("Remove a stock:", [""] + [s["ticker"] for s in st.session_state.stocks_list])
        if st.button("Remove Selected") and remove_ticker:
            st.session_state.stocks_list = [s for s in st.session_state.stocks_list if s["ticker"] != remove_ticker]
            st.rerun()

    #Simulation period entry that accepts any whole number from 1 to 50 (Written by Group)
    years = st.number_input("Simulation Period (1-50 years):", min_value=1, max_value=50, value=5, step=1)

    #Run simulation button (Written by Group)
    if st.button("Run Simulation", type="primary"):
        if not st.session_state.stocks_list:
            st.error("Please add at least one stock")
        else:
            #Checks to ensure user is okay with proceeding if not using the full amount (Written by Group)
            if remaining > 0:
                st.warning(f"You have ${remaining:,.2f} unallocated. Click Run Simulation again to continue anyway.")
                st.session_state._unallocated_warning = True
            if remaining == 0 or st.session_state.get("_unallocated_warning"):
                st.session_state._unallocated_warning = False
                st.session_state.sim_years = years
                go_to("results")
                st.rerun()

    if st.button("Back to Home"):
        st.session_state.stocks_list = []
        go_to("welcome")

#Results Screen (Written by Group)
def show_results_screen():
    years = st.session_state.sim_years
    st.title("Simulation Results")

    #Run simulation if not already cached in session state (Written by Group)
    if st.session_state.sim_results is None:
        with st.spinner("Simulating Results..."):
            st.session_state.sim_results = run_monte_carlo(years)

    results = st.session_state.sim_results

    #Calculate stats (Written mostly by group)
    final_values = results[:, -1] #Info from https://numpy.org/doc/stable/user/basics.indexing.html
    avg_final = np.mean(final_values)
    min_final = np.min(final_values)
    max_final = np.max(final_values)
    #Next two lines use info from https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
    p10 = np.percentile(final_values, 10)
    p90 = np.percentile(final_values, 90)

    #Build and display chart (50/50 mix of Group and other resources listed below)
    fig = build_chart(results, years, avg_final, min_final, max_final, p10, p90)
    st.pyplot(fig)

    #Save chart button (Written by Group)
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    st.download_button("Save Chart", data=buf.getvalue(), file_name="simulation_chart.png", mime="image/png")

    #Save to leaderboard button - guests see a login reminder instead (Written by Group)
    if st.session_state.current_user:
        if st.button("Save to Leaderboard"):
            save_to_leaderboard(years, avg_final, min_final, max_final)
    else:
        st.button("Save to Leaderboard (Login Required)", disabled=True)
        st.caption("You must be logged in to save to the leaderboard.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("New Simulation"):
            st.session_state.stocks_list = []
            st.session_state.sim_results = None
            go_to("stock_input")
            st.rerun()
    with col2:
        if st.button("Back to Home"):
            st.session_state.stocks_list = []
            st.session_state.sim_results = None
            go_to("welcome")
            st.rerun()

# ─────────────────────────────────────────────
# MAIN ROUTER — replaces root.mainloop()
# ─────────────────────────────────────────────
st.set_page_config(page_title=app_title, layout="centered")

screen = st.session_state.screen
if screen == "welcome":
    show_welcome_screen()
elif screen == "login":
    show_login_screen()
elif screen == "create_account":
    show_create_account_screen()
elif screen == "stock_input":
    show_stock_input_screen()
elif screen == "results":
    show_results_screen()
