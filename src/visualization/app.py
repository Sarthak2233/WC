import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add project root to path so we can import from src
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Import from database.py for later integration
try:
    from src.database import WorldCup, Match, Player, Country, Culture, Conflict, Narrative, Prediction, get_db
except ImportError as e:
    st.warning(f"Could not import database models: {e}. Ensure you are running from the project root.")

# Set page config
st.set_page_config(
    page_title="World Cup Psychopolitical Engine",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #e0e0e0;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(0, 0, 0, 0.2);
        padding: 10px 10px 0 10px;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px 8px 0 0;
        color: #b0bec5;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
        border-bottom: 2px solid #00e5ff;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f2027;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Dataframes */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.4);
    }
    
    /* Plot containers */
    .stPlotlyChart {
        background-color: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        padding: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Generate mock data
@st.cache_data
def get_mock_tournament_data():
    return pd.DataFrame({
        'Year': [1998, 2002, 2006, 2010, 2014, 2018, 2022],
        'Host': ['France', 'South Korea/Japan', 'Germany', 'South Africa', 'Brazil', 'Russia', 'Qatar'],
        'Winner': ['France', 'Brazil', 'Italy', 'Spain', 'Germany', 'France', 'Argentina'],
        'Runner-up': ['Brazil', 'Germany', 'France', 'Netherlands', 'Argentina', 'Croatia', 'France'],
        'Goals Scored': [171, 161, 147, 145, 171, 169, 172]
    })

@st.cache_data
def get_mock_simulation_data():
    teams = ['Brazil', 'France', 'Argentina', 'England', 'Spain', 'Germany', 'Portugal', 'Netherlands']
    probs = [0.18, 0.15, 0.14, 0.12, 0.10, 0.09, 0.07, 0.05]
    probs = np.array(probs) / sum(probs) # Normalize
    return pd.DataFrame({
        'Team': teams,
        'Win Probability (%)': probs * 100,
        'Expected Points': np.random.uniform(3, 7, len(teams)),
        'Avg Tournament Progress': ['Final', 'Semi-Final', 'Semi-Final', 'Quarter-Final', 'Quarter-Final', 'Round of 16', 'Round of 16', 'Round of 16']
    })

@st.cache_data
def get_mock_psychopolitical_data():
    np.random.seed(42)
    teams = ['Brazil', 'France', 'Argentina', 'England', 'Germany']
    return pd.DataFrame({
        'Team': teams,
        'Adversity Score': np.random.uniform(40, 90, len(teams)),
        'Psychological Pressure Index (PPI)': np.random.uniform(50, 95, len(teams)),
        'Media Sentiment': np.random.uniform(-1, 1, len(teams)),
        'National Pride Index': np.random.uniform(60, 100, len(teams)),
        'Political Stability': np.random.uniform(30, 90, len(teams))
    })

# Main App Layout
st.title("🌍 World Cup Psychopolitical Engine")
st.markdown("*Advanced analytics combining football statistics with geopolitical, economic, and cultural dimensions.*")

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/e/e3/2022_FIFA_World_Cup.svg/400px-2022_FIFA_World_Cup.svg.png", width=150)
    st.markdown("### Settings")
    selected_year = st.selectbox("Select Tournament Year", [2026, 2022, 2018, 2014, 2010], index=0)
    st.markdown("---")
    st.markdown("### Model Configuration")
    ppi_weight = st.slider("PPI Impact Weight", 0.0, 1.0, 0.6)
    adv_weight = st.slider("Adversity Impact Weight", 0.0, 1.0, 0.4)
    st.button("Update Model", use_container_width=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🏆 Tournament Explorer", "🔮 Simulation Explorer", "🧠 Psychopolitical Dossier"])

# --------------------------
# Tab 1: Tournament Explorer
# --------------------------
with tab1:
    st.header("Historical Tournament Explorer")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Tournaments Tracked", "22", "+1 upcoming")
    with col2:
        st.metric("Most Titles", "Brazil", "5 Titles")
    with col3:
        st.metric("Avg Goals/Tournament (Modern Era)", "160.8", "+2.4% vs past")
        
    st.markdown("### Historical Overview")
    df_tournaments = get_mock_tournament_data()
    st.dataframe(df_tournaments, use_container_width=True, hide_index=True)
    
    st.markdown("### Goals Scored Trend")
    fig1 = px.line(df_tournaments, x='Year', y='Goals Scored', markers=True, 
                   color_discrete_sequence=['#00e5ff'],
                   template="plotly_dark")
    fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                       xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'))
    st.plotly_chart(fig1, use_container_width=True)

# --------------------------
# Tab 2: Simulation Explorer
# --------------------------
with tab2:
    st.header("Monte Carlo Simulation Engine")
    st.markdown(f"Running 10,000 simulations for **{selected_year} World Cup** using psychopolitical weights.")
    
    df_sim = get_mock_simulation_data()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Win Probabilities")
        fig2 = px.bar(df_sim.sort_values('Win Probability (%)', ascending=True), 
                      x='Win Probability (%)', y='Team', orientation='h',
                      color='Win Probability (%)', color_continuous_scale='Viridis',
                      template="plotly_dark")
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                           yaxis=dict(showgrid=False),
                           coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)
        
    with col2:
        st.markdown("### Top Contender Stats")
        st.dataframe(df_sim[['Team', 'Avg Tournament Progress']], use_container_width=True, hide_index=True)
        st.info("💡 **Insight:** Combining economic stability and national pride significantly boosts Argentina's expected progression.")

    st.markdown("### Tournament Bracket Simulation (Most Likely Path)")
    # Mocking a bracket visual using a simple flowchart or columns
    bcol1, bcol2, bcol3 = st.columns(3)
    with bcol1:
        st.markdown("**Quarter-Finals**")
        st.success("🇧🇷 Brazil (2) vs (1) 🇪🇸 Spain")
        st.warning("🇫🇷 France (3) vs (0) 🇵🇹 Portugal")
    with bcol2:
        st.markdown("**Semi-Finals**")
        st.success("🇧🇷 Brazil (1) vs (1) 🇫🇷 France (Pens: 4-3)")
    with bcol3:
        st.markdown("**Final**")
        st.error("🏆 🇧🇷 Brazil (2) vs (1) 🇦🇷 Argentina")

# --------------------------
# Tab 3: Psychopolitical Dossier
# --------------------------
with tab3:
    st.header("Psychopolitical Team Dossiers")
    
    df_psycho = get_mock_psychopolitical_data()
    
    selected_team = st.selectbox("Select Team for Dossier", df_psycho['Team'].tolist())
    team_data = df_psycho[df_psycho['Team'] == selected_team].iloc[0]
    
    st.markdown(f"### 📋 Profile: {selected_team}")
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Adversity Score", f"{team_data['Adversity Score']:.1f}/100", 
                 delta="High resilience" if team_data['Adversity Score'] > 70 else "Average resilience")
    col_b.metric("PPI (Pressure)", f"{team_data['Psychological Pressure Index (PPI)']:.1f}/100", 
                 delta="-Critical" if team_data['Psychological Pressure Index (PPI)'] > 80 else "Manageable",
                 delta_color="inverse")
    col_c.metric("National Pride", f"{team_data['National Pride Index']:.1f}/100")

    st.markdown("### Radar Chart: Psychopolitical Dimensions")
    
    categories = ['Adversity Score', 'PPI', 'Media Sentiment (x100)', 'National Pride', 'Political Stability']
    # Adjust media sentiment to 0-100 scale for radar
    media_scaled = (team_data['Media Sentiment'] + 1) * 50 
    
    values = [
        team_data['Adversity Score'],
        team_data['Psychological Pressure Index (PPI)'],
        media_scaled,
        team_data['National Pride Index'],
        team_data['Political Stability']
    ]
    
    fig3 = go.Figure(data=go.Scatterpolar(
      r=values,
      theta=categories,
      fill='toself',
      line_color='#00e5ff',
      fillcolor='rgba(0, 229, 255, 0.3)'
    ))
    
    fig3.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 100],
          gridcolor='rgba(255,255,255,0.2)'
        ),
        angularaxis=dict(
            gridcolor='rgba(255,255,255,0.2)'
        ),
        bgcolor='rgba(0,0,0,0)'
      ),
      showlegend=False,
      template="plotly_dark",
      plot_bgcolor='rgba(0,0,0,0)',
      paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("---")
    st.markdown("""
    **Insight Engine output:**
    *The selected team shows a distinct profile. A high Adversity Score typically correlates with overperformance in knockout stages, while high PPI can lead to underperformance in penalty shootouts due to media and public pressure.*
    """)
