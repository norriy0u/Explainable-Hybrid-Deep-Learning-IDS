"""
Custom Cybersecurity SOC Theme and Styling for Streamlit Dashboard.
Implements a modern, dark security operations center UI aesthetic.
"""

import streamlit as st

def apply_soc_theme():
    """Injects custom CSS for a dark SOC aesthetic."""
    st.markdown("""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    }

    code, pre, .stCodeBlock {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Styling */
    .main {
        background-color: #0B0F17;
        color: #E2E8F0;
    }
    
    .stApp {
        background-color: #0B0F17;
    }

    /* Top Navbar / Header Card */
    .soc-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .soc-header-title {
        font-size: 24px;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.5px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .soc-header-subtitle {
        font-size: 13px;
        color: #94A3B8;
        margin-top: 6px;
        font-weight: 400;
    }

    /* KPI Metric Cards */
    .kpi-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover {
        border-color: #3B82F6;
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #9CA3AF;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 800;
        color: #F9FAFB;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-delta {
        font-size: 12px;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Alert Banners */
    .alert-banner-critical {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.2) 0%, rgba(185, 28, 28, 0.1) 100%);
        border: 1px solid #EF4444;
        border-left: 6px solid #EF4444;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 16px 0;
    }
    .alert-banner-normal {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.2) 0%, rgba(4, 120, 87, 0.1) 100%);
        border: 1px solid #10B981;
        border-left: 6px solid #10B981;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 16px 0;
    }
    .alert-banner-warning {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.2) 0%, rgba(180, 83, 9, 0.1) 100%);
        border: 1px solid #F59E0B;
        border-left: 6px solid #F59E0B;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 16px 0;
    }

    /* Badges */
    .badge-critical {
        background-color: #EF4444;
        color: #FFFFFF;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
    }
    .badge-high {
        background-color: #F97316;
        color: #FFFFFF;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
    }
    .badge-medium {
        background-color: #F59E0B;
        color: #111827;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
    }
    .badge-low {
        background-color: #06B6D4;
        color: #111827;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
    }
    .badge-normal {
        background-color: #10B981;
        color: #FFFFFF;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #0D131F !important;
        border-right: 1px solid #1E293B;
    }
    
    /* Streamlit Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: #FFFFFF;
        border: 1px solid #3B82F6;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
        border-color: #60A5FA;
        box-shadow: 0 0 12px rgba(59, 130, 246, 0.4);
    }

    /* Tables */
    .dataframe {
        border-collapse: collapse !important;
        width: 100% !important;
        background-color: #111827 !important;
        color: #E5E7EB !important;
        font-size: 12px !important;
    }
    .dataframe th {
        background-color: #1F2937 !important;
        color: #9CA3AF !important;
        padding: 8px 12px !important;
        text-align: left !important;
        border-bottom: 2px solid #374151 !important;
    }
    .dataframe td {
        padding: 8px 12px !important;
        border-bottom: 1px solid #1F2937 !important;
    }
    </style>
    """, unsafe_allow_html=True)

