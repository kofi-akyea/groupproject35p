"""
Executive UI theme configuration for Project Risk Intelligence Platform.
Provides clean, minimal, professional enterprise styling.
"""

import streamlit as st

COLORS = {
    # Primary accent color (Enterprise Indigo)
    "primary": "#4F46E5",
    "primary_dark": "#4338CA",
    "primary_light": "#6366F1",
    
    # Secondary neutral accents
    "secondary": "#64748B",
    "secondary_dark": "#475569",
    "secondary_light": "#94A3B8",
    
    # Accent color
    "accent": "#0284C7",
    "accent_light": "#38BDF8",
    
    # Background colors
    "background": "#F8FAFC",
    "card_background": "#FFFFFF",
    "sidebar_background": "#FFFFFF",
    
    # Text colors
    "text_primary": "#0F172A",
    "text_secondary": "#64748B",
    "text_tertiary": "#94A3B8",
    
    # Risk level colors
    "risk_low": "#2563EB",
    "risk_medium": "#D97706",
    "risk_high": "#EA580C",
    "risk_critical": "#DC2626",
    
    # UI elements
    "border": "#E2E8F0",
    "divider": "#E2E8F0",
    "shadow": "rgba(15, 23, 42, 0.04)",
    "shadow_hover": "rgba(15, 23, 42, 0.08)",
    
    # Status states
    "success": "#16A34A",
    "error": "#DC2626",
    "warning": "#D97706",
    "info": "#4F46E5",
}

FONTS = {
    "primary": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "heading": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "mono": "'SF Mono', Monaco, 'Cascadia Code', Consolas, monospace",
}

FONT_SIZES = {
    "xs": "12px",
    "sm": "13px",
    "base": "14px",
    "lg": "16px",
    "xl": "18px",
    "2xl": "22px",
    "3xl": "28px",
    "4xl": "32px",
}

SPACING = {
    "xs": "4px",
    "sm": "8px",
    "md": "16px",
    "lg": "20px",
    "xl": "24px",
    "2xl": "32px",
}

BORDER_RADIUS = {
    "sm": "6px",
    "md": "8px",
    "lg": "12px",
    "xl": "16px",
    "full": "9999px",
}

CUSTOM_CSS = f"""
<style>
    /* Global app styling */
    .stApp {{
        font-family: {FONTS['primary']};
        background-color: {COLORS['background']};
        color: {COLORS['text_primary']};
    }}
    
    /* Global spacing adjustments */
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }}
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {{
        font-family: {FONTS['heading']};
        color: {COLORS['text_primary']};
        font-weight: 600;
        letter-spacing: -0.01em;
    }}
    
    /* Header card */
    .app-header {{
        background-color: {COLORS['card_background']};
        border: 1px solid {COLORS['border']};
        border-radius: {BORDER_RADIUS['lg']};
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px {COLORS['shadow']};
    }}
    
    .app-header h1 {{
        font-size: {FONT_SIZES['2xl']};
        font-weight: 600;
        color: {COLORS['text_primary']};
        margin: 0;
    }}
    
    .app-header p {{
        font-size: {FONT_SIZES['sm']};
        color: {COLORS['text_secondary']};
        margin: 4px 0 0 0;
    }}
    
    /* Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
        padding: 4px 0;
        border-bottom: 1px solid {COLORS['border']};
        margin-bottom: 20px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 38px;
        background-color: transparent;
        border-radius: {BORDER_RADIUS['sm']};
        padding: 0 16px;
        border: none;
        transition: all 0.15s ease;
        color: {COLORS['text_secondary']};
        font-weight: 500;
        font-size: {FONT_SIZES['sm']};
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        color: {COLORS['text_primary']};
        background-color: #F1F5F9;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['card_background']} !important;
        color: {COLORS['primary']} !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 2px {COLORS['shadow']} !important;
        border: 1px solid {COLORS['border']} !important;
    }}
    
    /* Cards */
    .apple-card, .app-card {{
        background-color: {COLORS['card_background']};
        border-radius: {BORDER_RADIUS['lg']};
        box-shadow: 0 1px 3px {COLORS['shadow']};
        border: 1px solid {COLORS['border']};
        padding: {SPACING['lg']};
        margin-bottom: {SPACING['md']};
    }}
    
    /* Buttons */
    .stButton > button {{
        background-color: {COLORS['primary']};
        color: white;
        border: 1px solid transparent;
        border-radius: {BORDER_RADIUS['sm']};
        padding: 8px 16px;
        font-weight: 500;
        font-size: {FONT_SIZES['sm']};
        transition: all 0.15s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }}
    
    .stButton > button:hover {{
        background-color: {COLORS['primary_dark']};
        border-color: transparent;
        color: white;
    }}
    
    /* Metrics */
    [data-testid="stMetricValue"] {{
        font-family: {FONTS['heading']};
        font-size: {FONT_SIZES['3xl']};
        font-weight: 600;
        color: {COLORS['text_primary']};
    }}
    
    [data-testid="stMetricLabel"] {{
        font-family: {FONTS['primary']};
        font-size: {FONT_SIZES['xs']};
        font-weight: 600;
        color: {COLORS['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    /* Expanders & Forms */
    .stExpander {{
        border: 1px solid {COLORS['border']} !important;
        border-radius: {BORDER_RADIUS['md']} !important;
        background-color: {COLORS['card_background']} !important;
        box-shadow: none !important;
        margin-bottom: 8px !important;
    }}
    
    /* Badges */
    .apple-badge {{
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: {BORDER_RADIUS['full']};
        font-size: {FONT_SIZES['xs']};
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    
    .badge-low {{
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
    }}
    
    .badge-medium {{
        background-color: #FEF9C3;
        color: #A16207;
        border: 1px solid #FEF08A;
    }}
    
    .badge-high {{
        background-color: #FFEDD5;
        color: #C2410C;
        border: 1px solid #FED7AA;
    }}
    
    .badge-critical {{
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
    }}
</style>
"""

def apply_theme():
    """Apply the Executive UI theme to Streamlit."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)