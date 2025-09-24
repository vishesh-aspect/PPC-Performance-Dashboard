"""
ppc_dashboard.py
Complete PPC Marketing Dashboard with Updated Funnel and Cleanup
"""

import sys
import traceback
import pandas as pd
import streamlit as st
from simple_salesforce.login import SalesforceLogin
from simple_salesforce import Salesforce
from datetime import datetime, timedelta, date
import plotly.express as px
import plotly.graph_objects as go

# ----------------- STREAMLIT CONFIG -----------------
st.set_page_config(
    page_title="PPC Marketing Performance Dashboard", 
    page_icon="📊", 
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4F46E5;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 PPC Marketing Performance Dashboard")

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Date Selection Mode
    st.subheader("📅 Date Selection")
    date_mode = st.radio(
        "Select Date Mode",
        ["Single Day", "Date Range"],
        help="Choose between single day or date range analysis"
    )
    
    if date_mode == "Single Day":
        selected_date = st.date_input(
            "Select Date",
            value=date.today() - timedelta(days=1),  # Default to yesterday
            min_value=date(2023, 7, 1),
            max_value=date.today()
        )
        start_date = selected_date
        end_date = selected_date
    else:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=date(2025, 9, 15),
                min_value=date(2023, 7, 1),
                max_value=date.today()
            )
        with col2:
            end_date = st.date_input(
                "End Date", 
                value=date(2025, 9, 19),
                min_value=date(2023, 7, 1),
                max_value=date.today()
            )
    
    # Google Ads Upload
    st.markdown("---")
    st.subheader("📊 Google Ads Data")
    uploaded_file = st.file_uploader(
        "Upload Google Ads Export",
        type=['xlsx', 'csv'],
        help="Upload Google Ads data for complete ROI analysis"
    )
    
    # Refresh button
    st.markdown("---")
    if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ----------------- SALESFORCE AUTH -----------------
@st.cache_data(ttl=300, show_spinner=False)
def sf_client():
    try:
        username = st.secrets.get("SF_USERNAME")
        password = st.secrets.get("SF_PASSWORD")
        token = st.secrets.get("SF_TOKEN")
        domain = st.secrets.get("SF_DOMAIN", "login")
        session_id, instance = SalesforceLogin(
            username=username,
            password=password,
            security_token=token,
            domain=domain
        )
        return Salesforce(session_id=session_id, instance=instance), instance
    except Exception as e:
        st.error(f"Failed to connect to Salesforce: {str(e)}")
        return None, None

# ----------------- NEW FUNNEL FUNCTIONS -----------------
def get_not_rejected_bookings(sf, start_date, end_date):
    """Get count of PPC bookings that are Complete or Accepted/Live"""
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # OLB Not Rejected Query
    try:
        olb_not_rejected_query = f"""
            SELECT COUNT(Id)
            FROM Online_Bookings__c 
            WHERE CreatedDate >= {start_str}T00:00:00Z 
            AND CreatedDate <= {end_str}T23:59:59Z
            AND Work_Order__r.Status IN ('Complete','Accepted/Live')
            AND AFL_UTM_Medium__c LIKE '%Paid%'
        """
        result = sf.query(olb_not_rejected_query)
        olb_not_rejected = result['records'][0].get('expr0', 0) if result['records'] else 0
    except Exception as e:
        st.warning(f"Could not fetch OLB not rejected: {str(e)}")
        olb_not_rejected = 0
    
    # Call Not Rejected Query
    try:
        call_not_rejected_query = f"""
            SELECT COUNT(Id)
            FROM Task 
            WHERE ResponseTap_Call__r.RTap__Medium__c LIKE '%Paid%' 
            AND Predicted_Outcome__c LIKE '%Lead - Booked%' 
            AND CreatedDate >= {start_str}T00:00:00Z 
            AND CreatedDate <= {end_str}T23:59:59Z
            AND Work_Order__r.Status IN ('Complete','Accepted/Live')
        """
        result = sf.query(call_not_rejected_query)
        call_not_rejected = result['records'][0].get('expr0', 0) if result['records'] else 0
    except Exception as e:
        st.warning(f"Could not fetch Call not rejected: {str(e)}")
        call_not_rejected = 0
    
    return olb_not_rejected + call_not_rejected

def get_booking_opportunities(sf, start_date, end_date):
    """Get both Call and OLB booking opportunities"""
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Call Booking Opportunities
    try:
        call_booking_opps_query = f"""
            SELECT COUNT(Id)
            FROM RTap__ResponseTap_Call__c 
            WHERE Call_Date__c >= {start_str} AND Call_Date__c <= {end_str}
            AND RTap__Medium__c = 'Paid Search'
            AND Predicted_Type_of_Call__c LIKE '%Booking Opportunity%'
        """
        result = sf.query(call_booking_opps_query)
        call_booking_opps = result['records'][0].get('expr0', 0) if result['records'] else 0
    except Exception as e:
        st.warning(f"Could not fetch call booking opportunities: {str(e)}")
        call_booking_opps = 0
    
    # OLB Booking Opportunities (all OLBs are opportunities)
    try:
        olb_booking_opps_query = f"""
            SELECT COUNT(Id)
            FROM Online_Bookings__c 
            WHERE CreatedDate >= {start_str}T00:00:00Z AND CreatedDate <= {end_str}T23:59:59Z
            AND AFL_UTM_Medium__c = 'Paid Search'
        """
        result = sf.query(olb_booking_opps_query)
        olb_booking_opps = result['records'][0].get('expr0', 0) if result['records'] else 0
    except Exception as e:
        st.warning(f"Could not fetch OLB booking opportunities: {str(e)}")
        olb_booking_opps = 0
    
    return call_booking_opps + olb_booking_opps

# ----------------- PPC REVENUE FUNCTIONS FOR ROI -----------------
def get_ppc_olb_revenue(sf, start_date, end_date):
    """Get PPC OLB revenue data for ROI calculation"""
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    try:
        olb_revenue_query = f"""
            SELECT 
                Name,
                AFL_UTM_Medium__c,
                Work_Order__r.WorkOrderNumber, 
                Work_Order__r.CCT_Charge_NET__c, 
                Work_Order__r.Status
            FROM Online_Bookings__c 
            WHERE CreatedDate >= {start_str}T00:00:00Z 
            AND CreatedDate <= {end_str}T23:59:59Z
            AND Work_Order__r.Status = 'Complete'
            AND AFL_UTM_Medium__c LIKE '%Paid%'
        """
        
        result = sf.query_all(olb_revenue_query)
        
        if result['records']:
            df = pd.DataFrame(result['records'])
            
            # Clean up the nested structure
            if 'Work_Order__r' in df.columns:
                df['CCT_Charge_NET'] = df['Work_Order__r'].apply(
                    lambda x: x.get('CCT_Charge_NET__c') if isinstance(x, dict) and x else None
                )
                df['WorkOrderNumber'] = df['Work_Order__r'].apply(
                    lambda x: x.get('WorkOrderNumber') if isinstance(x, dict) and x else None
                )
                df['Status'] = df['Work_Order__r'].apply(
                    lambda x: x.get('Status') if isinstance(x, dict) and x else None
                )
            
            # Drop nested columns and attributes
            columns_to_drop = ['Work_Order__r', 'attributes']
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
            
            # Convert to numeric and sum
            df['CCT_Charge_NET'] = pd.to_numeric(df['CCT_Charge_NET'], errors='coerce').fillna(0)
            total_olb_revenue = df['CCT_Charge_NET'].sum()
            
            return total_olb_revenue, len(df), df
        else:
            return 0, 0, pd.DataFrame()
            
    except Exception as e:
        st.warning(f"Could not fetch OLB revenue data: {str(e)}")
        return 0, 0, pd.DataFrame()

def get_ppc_rtc_revenue(sf, start_date, end_date):
    """Get PPC RTC (ResponseTap Call) revenue data for ROI calculation"""
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    try:
        rtc_revenue_query = f"""
            SELECT 
                Id, 
                Work_Order__r.CCT_Charge_NET__c, 
                Predicted_Outcome__c, 
                ResponseTap_Call__r.RTap__Medium__c
            FROM Task 
            WHERE ResponseTap_Call__r.RTap__Medium__c LIKE '%Paid%' 
            AND Predicted_Outcome__c LIKE '%Lead - Booked%' 
            AND CreatedDate >= {start_str}T00:00:00Z 
            AND CreatedDate <= {end_str}T23:59:59Z
        """
        
        result = sf.query_all(rtc_revenue_query)
        
        if result['records']:
            df = pd.DataFrame(result['records'])
            
            # Clean up the nested structure
            if 'Work_Order__r' in df.columns:
                df['CCT_Charge_NET'] = df['Work_Order__r'].apply(
                    lambda x: x.get('CCT_Charge_NET__c') if isinstance(x, dict) and x else None
                )
            
            if 'ResponseTap_Call__r' in df.columns:
                df['RTap_Medium'] = df['ResponseTap_Call__r'].apply(
                    lambda x: x.get('RTap__Medium__c') if isinstance(x, dict) and x else None
                )
            
            # Drop nested columns and attributes
            columns_to_drop = ['Work_Order__r', 'ResponseTap_Call__r', 'attributes']
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
            
            # Convert to numeric and sum
            df['CCT_Charge_NET'] = pd.to_numeric(df['CCT_Charge_NET'], errors='coerce').fillna(0)
            total_rtc_revenue = df['CCT_Charge_NET'].sum()
            
            return total_rtc_revenue, len(df), df
        else:
            return 0, 0, pd.DataFrame()
            
    except Exception as e:
        st.warning(f"Could not fetch RTC revenue data: {str(e)}")
        return 0, 0, pd.DataFrame()

def calculate_roi(ppc_revenue, ppc_spend):
    """Calculate ROI: Revenue / Spend"""
    if ppc_spend > 0:
        return ppc_revenue / ppc_spend
    else:
        return 0

def calculate_roas(ppc_revenue, ppc_bookings):
    """Calculate ROAS: Revenue / Number of Bookings (Revenue per Booking)"""
    if ppc_bookings > 0:
        return ppc_revenue / ppc_bookings
    else:
        return 0

def get_campaign_rtc_data(sf, query_date):
    """Get RTC (ResponseTap Call) data by campaign for a specific date"""
    
    date_str = query_date.strftime('%Y-%m-%d')
    
    try:
        rtc_query = f"""
            SELECT 
                Id, 
                Work_Order__r.CCT_Charge_NET__c, 
                Predicted_Outcome__c, 
                ResponseTap_Call__r.RTap__Medium__c, 
                ResponseTap_Call__r.RTap__Campaign__c
            FROM Task 
            WHERE ResponseTap_Call__r.RTap__Medium__c LIKE '%Paid%' 
            AND Predicted_Outcome__c LIKE '%Lead - Booked%' 
            AND DAY_ONLY(CreatedDate) = {date_str}
        """
        
        result = sf.query_all(rtc_query)
        
        if result['records']:
            df = pd.DataFrame(result['records'])
            
            # Clean up the nested structure
            if 'Work_Order__r' in df.columns:
                df['CCT_Charge_NET'] = df['Work_Order__r'].apply(
                    lambda x: x.get('CCT_Charge_NET__c') if isinstance(x, dict) and x else None
                )
            
            if 'ResponseTap_Call__r' in df.columns:
                df['RTap_Medium'] = df['ResponseTap_Call__r'].apply(
                    lambda x: x.get('RTap__Medium__c') if isinstance(x, dict) and x else None
                )
                df['RTap_Campaign'] = df['ResponseTap_Call__r'].apply(
                    lambda x: x.get('RTap__Campaign__c') if isinstance(x, dict) and x else None
                )
            
            # Drop the nested columns and attributes
            columns_to_drop = ['Work_Order__r', 'ResponseTap_Call__r', 'attributes']
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
            
            # Convert charge to numeric
            df['CCT_Charge_NET'] = pd.to_numeric(df['CCT_Charge_NET'], errors='coerce')
            
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error fetching RTC campaign data: {str(e)}")
        return pd.DataFrame()

def get_campaign_olb_data(sf, query_date):
    """Get OLB (Online Bookings) data by campaign for a specific date"""
    
    date_str = query_date.strftime('%Y-%m-%d')
    
    try:
        olb_query = f"""
            SELECT 
                Name,
                AFL_UTM_Medium__c,
                Work_Order__r.WorkOrderNumber, 
                Work_Order__r.CCT_Charge_NET__c, 
                Work_Order__r.Status, 
                AFL_UTM_Campaign__c 
            FROM Online_Bookings__c  
            WHERE DAY_ONLY(CreatedDate) = {date_str}  
            AND Work_Order__r.Status = 'Complete' 
            AND AFL_UTM_Medium__c LIKE '%Paid%'
        """
        
        result = sf.query_all(olb_query)
        
        if result['records']:
            df = pd.DataFrame(result['records'])
            
            # Clean up the nested structure
            if 'Work_Order__r' in df.columns:
                df['WorkOrderNumber'] = df['Work_Order__r'].apply(
                    lambda x: x.get('WorkOrderNumber') if isinstance(x, dict) and x else None
                )
                df['CCT_Charge_NET'] = df['Work_Order__r'].apply(
                    lambda x: x.get('CCT_Charge_NET__c') if isinstance(x, dict) and x else None
                )
                df['WO_Status'] = df['Work_Order__r'].apply(
                    lambda x: x.get('Status') if isinstance(x, dict) and x else None
                )
            
            # Drop the nested columns and attributes
            columns_to_drop = ['Work_Order__r', 'attributes']
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
            
            # Convert charge to numeric
            df['CCT_Charge_NET'] = pd.to_numeric(df['CCT_Charge_NET'], errors='coerce')
            
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error fetching OLB campaign data: {str(e)}")
        return pd.DataFrame()

def process_google_ads_data(uploaded_file, start_date, end_date):
    """Process uploaded Google Ads data and extract Impressions, Clicks, and Conversions"""
    if not uploaded_file:
        return pd.DataFrame()
    
    try:
        # Read the file based on extension
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        # Print original columns for debugging
        st.write("Original columns:", df.columns.tolist())
        
        # Standardize column names - updated mapping based on your data
        column_mapping = {
            'Day': 'Date',
            'Campaign': 'Campaign',
            'Conversions': 'Conversions',
            'Impr.': 'Impressions',
            'Impressions': 'Impressions',
            'CTR': 'CTR',
            'Currency code': 'Currency',
            'Cost': 'Cost',
            'Clicks': 'Clicks'
        }
        
        # Apply column mapping
        df_renamed = df.rename(columns=column_mapping)
        
        # Ensure we have the required columns
        required_columns = ['Date', 'Campaign', 'Impressions', 'Clicks', 'Conversions', 'Cost']
        missing_columns = [col for col in required_columns if col not in df_renamed.columns]
        
        if missing_columns:
            st.warning(f"Missing columns in uploaded data: {missing_columns}")
            # Show available columns to help user
            st.write("Available columns after mapping:", df_renamed.columns.tolist())
        
        # Convert Date column to datetime
        df_renamed['Date'] = pd.to_datetime(df_renamed['Date'], errors='coerce')
        
        # Filter by date range
        df_filtered = df_renamed[
            (df_renamed['Date'] >= pd.to_datetime(start_date)) & 
            (df_renamed['Date'] <= pd.to_datetime(end_date))
        ]
        
        # Convert numeric columns to proper data types
        numeric_columns = ['Impressions', 'Clicks', 'Conversions', 'Cost']
        for col in numeric_columns:
            if col in df_filtered.columns:
                df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce').fillna(0)
        
        # Add CTR calculation if not present
        if 'CTR' not in df_filtered.columns and 'Impressions' in df_filtered.columns and 'Clicks' in df_filtered.columns:
            df_filtered['CTR'] = (df_filtered['Clicks'] / df_filtered['Impressions'] * 100).round(2)
        
        # Add Conversion Rate calculation
        if 'Conversion_Rate' not in df_filtered.columns and 'Conversions' in df_filtered.columns and 'Clicks' in df_filtered.columns:
            df_filtered['Conversion_Rate'] = (df_filtered['Conversions'] / df_filtered['Clicks'] * 100).round(2)
        
        # Add Cost per Conversion
        if 'Cost_Per_Conversion' not in df_filtered.columns and 'Cost' in df_filtered.columns and 'Conversions' in df_filtered.columns:
            df_filtered['Cost_Per_Conversion'] = (df_filtered['Cost'] / df_filtered['Conversions']).round(2)
            df_filtered['Cost_Per_Conversion'] = df_filtered['Cost_Per_Conversion'].replace([float('inf'), -float('inf')], 0)
        
        st.success(f"Successfully processed {len(df_filtered)} rows of Google Ads data")
        
        return df_filtered
        
    except Exception as e:
        st.error(f"Error processing Google Ads data: {str(e)}")
        st.write("Please ensure your file has the following columns: Day, Campaign, Impressions, Clicks, Conversions, Cost")
        return pd.DataFrame()

def get_google_ads_summary(df_ads):
    """Get summary metrics from Google Ads data"""
    if df_ads.empty:
        return {}
    
    total_impressions = df_ads['Impressions'].sum() if 'Impressions' in df_ads.columns else 0
    total_clicks = df_ads['Clicks'].sum() if 'Clicks' in df_ads.columns else 0
    total_conversions = df_ads['Conversions'].sum() if 'Conversions' in df_ads.columns else 0
    total_cost = df_ads['Cost'].sum() if 'Cost' in df_ads.columns else 0
    
    summary = {
        'Total_Impressions': int(total_impressions),
        'Total_Clicks': int(total_clicks),
        'Total_Conversions': int(total_conversions),
        'Total_Cost': round(total_cost, 2),
        # Calculate proper CTR from totals, not average of individual CTRs
        'Overall_CTR': round((total_clicks / total_impressions * 100), 2) if total_impressions > 0 else 0,
        # Calculate proper Conversion Rate from totals
        'Overall_Conversion_Rate': round((total_conversions / total_clicks * 100), 2) if total_clicks > 0 else 0,
        'Cost_Per_Click': round(total_cost / total_clicks, 2) if total_clicks > 0 else 0,
        'Cost_Per_Conversion': round(total_cost / total_conversions, 2) if total_conversions > 0 else 0
    }
    
    return summary

def get_google_ads_daily_breakdown(df_ads):
    """Create daily breakdown of Google Ads metrics"""
    if df_ads.empty:
        return pd.DataFrame()
    
    daily_breakdown = df_ads.groupby('Date').agg({
        'Impressions': 'sum',
        'Clicks': 'sum',
        'Conversions': 'sum',
        'Cost': 'sum'
    }).reset_index()
    
    # Calculate daily CTR and Conversion Rate
    daily_breakdown['CTR'] = (daily_breakdown['Clicks'] / daily_breakdown['Impressions'] * 100).round(2)
    daily_breakdown['Conversion_Rate'] = (daily_breakdown['Conversions'] / daily_breakdown['Clicks'] * 100).round(2)
    daily_breakdown['Cost_Per_Conversion'] = (daily_breakdown['Cost'] / daily_breakdown['Conversions']).round(2)
    daily_breakdown['Cost_Per_Click'] = (daily_breakdown['Cost'] / daily_breakdown['Clicks']).round(2)
    
    # Replace infinite values with 0
    daily_breakdown = daily_breakdown.replace([float('inf'), -float('inf')], 0)
    
    return daily_breakdown

# ----------------- NEW CLIENTS QUERY FUNCTION -----------------
def get_new_clients_count(sf, query_date):
    """Get count of new PPC clients for a specific date"""
    
    date_str = query_date.strftime('%Y-%m-%d')
    next_date_str = (query_date + timedelta(days=1)).strftime('%Y-%m-%d')
    
    new_clients = {
        'OLB_New_Clients': 0,
        'Call_New_Clients': 0
    }
    
    # Query for OLB New Clients
    try:
        olb_new_clients_query = f"""
            SELECT COUNT(Id)
            FROM WorkOrder
            WHERE DAY_ONLY(Job__r.CreatedDate) = {date_str}
            AND DAY_ONLY(Account.CreatedDate) = {date_str}
            AND Id IN (
                SELECT Work_Order__c
                FROM Online_Bookings__c
                WHERE AFL_UTM_Medium__c LIKE '%Paid Search%'
            )
        """
        
        result = sf.query(olb_new_clients_query)
        
        # COUNT queries return the count in 'expr0' field of the first record
        if result['records'] and len(result['records']) > 0:
            new_clients['OLB_New_Clients'] = result['records'][0].get('expr0', 0)
        else:
            new_clients['OLB_New_Clients'] = 0
            
    except Exception as e:
        # Try alternative query structure
        try:
            alt_query = f"""
                SELECT COUNT(Id) cnt
                FROM WorkOrder
                WHERE DAY_ONLY(Job__r.CreatedDate) = {date_str}
                AND DAY_ONLY(Account.CreatedDate) = {date_str}
                AND Id IN (
                    SELECT Work_Order__c
                    FROM Online_Bookings__c
                    WHERE AFL_UTM_Medium__c = 'Paid Search'
                )
            """
            result = sf.query(alt_query)
            if result['records'] and len(result['records']) > 0:
                new_clients['OLB_New_Clients'] = result['records'][0].get('cnt', 0)
        except:
            st.warning(f"Could not fetch OLB new clients for {date_str}: {str(e)}")
    
    # Query for Call New Clients
    try:
        # Using the provided query adapted for the specific date
        call_new_clients_query = f"""
            SELECT COUNT(Name)
            FROM RTap__ResponseTap_Call__c  
            WHERE DAY_ONLY(CreatedDate) = {date_str}
            AND Predicted_Summary__c = 'Booked'
            AND Predicted_Contact_Purpose__c = 'New Client'
            AND RTap__Medium__c = 'Paid Search'
        """
        
        result = sf.query(call_new_clients_query)
        
        # COUNT queries return the count in 'expr0' field of the first record
        if result['records'] and len(result['records']) > 0:
            new_clients['Call_New_Clients'] = result['records'][0].get('expr0', 0)
        else:
            new_clients['Call_New_Clients'] = 0
            
    except Exception as e:
        # Try with alias for better field naming
        try:
            alt_call_query = f"""
                SELECT COUNT(Name) cnt
                FROM RTap__ResponseTap_Call__c  
                WHERE DAY_ONLY(CreatedDate) = {date_str}
                AND Predicted_Summary__c = 'Booked'
                AND Predicted_Contact_Purpose__c = 'New Client'
                AND RTap__Medium__c = 'Paid Search'
            """
            result = sf.query(alt_call_query)
            if result['records'] and len(result['records']) > 0:
                new_clients['Call_New_Clients'] = result['records'][0].get('cnt', 0)
        except:
            st.warning(f"Could not fetch Call new clients for {date_str}: {str(e)}")
    
    return new_clients

# ----------------- SINGLE DAY OLB QUERY FUNCTION -----------------
def get_single_day_olb_count(sf, query_date):
    """Get OLB count for a single specific date"""
    
    date_str = query_date.strftime('%Y-%m-%d')
    next_date_str = (query_date + timedelta(days=1)).strftime('%Y-%m-%d')
    
    try:
        olb_query = f"""
            SELECT COUNT(Id)
            FROM Online_Bookings__c 
            WHERE CreatedDate >= {date_str}T00:00:00Z 
            AND CreatedDate < {next_date_str}T00:00:00Z
            AND AFL_UTM_Medium__c = 'Paid Search'
        """
        
        result = sf.query(olb_query)
        
        if result['records'] and len(result['records']) > 0:
            count_value = result['records'][0].get('expr0', 0)
            if count_value is None:
                count_value = result['records'][0].get('cnt', 0)
            return count_value if count_value is not None else 0
        else:
            return 0
            
    except Exception as e:
        st.warning(f"Could not fetch OLB data for {date_str}: {str(e)}")
        return 0
    
# ----------------- QUERY FUNCTIONS -----------------
def get_daily_breakdown(sf, start_date, end_date):
    """Fetch daily breakdown of all metrics"""
    
    # Format dates for SOQL
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Initialize data structure for each day
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_data = {}
    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        daily_data[date_str] = {
            'Date': date_str,
            'All Calls': 0,
            'PPC Calls': 0,
            'PPC Booking Opps': 0,
            'PPC Calls Booked': 0,
            'PPC OLB': 0,  # Initialize PPC OLB here
            'PPC New Clients': 0
        }
    
    # Query 1: Get ALL calls by date
    try:
        all_calls_query = f"""
            SELECT Call_Date__c, COUNT(Id) cnt
            FROM RTap__ResponseTap_Call__c 
            WHERE Call_Date__c >= {start_str} 
            AND Call_Date__c <= {end_str}
            GROUP BY Call_Date__c
        """
        result = sf.query_all(all_calls_query)
        for record in result['records']:
            if record['Call_Date__c'] in daily_data:
                daily_data[record['Call_Date__c']]['All Calls'] = record['cnt']
    except Exception as e:
        st.warning(f"Could not fetch all calls: {str(e)}")
    
    # Query 2: Get PPC calls by date
    try:
        ppc_calls_query = f"""
            SELECT Call_Date__c, COUNT(Id) cnt
            FROM RTap__ResponseTap_Call__c 
            WHERE Call_Date__c >= {start_str} 
            AND Call_Date__c <= {end_str}
            AND RTap__Medium__c = 'Paid Search'
            GROUP BY Call_Date__c
        """
        result = sf.query_all(ppc_calls_query)
        for record in result['records']:
            if record['Call_Date__c'] in daily_data:
                daily_data[record['Call_Date__c']]['PPC Calls'] = record['cnt']
    except Exception as e:
        st.warning(f"Could not fetch PPC calls: {str(e)}")
    
    # Query 3: Get PPC Booking Opportunities by date
    try:
        booking_opps_query = f"""
            SELECT Call_Date__c, COUNT(Id) cnt
            FROM RTap__ResponseTap_Call__c 
            WHERE Call_Date__c >= {start_str} 
            AND Call_Date__c <= {end_str}
            AND RTap__Medium__c = 'Paid Search'
            AND Predicted_Type_of_Call__c LIKE '%Booking Opportunity%'
            GROUP BY Call_Date__c
        """
        result = sf.query_all(booking_opps_query)
        for record in result['records']:
            if record['Call_Date__c'] in daily_data:
                daily_data[record['Call_Date__c']]['PPC Booking Opps'] = record['cnt']
    except Exception as e:
        st.warning(f"Could not fetch booking opportunities: {str(e)}")
    
    # Query 4: Get PPC Calls Booked by date
    try:
        calls_booked_query = f"""
            SELECT Call_Date__c, COUNT(Id) cnt
            FROM RTap__ResponseTap_Call__c 
            WHERE Call_Date__c >= {start_str} 
            AND Call_Date__c <= {end_str}
            AND RTap__Medium__c = 'Paid Search'
            AND Predicted_Summary__c = 'Booked'
            GROUP BY Call_Date__c
        """
        result = sf.query_all(calls_booked_query)
        for record in result['records']:
            if record['Call_Date__c'] in daily_data:
                daily_data[record['Call_Date__c']]['PPC Calls Booked'] = record['cnt']
    except Exception as e:
        st.warning(f"Could not fetch calls booked: {str(e)}")
    
    # Query 5: Get PPC OLB by date  
    try:
        # For OLB, we need to convert CreatedDate to date format
        for single_date in date_range:
            date_str = single_date.strftime('%Y-%m-%d')
            next_date = (single_date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            olb_query = f"""
                SELECT COUNT(Id) cnt
                FROM Online_Bookings__c 
                WHERE CreatedDate >= {date_str}T00:00:00Z 
                AND CreatedDate < {next_date}T00:00:00Z
                AND AFL_UTM_Medium__c = 'Paid Search'
            """
            
            result = sf.query(olb_query)
            
            # Fix: Get the actual count value, not just checking if > 0
            if result['records'] and len(result['records']) > 0:
                # COUNT queries return the count in 'expr0' field
                count_value = result['records'][0].get('expr0', 0)
                if count_value is None:
                    # Try with 'cnt' alias
                    count_value = result['records'][0].get('cnt', 0)
                daily_data[date_str]['PPC OLB'] = count_value if count_value is not None else 0
            
            # Get new clients for this date
            new_clients = get_new_clients_count(sf, single_date)
            daily_data[date_str]['PPC New Clients'] = new_clients['OLB_New_Clients'] + new_clients['Call_New_Clients']
                
    except Exception as e:
        st.warning(f"Could not fetch OLB data: {str(e)}")
    
    # Convert to DataFrame
    if daily_data:
        df = pd.DataFrame(list(daily_data.values()))
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df['Total PPC Bookings'] = df['PPC Calls Booked'] + df['PPC OLB']
        return df
    else:
        return pd.DataFrame()

def get_summary_kpis(sf, start_date, end_date):
    """Get summary KPIs using the working queries"""
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    QUERIES = {
        "All Calls": f"""
            SELECT Id 
            FROM RTap__ResponseTap_Call__c 
            WHERE Call_Date__c >= {start_str} AND Call_Date__c <= {end_str}
        """,
        "PPC Calls": f"""
            SELECT Id 
            FROM RTap__ResponseTap_Call__c 
            WHERE Call_Date__c >= {start_str} AND Call_Date__c <= {end_str}
            AND RTap__Medium__c = 'Paid Search'
        """,
        "PPC OLB": f"""
            SELECT Id, CreatedDate
            FROM Online_Bookings__c 
            WHERE CreatedDate >= {start_str}T00:00:00Z AND CreatedDate <= {end_str}T23:59:59Z
            AND AFL_UTM_Medium__c = 'Paid Search'
        """,
        "PPC Calls Booked": f"""
            SELECT Id 
            FROM RTap__ResponseTap_Call__c 
            WHERE Call_Date__c >= {start_str} AND Call_Date__c <= {end_str}
            AND RTap__Medium__c = 'Paid Search'
            AND Predicted_Summary__c = 'Booked'
        """
    }
    
    results = {}
    
    for label, q in QUERIES.items():
        try:
            res = sf.query_all(q)
            results[label] = len(res["records"])
        except Exception as e:
            results[label] = 0
            st.warning(f"Query failed for {label}: {e}")
    
    # Get combined booking opportunities (Call + OLB)
    results["PPC Booking Opps"] = get_booking_opportunities(sf, start_date, end_date)
    
    # Get new clients count
    total_new_clients = 0
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    for single_date in date_range:
        new_clients = get_new_clients_count(sf, single_date)
        total_new_clients += new_clients['OLB_New_Clients'] + new_clients['Call_New_Clients']
    
    results["PPC New Clients"] = total_new_clients
    
    # Derived metrics
    results["Total PPC Bookings"] = results["PPC Calls Booked"] + results["PPC OLB"]
    
    return results

# ----------------- MAIN APP -----------------
if date_mode == "Single Day":
    st.markdown(f"**Date:** {selected_date.strftime('%B %d, %Y')}")
else:
    st.markdown(f"**Date Range:** {start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}")

# Initialize Salesforce connection
sf, instance = sf_client()

if sf:
    # Create tabs - Updated with removed Campaign Analysis
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview", "📈 Daily Breakdown", "📉 Visualizations", "🎯 Google Ads Analysis", "📈 Campaign Analysis", "📋 Raw Data"])
    
    with tab1:
        # Fetch summary KPIs and revenue data
        with st.spinner("Fetching data from Salesforce..."):
            kpis = get_summary_kpis(sf, start_date, end_date)
            
            # Fetch PPC revenue data for ROI calculation
            olb_revenue, olb_revenue_count, olb_revenue_df = get_ppc_olb_revenue(sf, start_date, end_date)
            rtc_revenue, rtc_revenue_count, rtc_revenue_df = get_ppc_rtc_revenue(sf, start_date, end_date)
            
            total_ppc_revenue = olb_revenue + rtc_revenue
            total_revenue_records = olb_revenue_count + rtc_revenue_count
            
        # Process Google Ads data if uploaded
        df_ads = pd.DataFrame()
        ads_summary = {}
        roi = 0
        roas = 0
        if uploaded_file:
            df_ads = process_google_ads_data(uploaded_file, start_date, end_date)
            if not df_ads.empty:
                ads_summary = get_google_ads_summary(df_ads)
                # Calculate ROI
                ppc_spend = ads_summary.get('Total_Cost', 0)
                if ppc_spend > 0:
                    roi = calculate_roi(total_ppc_revenue, ppc_spend)
        
        # Calculate ROAS (Revenue per Booking)
        if kpis['Total PPC Bookings'] > 0:
            roas = calculate_roas(total_ppc_revenue, kpis['Total PPC Bookings'])
                
        # Display KPIs
        st.subheader("⚡ Key Performance Indicators")
        
        # First row of KPIs - Salesforce metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("All Calls", f"{kpis['All Calls']:,}")
        with col2:
            st.metric("PPC Calls", f"{kpis['PPC Calls']:,}")
        with col3:
            st.metric("PPC Booking Opps", f"{kpis['PPC Booking Opps']:,}")
        with col4:
            st.metric("PPC Calls Booked", f"{kpis['PPC Calls Booked']:,}")
        with col5:
            st.metric("PPC New Clients", f"{kpis['PPC New Clients']:,}")
        
        # Second row - Mixed metrics including revenue and ROI
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("PPC OLB", f"{kpis['PPC OLB']:,}")
        with col2:
            st.metric("Total PPC Bookings", f"{kpis['Total PPC Bookings']:,}")
        with col3:
            st.metric("Total PPC Revenue", f"£{total_ppc_revenue:,.2f}")
        with col4:
            if ads_summary and total_ppc_revenue > 0 and ads_summary.get('Total_Cost', 0) > 0:
                st.metric("ROI", f"{roi:.2f}", help="Return on Investment: Revenue ÷ Spend (e.g., 2.5 = £2.50 generated per £1 spent)")
            else:
                if not ads_summary:
                    st.metric("ROI", "No Google Ads data", help="Upload Google Ads data to calculate ROI")
                elif total_ppc_revenue <= 0:
                    st.metric("ROI", "No revenue data", help="No PPC revenue found for the selected period")
                elif ads_summary.get('Total_Cost', 0) <= 0:
                    st.metric("ROI", "No spend data", help="No Google Ads spend found")
                else:
                    st.metric("ROI", "Unable to calculate", help="Check data availability")
        
        # Third row - Revenue breakdown and conversion metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("OLB Revenue", f"£{olb_revenue:,.2f}", help=f"From {olb_revenue_count} completed work orders")
        with col2:
            st.metric("RTC Revenue", f"£{rtc_revenue:,.2f}", help=f"From {rtc_revenue_count} booked calls")
        with col3:
            conversion_rate = (kpis['PPC Calls Booked'] / kpis['PPC Calls'] * 100) if kpis['PPC Calls'] > 0 else 0
            st.metric("Call Conversion", f"{conversion_rate:.1f}%")
        with col4:
            if total_ppc_revenue > 0 and kpis['Total PPC Bookings'] > 0:
                st.metric("ROAS", f"£{roas:.2f}", help="Revenue per Booking: Total Revenue ÷ Total Bookings")
            else:
                st.metric("ROAS", "No data", help="Need both revenue and booking data to calculate ROAS")
        
        # Fourth row - Google Ads cost metrics
        if ads_summary:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Cost per Click", f"£{ads_summary['Cost_Per_Click']:.2f}")
            with col2:
                st.metric("Cost per Conversion", f"£{ads_summary['Cost_Per_Conversion']:.2f}")
            with col3:
                cost_per_booking = ads_summary['Total_Cost'] / kpis['Total PPC Bookings'] if kpis['Total PPC Bookings'] > 0 else 0
                st.metric("Cost per Booking", f"£{cost_per_booking:.2f}")
            with col4:
                # Calculate efficiency ratio
                if kpis['Total PPC Bookings'] > 0:
                    efficiency = ads_summary['Total_Conversions'] / kpis['Total PPC Bookings']
                    st.metric("Ads to SF Efficiency", f"{efficiency:.2f}:1")
                else:
                    st.metric("Ads to SF Efficiency", "N/A")
        
        # Google Ads specific metrics (if data available)
        if ads_summary:
            st.markdown("---")
            st.subheader("Google Ads Performance")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Impressions", f"{ads_summary['Total_Impressions']:,}")
            with col2:
                st.metric("Clicks", f"{ads_summary['Total_Clicks']:,}")
            with col3:
                st.metric("Conversions", f"{ads_summary['Total_Conversions']:,}")
            with col4:
                st.metric("CTR", f"{ads_summary['Overall_CTR']:.2f}%")
            with col5:
                st.metric("Total Spend", f"£{ads_summary['Total_Cost']:,.2f}")
        
        # ROI and ROAS Analysis Section
        if total_ppc_revenue > 0:
            st.markdown("---")
            st.subheader("📈 ROI & ROAS Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Revenue Breakdown:**")
                revenue_data = {
                    'Source': ['OLB Revenue', 'RTC Revenue', 'Total'],
                    'Amount': [f"£{olb_revenue:,.2f}", f"£{rtc_revenue:,.2f}", f"£{total_ppc_revenue:,.2f}"],
                    'Records': [olb_revenue_count, rtc_revenue_count, total_revenue_records]
                }
                revenue_df = pd.DataFrame(revenue_data)
                st.dataframe(revenue_df, use_container_width=True, hide_index=True)
            
            with col2:
                if ads_summary:
                    st.write("**ROI Calculation:**")
                    ppc_spend = ads_summary['Total_Cost']
                    st.write(f"Revenue: £{total_ppc_revenue:,.2f}")
                    st.write(f"Spend: £{ppc_spend:,.2f}")
                    st.write(f"ROI: {roi:.2f}")
                    
                    # ROI interpretation
                    if roi >= 3.0:
                        st.success(f"🎉 Excellent ROI! Every £1 spent generates £{roi:.2f}")
                    elif roi >= 2.0:
                        st.success(f"✅ Great ROI! Every £1 spent generates £{roi:.2f}")
                    elif roi >= 1.0:
                        st.success(f"✅ Positive ROI! Breaking even and generating £{roi:.2f} per £1 spent")
                    elif roi >= 0.5:
                        st.warning(f"⚠️ Low ROI - Only generating £{roi:.2f} per £1 spent")
                    else:
                        st.error(f"❌ Negative ROI - Only generating £{roi:.2f} per £1 spent")
                else:
                    st.info("Upload Google Ads data to see ROI calculation")
            
            with col3:
                st.write("**ROAS Calculation:**")
                st.write(f"Revenue: £{total_ppc_revenue:,.2f}")
                st.write(f"Total Bookings: {kpis['Total PPC Bookings']}")
                st.write(f"ROAS: £{roas:.2f}")
                
                # ROAS interpretation
                if roas >= 500:
                    st.success(f"🎉 Excellent! £{roas:.2f} revenue per booking")
                elif roas >= 300:
                    st.success(f"✅ Great! £{roas:.2f} revenue per booking")
                elif roas >= 150:
                    st.success(f"✅ Good! £{roas:.2f} revenue per booking")
                elif roas >= 50:
                    st.warning(f"⚠️ Low revenue per booking: £{roas:.2f}")
                else:
                    st.error(f"❌ Very low revenue per booking: £{roas:.2f}")
        
        # Summary table
        st.markdown("---")
        st.subheader("📋 Summary Table")
        
        # Create comprehensive summary including Google Ads data and revenue
        summary_data = []
        
        # Add Salesforce metrics
        for metric, count in kpis.items():
            summary_data.append({"Metric": metric, "Count": count})
        
        # Add revenue metrics
        summary_data.extend([
            {"Metric": "OLB Revenue", "Count": f"£{olb_revenue:,.2f}"},
            {"Metric": "RTC Revenue", "Count": f"£{rtc_revenue:,.2f}"},
            {"Metric": "Total PPC Revenue", "Count": f"£{total_ppc_revenue:,.2f}"},
            {"Metric": "Revenue Records", "Count": total_revenue_records},
            {"Metric": "ROAS (Revenue per Booking)", "Count": f"£{roas:.2f}"}
        ])
        
        # Add Google Ads metrics if available
        if ads_summary:
            summary_data.extend([
                {"Metric": "GA Impressions", "Count": ads_summary['Total_Impressions']},
                {"Metric": "GA Clicks", "Count": ads_summary['Total_Clicks']},
                {"Metric": "GA Conversions", "Count": ads_summary['Total_Conversions']},
                {"Metric": "GA Total Spend", "Count": f"£{ads_summary['Total_Cost']:.2f}"},
                {"Metric": "GA CTR", "Count": f"{ads_summary['Overall_CTR']:.2f}%"},
                {"Metric": "GA Conversion Rate", "Count": f"{ads_summary['Overall_Conversion_Rate']:.2f}%"},
                {"Metric": "GA Cost per Click", "Count": f"£{ads_summary['Cost_Per_Click']:.2f}"},
                {"Metric": "GA Cost per Conversion", "Count": f"£{ads_summary['Cost_Per_Conversion']:.2f}"},
                {"Metric": "ROI", "Count": f"{roi:.2f}"}
            ])
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Download button
        csv = summary_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Summary CSV",
            data=csv.encode("utf-8"),
            file_name=f"ppc_summary_with_roi_{start_date}_{end_date}.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("📅 Daily Performance Breakdown")
        
        if date_mode == "Single Day":
            # For single day, show hourly breakdown or detailed metrics
            st.info(f"Detailed breakdown for {selected_date.strftime('%B %d, %Y')}")
            
            with st.spinner("Fetching detailed data..."):
                # Get the single day's metrics using individual queries for accuracy
                single_day_metrics = {}
                
                # Get All Calls for the day
                try:
                    date_str = selected_date.strftime('%Y-%m-%d')
                    all_calls_query = f"""
                        SELECT COUNT(Id) 
                        FROM RTap__ResponseTap_Call__c 
                        WHERE Call_Date__c = {date_str}
                    """
                    result = sf.query(all_calls_query)
                    single_day_metrics['All Calls'] = result['records'][0].get('expr0', 0) if result['records'] else 0
                except:
                    single_day_metrics['All Calls'] = 0
                
                # Get PPC Calls for the day
                try:
                    ppc_calls_query = f"""
                        SELECT COUNT(Id) 
                        FROM RTap__ResponseTap_Call__c 
                        WHERE Call_Date__c = {date_str}
                        AND RTap__Medium__c = 'Paid Search'
                    """
                    result = sf.query(ppc_calls_query)
                    single_day_metrics['PPC Calls'] = result['records'][0].get('expr0', 0) if result['records'] else 0
                except:
                    single_day_metrics['PPC Calls'] = 0
                
                # Get PPC Booking Opps for the day
                try:
                    booking_opps_query = f"""
                        SELECT COUNT(Id) 
                        FROM RTap__ResponseTap_Call__c 
                        WHERE Call_Date__c = {date_str}
                        AND RTap__Medium__c = 'Paid Search'
                        AND Predicted_Type_of_Call__c LIKE '%Booking Opportunity%'
                    """
                    result = sf.query(booking_opps_query)
                    single_day_metrics['PPC Booking Opps'] = result['records'][0].get('expr0', 0) if result['records'] else 0
                except:
                    single_day_metrics['PPC Booking Opps'] = 0
                
                # Get PPC Calls Booked for the day
                try:
                    calls_booked_query = f"""
                        SELECT COUNT(Id) 
                        FROM RTap__ResponseTap_Call__c 
                        WHERE Call_Date__c = {date_str}
                        AND RTap__Medium__c = 'Paid Search'
                        AND Predicted_Summary__c = 'Booked'
                    """
                    result = sf.query(calls_booked_query)
                    single_day_metrics['PPC Calls Booked'] = result['records'][0].get('expr0', 0) if result['records'] else 0
                except:
                    single_day_metrics['PPC Calls Booked'] = 0
                
                # Get PPC OLB for the day - FIXED
                single_day_metrics['PPC OLB'] = get_single_day_olb_count(sf, selected_date)
                
                # Get PPC New Clients for the day
                new_clients = get_new_clients_count(sf, selected_date)
                single_day_metrics['PPC New Clients'] = new_clients['OLB_New_Clients'] + new_clients['Call_New_Clients']
                
                # Calculate total bookings
                single_day_metrics['Total PPC Bookings'] = single_day_metrics['PPC Calls Booked'] + single_day_metrics['PPC OLB']
                
                # Display metrics for the selected day
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("All Calls", f"{single_day_metrics['All Calls']:,}")
                    st.metric("PPC Calls", f"{single_day_metrics['PPC Calls']:,}")
                
                with col2:
                    st.metric("PPC Booking Opps", f"{single_day_metrics['PPC Booking Opps']:,}")
                    st.metric("PPC Calls Booked", f"{single_day_metrics['PPC Calls Booked']:,}")
                
                with col3:
                    st.metric("PPC OLB", f"{single_day_metrics['PPC OLB']:,}")
                    st.metric("PPC New Clients", f"{single_day_metrics['PPC New Clients']:,}")
                
                st.markdown("---")
                
                # Try to get hourly breakdown for calls
                try:
                    hourly_query = f"""
                        SELECT 
                            HOUR_IN_DAY(Call_Time__c) hour,
                            COUNT(Id) call_count
                        FROM RTap__ResponseTap_Call__c
                        WHERE Call_Date__c = {selected_date.strftime('%Y-%m-%d')}
                        AND RTap__Medium__c = 'Paid Search'
                        GROUP BY HOUR_IN_DAY(Call_Time__c)
                        ORDER BY HOUR_IN_DAY(Call_Time__c)
                    """
                    
                    result = sf.query(hourly_query)
                    if result['records']:
                        hourly_df = pd.DataFrame(result['records'])
                        if 'attributes' in hourly_df.columns:
                            hourly_df = hourly_df.drop('attributes', axis=1)
                        
                        # Create hourly chart
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=hourly_df['hour'],
                            y=hourly_df['call_count'],
                            name='PPC Calls by Hour',
                            marker_color='#4F46E5'
                        ))
                        
                        fig.update_layout(
                            title=f'Hourly PPC Call Distribution - {selected_date.strftime("%B %d, %Y")}',
                            xaxis_title='Hour of Day',
                            yaxis_title='Number of Calls',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                except:
                    # If hourly doesn't work, show the summary
                    st.write("### Summary for Selected Date")
                    summary_data = {
                        'Metric': list(single_day_metrics.keys()),
                        'Count': list(single_day_metrics.values())
                    }
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                
                # Download option for single day
                single_day_df = pd.DataFrame([single_day_metrics])
                csv = single_day_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Day Summary",
                    data=csv.encode("utf-8"),
                    file_name=f"ppc_summary_{selected_date.strftime('%Y-%m-%d')}.csv",
                    mime="text/csv"
                )
                    
        else:
            # Original date range functionality
            with st.spinner("Fetching daily data..."):
                daily_df = get_daily_breakdown(sf, start_date, end_date)
            
            if not daily_df.empty:
                # Display daily data table
                st.dataframe(daily_df, use_container_width=True)
                
                # Download daily data
                csv = daily_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Daily Breakdown CSV",
                    data=csv.encode("utf-8"),
                    file_name=f"ppc_daily_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )
                
                # Daily trends chart
                st.markdown("---")
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=daily_df['Date'], 
                    y=daily_df['PPC Calls'],
                    mode='lines+markers',
                    name='PPC Calls',
                    line=dict(color='blue', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=daily_df['Date'], 
                    y=daily_df['PPC Booking Opps'],
                    mode='lines+markers',
                    name='PPC Booking Opps',
                    line=dict(color='orange', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=daily_df['Date'], 
                    y=daily_df['Total PPC Bookings'],
                    mode='lines+markers',
                    name='Total PPC Bookings',
                    line=dict(color='green', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=daily_df['Date'], 
                    y=daily_df['PPC New Clients'],
                    mode='lines+markers',
                    name='PPC New Clients',
                    line=dict(color='purple', width=2)
                ))
                
                fig.update_layout(
                    title='Daily PPC Performance Trends',
                    xaxis_title='Date',
                    yaxis_title='Count',
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No daily data available for the selected period")
    
    with tab3:
        st.subheader("📊 Performance Visualizations")
        
        if 'kpis' in locals():
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart of call distribution
                call_data = {
                    'Type': ['PPC Calls', 'Non-PPC Calls'],
                    'Count': [kpis['PPC Calls'], kpis['All Calls'] - kpis['PPC Calls']]
                }
                df_pie = pd.DataFrame(call_data)
                fig = px.pie(df_pie, values='Count', names='Type', 
                           title='Call Distribution', hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
            
            import plotly.express as px

            with col2:
                with st.spinner("Fetching funnel data..."):
                    not_rejected_count = get_not_rejected_bookings(sf, start_date, end_date)

                # Breakdown values
                ppc_calls = kpis['PPC Calls']
                ppc_olbs = kpis['PPC OLB']
                combined_calls_olbs = ppc_calls + ppc_olbs

                # Real dynamic values
                real_numbers = [
                    ads_summary.get('Total_Impressions', 0) if ads_summary else 0,
                    ads_summary.get('Total_Clicks', 0) if ads_summary else 0,
                    combined_calls_olbs,
                    kpis['PPC Booking Opps'],
                    kpis['Total PPC Bookings'],
                    not_rejected_count
                ]

                # Static widths just for funnel shape (uniform decreasing steps)
                static_widths = [100, 90, 80, 70, 60, 50]

                # Build funnel data
                data = dict(
                    number=static_widths,  # for shape only
                    stage=[
                        "Impressions",
                        "Clicks",
                        f"PPC Calls: {ppc_calls} | PPC OLBs: {ppc_olbs}",
                        "Booking Opps",
                        "Booked",
                        "Not Rejected"
                    ],
                    label=[f"{val:,}" for val in real_numbers]  # display real values
                )

                # Funnel chart
                fig = px.funnel(
                    data,
                    x="number",
                    y="stage",
                    text="label"  # display real dynamic values
                )

                fig.update_traces(textposition="inside", textfont_size=14, texttemplate="%{text}")

                fig.update_layout(
                    title="PPC Conversion Funnel (Static Shape)",
                    height=500,
                    font=dict(size=12),
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

                
                # Show breakdown text
                st.write(f"**Breakdown:** {kpis['PPC Calls']} calls + {kpis['PPC OLB']} OLBs = {combined_calls_olbs} total")
            
            # KEY METRICS SUMMARY
            st.markdown("---")
            st.subheader("📈 Key Performance Metrics")
            
            # Calculate conversion rate
            conversion_rate = (kpis['Total PPC Bookings'] / kpis['PPC Booking Opps'] * 100) if kpis['PPC Booking Opps'] > 0 else 0
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                total_spend = ads_summary.get('Total_Cost', 0) if ads_summary else 0
                st.metric("Total Spend", f"£{total_spend:,.2f}")
            
            with col2:
                st.metric("Total PPC New Clients", f"{kpis['PPC New Clients']:,}")
            
            with col3:
                st.metric("Total Booking Count", f"{kpis['Total PPC Bookings']:,}")
            
            with col4:
                st.metric("Conversion Rate", f"{conversion_rate:.1f}%", 
                         help="(Booked / Booking Opps) * 100")
            
            with col5:
                if 'total_ppc_revenue' in locals():
                    st.metric("Total Revenue", f"£{total_ppc_revenue:,.2f}")
                else:
                    st.metric("Total Revenue", "£0.00")
    
    with tab4:
        st.subheader("🎯 Google Ads Performance Analysis")
        
        if not df_ads.empty:
            # Daily breakdown chart
            ads_daily = get_google_ads_daily_breakdown(df_ads)
            
            # Create metrics over time chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=ads_daily['Date'], 
                y=ads_daily['Impressions'],
                mode='lines+markers',
                name='Impressions',
                yaxis='y',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=ads_daily['Date'], 
                y=ads_daily['Clicks'],
                mode='lines+markers',
                name='Clicks',
                yaxis='y2',
                line=dict(color='green', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=ads_daily['Date'], 
                y=ads_daily['Conversions'],
                mode='lines+markers',
                name='Conversions',
                yaxis='y2',
                line=dict(color='red', width=2)
            ))
            
            # Create subplot with secondary y-axis
            fig.update_layout(
                title='Google Ads Performance Over Time',
                xaxis=dict(title='Date'),
                yaxis=dict(title='Impressions', side='left', color='blue'),
                yaxis2=dict(title='Clicks & Conversions', side='right', overlaying='y', color='green'),
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # TOP 5 BEST CAMPAIGNS by Conversions
            if 'Campaign' in df_ads.columns:
                st.markdown("---")
                st.subheader("🏆 Top 5 Best Campaigns (by Conversions)")
                
                campaign_summary = df_ads.groupby('Campaign').agg({
                    'Impressions': 'sum',
                    'Clicks': 'sum',
                    'Conversions': 'sum',
                    'Cost': 'sum'
                }).reset_index()
                
                # Calculate metrics
                campaign_summary['CTR'] = (campaign_summary['Clicks'] / campaign_summary['Impressions'] * 100).round(2)
                campaign_summary['Conversion_Rate'] = (campaign_summary['Conversions'] / campaign_summary['Clicks'] * 100).round(2)
                campaign_summary['Cost_Per_Conversion'] = (campaign_summary['Cost'] / campaign_summary['Conversions']).round(2)
                
                # Replace infinite values
                campaign_summary = campaign_summary.replace([float('inf'), -float('inf')], 0)
                
                # Sort by conversions descending and take top 5
                top_5_campaigns = campaign_summary.sort_values('Conversions', ascending=False).head(5)
                
                st.dataframe(top_5_campaigns, use_container_width=True)
                
                # Top 5 campaigns performance chart
                fig = px.scatter(top_5_campaigns, 
                               x='Cost', 
                               y='Conversions',
                               size='Clicks',
                               hover_name='Campaign',
                               title='Top 5 Campaigns: Cost vs Conversions (Bubble size = Clicks)')
                st.plotly_chart(fig, use_container_width=True)
            
            # Cost efficiency analysis
            st.markdown("---")
            st.subheader("💰 Cost Efficiency Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Daily cost trend
                fig = px.line(ads_daily, x='Date', y='Cost', 
                             title='Daily Spend Trend',
                             markers=True)
                fig.update_traces(line_color='orange', line_width=3)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Cost per conversion trend
                fig = px.line(ads_daily, x='Date', y='Cost_Per_Conversion', 
                             title='Cost per Conversion Trend',
                             markers=True)
                fig.update_traces(line_color='red', line_width=3)
                st.plotly_chart(fig, use_container_width=True)
            
            # Download Google Ads data
            st.markdown("---")
            csv_ads = df_ads.to_csv(index=False)
            st.download_button(
                "⬇️ Download Google Ads Data",
                data=csv_ads.encode("utf-8"),
                file_name=f"google_ads_data_{start_date}_{end_date}.csv",
                mime="text/csv"
            )
            
        else:
            st.info("Upload Google Ads data to see detailed analysis")
            st.markdown("""
            **Expected data format:**
            - Day (date column)
            - Campaign 
            - Impressions (Impr.)
            - Clicks
            - Conversions
            - Cost
            - CTR (optional, will be calculated)
            """)
    
    with tab5:
        st.subheader("📈 Campaign Analysis")
        
        # Date selection for campaign analysis
        if date_mode == "Single Day":
            analysis_date = selected_date
            st.info(f"Analyzing campaigns for: {analysis_date.strftime('%B %d, %Y')}")
        else:
            st.info("Campaign Analysis works with single day data. Analyzing latest date from your range.")
            analysis_date = end_date
            st.info(f"Analyzing campaigns for: {analysis_date.strftime('%B %d, %Y')}")
        
        # Fetch campaign data
        with st.spinner("Fetching campaign data from Salesforce..."):
            try:
                # Get RTC (ResponseTap Calls) data
                rtc_df = get_campaign_rtc_data(sf, analysis_date)
                st.success(f"Fetched {len(rtc_df)} RTC records")
                
                # Get OLB (Online Bookings) data  
                olb_df = get_campaign_olb_data(sf, analysis_date)
                st.success(f"Fetched {len(olb_df)} OLB records")
                
            except Exception as e:
                st.error(f"Error fetching campaign data: {str(e)}")
                rtc_df = pd.DataFrame()
                olb_df = pd.DataFrame()
        
        # Display the fetched data
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📞 RTC (ResponseTap Calls) Data")
            if not rtc_df.empty:
                st.write(f"**Records found:** {len(rtc_df)}")
                
                # Show summary stats
                if 'CCT_Charge_NET' in rtc_df.columns:
                    total_rtc_revenue = rtc_df['CCT_Charge_NET'].sum()
                    avg_rtc_revenue = rtc_df['CCT_Charge_NET'].mean()
                    st.metric("Total RTC Revenue", f"£{total_rtc_revenue:,.2f}")
                    st.metric("Average RTC Revenue", f"£{avg_rtc_revenue:.2f}")
                
                # Show campaigns breakdown
                if 'RTap_Campaign' in rtc_df.columns:
                    st.write("**Campaigns in RTC Data:**")
                    campaign_counts = rtc_df['RTap_Campaign'].value_counts()
                    st.write(campaign_counts)
                
                # Download button
                csv_rtc = rtc_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download RTC Data",
                    data=csv_rtc.encode("utf-8"),
                    file_name=f"rtc_campaign_data_{analysis_date.strftime('%Y-%m-%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No RTC data found for the selected date")
        
        with col2:
            st.subheader("🌐 OLB (Online Bookings) Data")
            if not olb_df.empty:
                st.write(f"**Records found:** {len(olb_df)}")
                
                # Show summary stats
                if 'CCT_Charge_NET' in olb_df.columns:
                    total_olb_revenue = olb_df['CCT_Charge_NET'].sum()
                    avg_olb_revenue = olb_df['CCT_Charge_NET'].mean()
                    st.metric("Total OLB Revenue", f"£{total_olb_revenue:,.2f}")
                    st.metric("Average OLB Revenue", f"£{avg_olb_revenue:.2f}")
                
                # Show campaigns breakdown
                if 'AFL_UTM_Campaign__c' in olb_df.columns:
                    st.write("**Campaigns in OLB Data:**")
                    campaign_counts = olb_df['AFL_UTM_Campaign__c'].value_counts()
                    st.write(campaign_counts)
                
                # Download button
                csv_olb = olb_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download OLB Data",
                    data=csv_olb.encode("utf-8"),
                    file_name=f"olb_campaign_data_{analysis_date.strftime('%Y-%m-%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No OLB data found for the selected date")
        
        # Data Structure Information
        st.markdown("---")
        with st.expander("🔍 Data Structure Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**RTC Data Structure:**")
                if not rtc_df.empty:
                    st.write("Columns:", rtc_df.columns.tolist())
                    st.write("Shape:", rtc_df.shape)
                else:
                    st.write("No RTC data available")
            
            with col2:
                st.write("**OLB Data Structure:**")
                if not olb_df.empty:
                    st.write("Columns:", olb_df.columns.tolist())
                    st.write("Shape:", olb_df.shape)
                else:
                    st.write("No OLB data available")
    
    with tab6:
        st.subheader("🔍 Query Diagnostics")
        
        # Show the actual queries being used
        with st.expander("View SOQL Queries"):
            st.json({
                "Instance": instance,
                "Date Range": {
                    "Start": start_date.strftime('%Y-%m-%d'),
                    "End": end_date.strftime('%Y-%m-%d')
                },
                "Objects Used": {
                    "Calls": "RTap__ResponseTap_Call__c",
                    "Online Bookings": "Online_Bookings__c",
                    "Work Orders": "WorkOrder"
                },
                "Fields": {
                    "Call Date": "Call_Date__c",
                    "Medium": "RTap__Medium__c",
                    "Call Type": "Predicted_Type_of_Call__c",
                    "Summary": "Predicted_Summary__c",
                    "OLB Medium": "AFL_UTM_Medium__c"
                }
            })
        
        # Test queries section
        st.markdown("---")
        if st.checkbox("Run Custom Test Query"):
            custom_query = st.text_area(
                "Enter SOQL Query",
                height=100
            )
            
            if st.button("Execute Query"):
                try:
                    result = sf.query(custom_query)
                    df_result = pd.DataFrame(result['records'])
                    if 'attributes' in df_result.columns:
                        df_result = df_result.drop('attributes', axis=1)
                    st.success(f"Found {len(df_result)} records")
                    st.dataframe(df_result)
                except Exception as e:
                    st.error(f"Query failed: {str(e)}")

else:
    st.error("❌ Failed to connect to Salesforce. Please check your credentials in `.streamlit/secrets.toml`")