import io
import streamlit as st
import pandas as pd
import boto3
import time
from datetime import datetime
import plotly.express as px
import joblib

# ==========================================
# CONFIGURATION (IF MADE CHANGES REPLACE WITH THE NAME YOU MADE)
# ==========================================

ATHENA_DB = "security_data_lake"
ATHENA_TABLE = "analytics"
OUTPUT_S3_BUCKET = "s3://security-data-lake-1.0/athena-results/"
REGION = "us-east-1"

LOG_GROUP = "ThreatDashboardLogs"

# ==========================================
# AWS CLIENTS
# ==========================================

athena_client = boto3.client("athena", region_name=REGION)
s3_client = boto3.client("s3", region_name=REGION)
cw_client = boto3.client("logs", region_name=REGION)

# ==========================================
# CLOUDWATCH
# ==========================================

def write_cloudwatch_log(stream_name, message):
    try:
        timestamp = int(round(time.time() * 1000))

        kwargs = {
            "logGroupName": LOG_GROUP,
            "logStreamName": stream_name,
            "logEvents": [
                {
                    "timestamp": timestamp,
                    "message": message
                }
            ]
        }

        try:
            response = cw_client.describe_log_streams(
                logGroupName=LOG_GROUP,
                logStreamNamePrefix=stream_name
            )

            streams = response.get("logStreams", [])
            if streams and "uploadSequenceToken" in streams[0]:
                kwargs["sequenceToken"] = streams[0]["uploadSequenceToken"]

        except Exception:
            pass

        cw_client.put_log_events(**kwargs)

    except Exception as e:
        print(f"CloudWatch logging failed: {e}")

# Log dashboard access safely
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
write_cloudwatch_log(
    "AccessLogs",
    f"Dashboard accessed via web browser at {current_time}"
)

def get_recent_logs(log_stream_name, limit=5):
    """Fetches the most recent logs from AWS CloudWatch."""
    try:
        response = cw_client.filter_log_events(
            logGroupName=LOG_GROUP,
            logStreamNames=[log_stream_name],
            limit=limit
        )

        fetched_logs = []

        for event in response.get("events", []):
            dt_object = datetime.fromtimestamp(event["timestamp"] / 1000.0)

            fetched_logs.append({
                "time": dt_object.strftime("%Y-%m-%d %H:%M:%S"),
                "message": event["message"]
            })

        return list(reversed(fetched_logs))

    except Exception as e:
        return [{
            "time": "System Error",
            "message": f"Could not load logs: {str(e)}"
        }]

# ==========================================
# LOAD MACHINE LEARNING MODELS
# ==========================================

@st.cache_resource
def load_models():
    natural_model = joblib.load("models/xgboost_natural.pkl")
    balanced_model = joblib.load("models/xgboost_balanced.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return natural_model, balanced_model, label_encoder

natural_model, balanced_model, le = load_models()

# ==========================================
# ATHENA QUERY FUNCTION
# ==========================================

def run_athena_query(query):
    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": ATHENA_DB},
        ResultConfiguration={"OutputLocation": OUTPUT_S3_BUCKET}
    )

    query_execution_id = response["QueryExecutionId"]

    while True:
        query_status = athena_client.get_query_execution(
            QueryExecutionId=query_execution_id
        )

        status = query_status["QueryExecution"]["Status"]["State"]

        if status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            break

        time.sleep(1)

    if status == "SUCCEEDED":
        raw_bucket = OUTPUT_S3_BUCKET.replace("s3://", "").split("/")[0]
        prefix = "/".join(OUTPUT_S3_BUCKET.replace("s3://", "").split("/")[1:])

        if prefix.endswith("/"):
            s3_key = f"{prefix}{query_execution_id}.csv"
        else:
            s3_key = f"{prefix}/{query_execution_id}.csv"

        obj = s3_client.get_object(Bucket=raw_bucket, Key=s3_key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))

    else:
        error_message = f"Athena query failed with status: {status}"
        st.error(error_message)
        write_cloudwatch_log("ErrorLogs", error_message)
        return pd.DataFrame()

# ==========================================
# STREAMLIT DASHBOARD UI
# ==========================================

st.set_page_config(page_title="ML Analytics Dashboard", layout="wide")

st.title("Serverless Security Data Lake")

tab1, tab2, tab3, tab4= st.tabs([
    "Attack Analytics",
    "ML Predictor",
    "ML Model Performance and Info",
    "CloudWatch Logs"
])

# ==========================================
# TAB 1: ATTACK ANALYTICS
# ==========================================

with tab1:
    st.markdown("Querying network traffic logs using AWS Athena & Parquet")

    if st.sidebar.button("Query network logs"):
        st.sidebar.success("Querying Athena Engine...")

        try:
            q_percent = f'''
            SELECT 
                ROUND(100.0 * SUM(CASE WHEN label <> 'Benign' THEN 1 ELSE 0 END) / COUNT(*), 2) AS malicious_percent,
                COUNT(*) AS total_logs
            FROM "{ATHENA_DB}"."{ATHENA_TABLE}";
            '''

            df_percent = run_athena_query(q_percent)

            q_attacks = f'''
            SELECT label, COUNT(*) AS attack_count
            FROM "{ATHENA_DB}"."{ATHENA_TABLE}"
            WHERE label <> 'Benign'
            GROUP BY label
            ORDER BY attack_count DESC
            LIMIT 10;
            '''

            df_attacks = run_athena_query(q_attacks)

            q_timeline = f'''
            SELECT
                DATE_TRUNC('hour', CAST(timestamp AS TIMESTAMP)) AS attack_hour,
                COUNT(*) AS attack_count
            FROM "{ATHENA_DB}"."{ATHENA_TABLE}"
            WHERE label <> 'Benign'
            GROUP BY 1
            ORDER BY attack_hour ASC;
            '''

            df_timeline = run_athena_query(q_timeline)

            q_ports = f'''
            SELECT "dst port" AS dst_port, COUNT(*) AS attack_count
            FROM "{ATHENA_DB}"."{ATHENA_TABLE}"
            WHERE label <> 'Benign'
            GROUP BY "dst port"
            ORDER BY attack_count DESC
            LIMIT 10;
            '''

            df_ports = run_athena_query(q_ports)

            # CloudWatch threat logging AFTER df_attacks exists
            total_threats = df_attacks["attack_count"].sum() if not df_attacks.empty else 0

            if total_threats > 10000:
                write_cloudwatch_log(
                    "ThreatLogs",
                    f"CRITICAL: High volume of threats detected ({total_threats} incidents) at {current_time}"
                )

            # KPI cards
            if not df_percent.empty:
                m_col1, m_col2, m_col3 = st.columns(3)

                with m_col1:
                    total_events = int(df_percent["total_logs"].iloc[0])
                    st.metric("Total Network Logs Analyzed", f"{total_events:,}")

                with m_col2:
                    mal_pct = df_percent["malicious_percent"].iloc[0]
                    st.metric("Malicious Traffic", f"{mal_pct}%")

                with m_col3:
                    st.metric("Total Malicious Traffic", f"{int(total_threats):,}")

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                if not df_attacks.empty:
                    fig_pie = px.pie(
                        df_attacks,
                        values="attack_count",
                        names="label",
                        title="Malicious Traffic Donut",
                        hole=0.4,
                        color_discrete_sequence=px.colors.sequential.Reds_r
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                if not df_ports.empty:
                    df_ports["dst_port"] = df_ports["dst_port"].astype(str)
                    unique_ports = df_ports["dst_port"].tolist()

                    fig_port_bar = px.bar(
                        df_ports,
                        x="attack_count",
                        y="dst_port",
                        orientation="h",
                        title="Most Targeted Ports",
                        labels={
                            "attack_count": "Number of Attacks",
                            "dst_port": "Port ID"
                        },
                        color="attack_count",
                        color_continuous_scale="YlOrRd"
                    )

                    fig_port_bar.update_layout(
                        yaxis={
                            "type": "category",
                            "categoryorder": "array",
                            "categoryarray": unique_ports[::-1]
                        }
                    )

                    st.plotly_chart(fig_port_bar, use_container_width=True)

            st.markdown("---")

            if not df_timeline.empty:
                st.subheader("Incident Detection Timeline")

                fig_line = px.line(
                    df_timeline,
                    x="attack_hour",
                    y="attack_count",
                    title="Hourly Distribution of Network Attacks",
                    labels={
                        "attack_hour": "Time of Day Hourly",
                        "attack_count": "Attack Count"
                    },
                    markers=True
                )

                fig_line.update_traces(line=dict(color="#DC3545", width=3))
                st.plotly_chart(fig_line, use_container_width=True)

            st.markdown("---")
            st.subheader("Network Attack Summary")

            t_col1, t_col2 = st.columns(2)

            with t_col1:
                st.markdown("**Types of Attacks**")
                st.dataframe(df_attacks, use_container_width=True)

            with t_col2:
                st.markdown("**Targeted Destination Ports**")
                st.dataframe(df_ports, use_container_width=True)

        except Exception as e:
            error_message = f"Application Error: {str(e)}"
            st.error(error_message)
            write_cloudwatch_log("ErrorLogs", error_message)

    else:
        st.info("Welcome to the Automated Serverless Security Dashboard. Click 'Query network logs' to load analytics.")

# ==========================================
# TAB 2: ML PREDICTOR
# ==========================================

with tab2:
    st.header("ML Predictor")
    st.markdown(
        "Enter network values and compare predictions between "
        "Machine Natural Sample and Machine Balanced Sample."
    )

    col1, col2 = st.columns(2)

    with col1:
        dst_port = st.number_input("Dst Port", value=80)
        protocol = st.number_input("Protocol", value=6)
        flow_duration = st.number_input("Flow Duration", value=100000)
        flow_byts = st.number_input("Flow Byts/s", value=5000.0)

    with col2:
        flow_pkts = st.number_input("Flow Pkts/s", value=100.0)
        tot_fwd = st.number_input("Tot Fwd Pkts", value=20)
        tot_bwd = st.number_input("Tot Bwd Pkts", value=18)

    if st.button("Run ML Prediction"):
        flow = pd.DataFrame([{
            "Dst Port": dst_port,
            "Protocol": protocol,
            "Flow Duration": flow_duration,
            "Flow Byts/s": flow_byts,
            "Flow Pkts/s": flow_pkts,
            "Tot Fwd Pkts": tot_fwd,
            "Tot Bwd Pkts": tot_bwd
        }])

        natural_pred = natural_model.predict(flow)[0]
        balanced_pred = balanced_model.predict(flow)[0]

        natural_label = le.inverse_transform([natural_pred])[0]
        balanced_label = le.inverse_transform([balanced_pred])[0]

        st.markdown("---")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric("Natural Model Prediction", natural_label)

        with result_col2:
            st.metric("Balanced Model Prediction", balanced_label)

        st.subheader("Input Flow Used")
        st.dataframe(flow, use_container_width=True)

# ==========================================
# TAB 3: MODEL PERFORMANCE
# ==========================================

with tab3:
    comparison_df = pd.DataFrame({
        "Metric": [
            "F1 Score",
            "Rare Attack Detection",
            "Overall Accuracy"
        ],
        "Natural Model": [
            "0.97",
            "Poor",
            "Excellent"
        ],
        "Balanced Model": [
            "0.84",
            "Good",
            "Pretty Good"
        ]
    })

    st.subheader("Model Comparison")
    st.dataframe(comparison_df, use_container_width=True)

    st.subheader("Confusion Matrices")

    col1, col2 = st.columns(2)

    with col1:
        st.image("static/natural.png", caption="Natural Model Confusion Matrix")

    with col2:
        st.image("static/balanced.png", caption="Balanced Model Confusion Matrix")
        
# ==========================================
# TAB 4: CLOUDWATCH LOGS
# ==========================================

with tab4:
    st.header("Live System Logs")
    st.markdown("Recent logs pulled from AWS CloudWatch.")

    if st.button("Refresh CloudWatch Logs"):
        access_logs = get_recent_logs("AccessLogs", limit=5)
        threat_logs = get_recent_logs("ThreatLogs", limit=5)
        error_logs = get_recent_logs("ErrorLogs", limit=5)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Access Logs")
            for log in access_logs:
                st.info(f"**{log['time']}**\n\n{log['message']}")

        with col2:
            st.subheader("Critical Threat Alerts")
            for log in threat_logs:
                st.warning(f"**{log['time']}**\n\n{log['message']}")

        with col3:
            st.subheader("Application Errors")
            for log in error_logs:
                st.error(f"**{log['time']}**\n\n{log['message']}")

    else:
        st.info("Click Refresh CloudWatch Logs to load recent system logs.")