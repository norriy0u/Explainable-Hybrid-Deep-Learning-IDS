"""
Dashboard Page 7: Security Alert History and Incident Response Log.
Maintains forensic audit trails, severity filters, playbook details, and CSV export.
"""

from pathlib import Path
import streamlit as st
import pandas as pd

from config.config import SEVERITY_COLORS, CLASS_NAMES
from detection.detection_history import DetectionHistoryManager

def render_alerts_page():
    st.markdown("""
    <div class="soc-header">
        <h1 class="soc-header-title">🚨 INCIDENT RESPONSE LOG & ALERT AUDIT TRAIL</h1>
        <div class="soc-header-subtitle">
            Forensic Incident Logging, Severity Triage, Defensive Playbooks, and Export
        </div>
    </div>
    """, unsafe_allow_html=True)

    history_mgr = DetectionHistoryManager()
    alerts = history_mgr.get_all()

    if not alerts:
        st.info("No security alerts currently logged. Execute detections on the **Attack Detection** page to populate.")
        return

    # 1. Alert Filters
    fcol1, fcol2, fcol3 = st.columns([1, 1, 1])
    with fcol1:
        severity_filter = st.selectbox("Filter by Severity:", ["All", "Critical", "High", "Medium", "Low", "Informational"])
    with fcol2:
        cat_filter = st.selectbox("Filter by Attack Category:", ["All"] + CLASS_NAMES)
    with fcol3:
        intrusions_only = st.checkbox("Intrusions Only (Exclude Normal)", value=False)

    filtered_alerts = history_mgr.filter_records(
        severity=severity_filter,
        attack_category=cat_filter,
        intrusions_only=intrusions_only
    )

    st.markdown(f"**Showing {len(filtered_alerts)} of {len(alerts)} Incident Records**")

    # 2. Alerts Table
    display_rows = []
    for a in filtered_alerts:
        display_rows.append({
            "Alert ID": a["alert_id"],
            "Timestamp": a["timestamp"],
            "Category": a["predicted_class"],
            "Confidence": a["confidence_pct"],
            "Severity": a["severity"],
            "Flow ID": a["flow_identifier"],
            "Playbook Action": a["defensive_playbook"][:55] + "..."
        })

    df_alerts = pd.DataFrame(display_rows)
    st.dataframe(df_alerts, use_container_width=True, hide_index=True)

    # 3. Incident Detail Inspector
    st.markdown("---")
    st.markdown("### 🔍 Incident Detail Inspector")
    if filtered_alerts:
        selected_alert_id = st.selectbox(
            "Select Incident Alert to Inspect:",
            [a["alert_id"] for a in filtered_alerts]
        )
        selected_alert = next((a for a in filtered_alerts if a["alert_id"] == selected_alert_id), None)

        if selected_alert:
            scolor = selected_alert.get("severity_color", "#3B82F6")
            st.markdown(f"""
            <div style="background-color: #111827; border: 1px solid #1F2937; border-left: 6px solid {scolor}; border-radius: 8px; padding: 20px;">
                <div style="font-size: 20px; font-weight: 800; color: #F9FAFB;">
                    {selected_alert['alert_id']} — {selected_alert['predicted_class'].upper()}
                    <span style="background-color: {scolor}; color: #FFFFFF; font-size: 12px; padding: 4px 10px; border-radius: 4px; margin-left: 12px;">
                        {selected_alert['severity']}
                    </span>
                </div>
                <div style="font-size: 13px; color: #9CA3AF; margin-top: 6px;">
                    Timestamp: <b>{selected_alert['timestamp']}</b> | 
                    Confidence: <b>{selected_alert['confidence_pct']}</b> | 
                    Flow Identifier: <code>{selected_alert['flow_identifier']}</code>
                </div>
                <div style="margin-top: 14px; padding: 12px; background: rgba(0,0,0,0.4); border-radius: 6px;">
                    <b style="color: #38BDF8;">Defensive Response Recommendation:</b><br>
                    <span style="font-size: 13px; color: #E5E7EB;">{selected_alert['defensive_playbook']}</span>
                </div>
                <div style="margin-top: 12px;">
                    <b style="color: #9CA3AF; font-size: 12px;">Explanation Snapshot:</b><br>
                    <span style="font-size: 13px; color: #D1D5DB;">{selected_alert['explanation_summary']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 4. Actions: CSV Export and Clear
    st.markdown("<br>", unsafe_allow_html=True)
    acol1, acol2 = st.columns([1, 1])
    with acol1:
        if st.button("📥 Export Alert History to CSV", use_container_width=True):
            csv_path = history_mgr.export_csv()
            with open(csv_path, "rb") as f:
                st.download_button(
                    label="Download Incident Log CSV",
                    data=f,
                    file_name="soc_incident_alert_history.csv",
                    mime="text/csv"
                )
    with acol2:
        if st.button("🗑️ Clear Incident History", use_container_width=True):
            history_mgr.clear()
            st.success("Incident history cleared.")
            st.rerun()

