
import streamlit as st

st.title("Report Page")


#import streamlit as st

# Sample data
reports = {
    105: {'date': '2022-05-17', 'result': 'PATHOLOGICAL'},
    106: {'date': '2022-05-17', 'result': 'NORMAL'},
    109: {'date': '2022-05-17', 'result': 'NORMAL'},
    113: {'date': '2022-05-17', 'result': 'NORMAL'},
    116: {'date': '2022-06-21', 'result': 'NORMAL'},
    118: {'date': '2022-06-22', 'result': 'NORMAL'},
    129: {'date': '2022-06-24', 'result': 'NORMAL'},
}

st.title("Fetal Health Reports")

selected_report_id = st.selectbox(
    "Choose a Report ID",
    list(reports.keys()),
    format_func=lambda x: '<b>{}</b>'.format(x),
)

if selected_report_id in reports:
    report = reports[selected_report_id]
    st.subheader("Report ID: **{}**".format(selected_report_id))
    st.write("Date: ", report['date'])
    st.write("Result: ", report['result'])
    st.write("---")
    st.markdown("[VIEW FULL REPORT]", unsafe_allow_html=True)