import matplotlib.pyplot as plt
import streamlit as st


def show_plot(fig):
    st.pyplot(fig)
    plt.close(fig)