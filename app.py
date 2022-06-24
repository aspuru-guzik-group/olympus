import streamlit as st
from olympus import Olympus, list_planners, list_datasets, Database, Campaign, Plotter
import numpy as np
import matplotlib.pyplot as plt

st.title("🏛️ Olympus")

"""Benchmark different autonomous planning algorithms on experimentally derived 🧪 datasets!"""
# initalize the Olympus orchestrator
olymp = Olympus()

# we declare a local database to which we store our campaign results
database = Database()
from time import time
# import time

# with st.empty():
#      for seconds in range(3):
#          st.code(f"⏳ {seconds} seconds have passed")
#          time.sleep(1)
#      st.write("✔️ 1 minute over!")


with st.form('run_planners'):
    datasets = list_datasets()
    first, second, third = st.columns(3)
    with first:
        DATASET = st.selectbox("Dataset", datasets, index=datasets.index('hplc'))
    with second:
        NUM_ITER = st.number_input('Evaluations per campaign', 1, 1000, 100, 1, help="The number of measurements each planner can make during a campaign")
    with third:
        NUM_REPETITIONS = st.number_input('Number of campaigns', 1, 5, 3, 1, help="The number of *entire* campaigns to repeat for each planner")
    PLANNERS = st.multiselect("Planners", list_planners(), ['Gpyopt', 'Sobol', 'Genetic', 'RandomSearch'])
    elapsed_times = {'planner': [], 'time': []}

    submitted = st.form_submit_button('Run!')
    if submitted:
        with st.spinner("Running planners..."):
            code_output = []
            placeholder = st.empty()
            for PLANNER in PLANNERS:
                for repetition in range(NUM_REPETITIONS):
                    code_output.append(f"Algorithm: {PLANNER} [repetition {repetition+1}]")
                    placeholder.code("\n".join(code_output))

                    start_time = time()
                    olymp.run(
                        planner=PLANNER,      # run simulation with <PLANNER>,
                        dataset=DATASET,      # on emulator trained on dataset <DATASET>;
                        campaign=Campaign(),  # store results in a new campaign, 
                        database=database,    # but use the same database to store campaign;
                        num_iter=100,         # run benchmark for 100 iterations
                    )
                    elapsed_time = time() - start_time
                    elapsed_times['planner'].append(PLANNER)
                    elapsed_times['time'].append(elapsed_time)
            placeholder.empty()

        # campaigns = [campaign for campaign in database]
        # for campaign in campaigns:
        #     st.write(repr(campaign))
        with st.spinner("Generating plot..."):
            plotter = Plotter()
            plotter.plot_from_db(database)
        st.pyplot(plt.gcf())