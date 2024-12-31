# Imports
import os


# Paths
base_directory = os.path.dirname(os.path.abspath(__file__))
data_directory = os.path.join(base_directory, 'data')
ticker_directory = os.path.join(data_directory, 'tickers')
scripts_directory = os.path.join(base_directory, 'scripts')
environment_directory = os.path.join(scripts_directory, 'environment')

# Initialize Variables
dynamic_load_allocation = {
    'Efficient Frontier': 0.40,
    'LLM Insights': 0.10,
    'Dynamic Hedging': 0.095,
    'Long Term Diversification': 0.5,
}

# Load & Update Data
pass

# Effecient Frontier
efficient_frontier_path = os.path.join(scripts_directory, 'Efficient Frontier')
if not os.path.exists(efficient_frontier_path):
    raise "Efficent Frontier not found at path " + efficient_frontier_path

# TODO: FINISH OUTLINE BY TOMORROW 




