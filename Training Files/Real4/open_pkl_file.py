import pickle

filename = '/Users/qiyangyan/Desktop/Training Files/Training3_SI/Training3/data_check.pkl'
with open(filename, 'rb') as file:
    # Load the data from the file
    data = pickle.load(file)

t_success_rate = data['t_success_rate']
running_reward_list = data['running_reward_list']

print(t_success_rate)
print(running_reward_list)