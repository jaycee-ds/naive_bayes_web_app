import numpy as np
import pandas as pd

my_array = np.random.choice(('spam', 'not spam'), size=1000, p=(0.5, 0.5))
tags = [['ad', 'not an ad'], ['phishing', 'not phishing'], ['unknown', 'contact']]

for tag_list in range(len(tags)):
    my_array = np.column_stack((my_array, np.random.choice(tags[tag_list], size=1000, p=(0.5, 0.5))))

df = pd.DataFrame(my_array, columns=['filter', 'tag_1', 'tag_2', 'tag_3'])
df.to_csv('Generated_data.csv', index=False, mode='w')
print('A dataframe has been generated and saved to a csv file.')
