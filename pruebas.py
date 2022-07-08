data= [
    [i for i in range(20)],
    [i*2 for i in range(20)],
    [i/2 for i in range(20)]
]

import numpy as np
data = np.array(data)

print(data[:,-11:])