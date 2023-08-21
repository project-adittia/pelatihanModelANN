# x_normalized = (x-min_value)/(max_value-min_value)

import numpy as np
import time
from keras.models import load_model


bahan_uji = np.array([
    [17,2003,16.23,1608,114,14,8,20,201],
    [12,2002,28.13,3135,197,25,8,21,392],
    [15,2004,33.46,4121,234,29,8,20,515],
    [13,2002,14.85,1416,104,10,10,21,142],
])

for i in range(len(bahan_uji)):
    # print(bahan_uji[i])
    for j in range(len(bahan_uji[i])):
        # print(i, j)
        # print(bahan_uji[i][j])
        
        if j == 0:
            # print(bahan_uji[i][j], "blok")
            bahan_uji[i][j] = (bahan_uji[i][j] - 1) / (22 - 1)
        elif j == 1:
            # print(bahan_uji[i][j], "tahun")
            bahan_uji[i][j] = (bahan_uji[i][j] - 2001) / (2004 - 2001)
        elif j == 2:
            # print(bahan_uji[i][j], "luas")
            bahan_uji[i][j] = (bahan_uji[i][j] - 3.8) / (39.2 - 3.8)
        elif j == 3:
            # print(bahan_uji[i][j], "pokok_total")
            bahan_uji[i][j] = (bahan_uji[i][j] - 444) / (5761 - 444)
        elif j == 4:
            # print(bahan_uji[i][j], "pokok_sampel")
            bahan_uji[i][j] = (bahan_uji[i][j] - 56) / (274 - 56)
        elif j == 5:
            # print(bahan_uji[i][j], "tandan_matang")
            bahan_uji[i][j] = (bahan_uji[i][j] - 4) / (56 - 4)
        elif j == 6:
            # print(bahan_uji[i][j], "akp")
            bahan_uji[i][j] = (bahan_uji[i][j] - 3) / (20 - 3)
        elif j == 7:
            # print(bahan_uji[i][j], "rbt")
            bahan_uji[i][j] = (bahan_uji[i][j] - 20) / (22 - 20)
        elif j == 8:
            # print(bahan_uji[i][j], "total_tandan")
            bahan_uji[i][j] = (bahan_uji[i][j] - 56) / (860 - 56)
    #     bahan_uji[i][j] = (bahan_uji[i][j] - min_value[j]) / (max_value[j] - min_value[j])

print(bahan_uji)

# start_time = time.time()
# Kode untuk melakukan prediksi dengan model ANN

model = load_model('model0001_1hl6n_5.h5')
result = model.predict(bahan_uji)
print(result)

# end_time = time.time()
# waktu = end_time - start_time

# print("waktu yang dibutuhkan: ", waktu)
