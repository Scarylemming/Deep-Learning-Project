import pandas as pd
import matplotlib.pyplot as plt

Oja_df = pd.read_csv("Datasets_DONT_TOUCH/Oja.csv")
streaks_Oja = Oja_df.mean(axis = 1)
EigenGame_df = pd.read_csv("Datasets_DONT_TOUCH/EigenGame.csv")
streaks_EigenGame = EigenGame_df.mean(axis = 1)

plt.plot(list(range(len(streaks_Oja)-1)),streaks_Oja[1:],label = "Oja")
plt.plot(list(range(len(streaks_EigenGame)-1)),streaks_EigenGame[1:], label = "EigenGame")
plt.title("Biggest eigenvector error based on number of iterations performed")
plt.xlabel("Number of iterations")
plt.ylabel("Biggest eigenvector error")
plt.legend(loc='upper right')
plt.yscale("log")
plt.show()