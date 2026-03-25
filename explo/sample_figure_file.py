import matplotlib.pyplot as plt
import numpy as np

# Configuration de matplotlib
import matplotlib
# Ajuster la taille de police
font = {'family' : 'sans',
        'size'   : 12}

matplotlib.rc('font', **font)


x = np.linspace(-np.pi, np.pi, 100)
y = [np.sin(ii) for ii in x]

plt.plot(x, y, label='a sinusoid')

plt.grid()
plt.legend()

# Ajouter les labels des axes et le titre
plt.xlabel(r'The $x$ coordinate')
plt.ylabel(r'The $y$ coordinate')
plt.title('A sample graph')

# Sauvegarder la figure en .pdf
plt.tight_layout()
plt.savefig('Samplefig.pdf')
