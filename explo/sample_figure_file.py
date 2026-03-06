import matplotlib.pyplot as plt
import numpy as np

# matplotlib setup
import matplotlib
# adjust the font size accordingly
font = {'family' : 'sans',
        'size'   : 12}

matplotlib.rc('font', **font)


x = np.linspace(-np.pi, np.pi, 100)
y = [np.sin(ii) for ii in x]

plt.plot(x, y, label='a sinusoid')

plt.grid()
plt.legend()

# Label your axes and title
plt.xlabel(r'The $x$ coordinate')
plt.ylabel(r'The $y$ coordinate')
plt.title('A sample graph')

# Save your figure in .pdf
plt.tight_layout()
plt.savefig('Samplefig.pdf')
