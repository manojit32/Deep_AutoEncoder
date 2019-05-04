def run1():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_digits
    from sklearn.neighbors import KernelDensity
    from sklearn.decomposition import PCA
    from sklearn.model_selection import GridSearchCV
    digits = load_digits()
    pca = PCA(n_components=15, whiten=False)
    data = pca.fit_transform(digits.data)
    params = {'bandwidth': np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params, cv=5)
    grid.fit(data)
    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    kde = grid.best_estimator_
    new_data = kde.sample(48, random_state=0)
    new_data = pca.inverse_transform(new_data)
    new_data = new_data.reshape((4, 12, -1))
    real_data = digits.data[:48].reshape((4, 12, -1))
    fig, ax = plt.subplots(9, 12, subplot_kw=dict(xticks=[], yticks=[]))
    for j in range(12):
        ax[4, j].set_visible(False)
        for i in range(4):
            im = ax[i, j].imshow(real_data[i, j].reshape((8, 8)),
                                 cmap=plt.cm.binary, interpolation='nearest')
            im.set_clim(0, 16)
            im = ax[i + 5, j].imshow(new_data[i, j].reshape((8, 8)),
                                     cmap=plt.cm.binary, interpolation='nearest')
            im.set_clim(0, 16)

    ax[0, 5].set_title('Selection from the input data')
    ax[5, 5].set_title('"New" digits drawn from the kernel density model')
    plt.show()
    
    
def run2():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_digits
    from sklearn.neighbors import KernelDensity
    from sklearn.decomposition import PCA
    from sklearn.model_selection import GridSearchCV
    digits = load_digits()
    pca = PCA(n_components=25, whiten=False)
    data = pca.fit_transform(digits.data)
    params = {'bandwidth': np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params, cv=5)
    grid.fit(data)
    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    kde = grid.best_estimator_
    new_data = kde.sample(48, random_state=0)
    new_data = pca.inverse_transform(new_data)
    new_data = new_data.reshape((4, 12, -1))
    real_data = digits.data[:48].reshape((4, 12, -1))
    fig, ax = plt.subplots(9, 12, subplot_kw=dict(xticks=[], yticks=[]))
    for j in range(12):
        ax[4, j].set_visible(False)
        for i in range(4):
            im = ax[i, j].imshow(real_data[i, j].reshape((8, 8)),
                                 cmap=plt.cm.binary, interpolation='nearest')
            im.set_clim(0, 16)
            im = ax[i + 5, j].imshow(new_data[i, j].reshape((8, 8)),
                                     cmap=plt.cm.binary, interpolation='nearest')
            im.set_clim(0, 16)

    ax[0, 5].set_title('Selection from the input data')
    ax[5, 5].set_title('"New" digits drawn from the kernel density model')
    plt.show()
    
    
def run3():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_digits
    from sklearn.neighbors import KernelDensity
    from sklearn.decomposition import PCA
    from sklearn.model_selection import GridSearchCV
    digits = load_digits()
    pca = PCA(n_components=35, whiten=False)
    data = pca.fit_transform(digits.data)
    params = {'bandwidth': np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params, cv=5)
    grid.fit(data)
    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    kde = grid.best_estimator_
    new_data = kde.sample(48, random_state=0)
    new_data = pca.inverse_transform(new_data)
    new_data = new_data.reshape((4, 12, -1))
    real_data = digits.data[:48].reshape((4, 12, -1))
    fig, ax = plt.subplots(9, 12, subplot_kw=dict(xticks=[], yticks=[]))
    for j in range(12):
        ax[4, j].set_visible(False)
        for i in range(4):
            im = ax[i, j].imshow(real_data[i, j].reshape((8, 8)),
                                 cmap=plt.cm.binary, interpolation='nearest')
            im.set_clim(0, 16)
            im = ax[i + 5, j].imshow(new_data[i, j].reshape((8, 8)),
                                     cmap=plt.cm.binary, interpolation='nearest')
            im.set_clim(0, 16)

    ax[0, 5].set_title('Selection from the input data')
    ax[5, 5].set_title('"New" digits drawn from the kernel density model')
    plt.show()