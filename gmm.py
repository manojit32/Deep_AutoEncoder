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
    from sklearn import mixture
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 10)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(data)
            bic.append(gmm.bic(data))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    print(best_gmm)        
    new_data = best_gmm.sample(44)
    new_data = pca.inverse_transform(new_data[0])
    new_data = new_data.reshape((4, 11, -1))
    real_data = digits.data[:44].reshape((4, 11, -1))
    fig, ax = plt.subplots(9, 11, subplot_kw=dict(xticks=[], yticks=[]))
    for j in range(11):
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
    from sklearn import mixture
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 10)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(data)
            bic.append(gmm.bic(data))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    print(best_gmm)        
    new_data = best_gmm.sample(44)
    new_data = pca.inverse_transform(new_data[0])
    new_data = new_data.reshape((4, 11, -1))
    real_data = digits.data[:44].reshape((4, 11, -1))
    fig, ax = plt.subplots(9, 11, subplot_kw=dict(xticks=[], yticks=[]))
    for j in range(11):
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
    from sklearn import mixture
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 10)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(data)
            bic.append(gmm.bic(data))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    print(best_gmm)        
    new_data = best_gmm.sample(44)
    new_data = pca.inverse_transform(new_data[0])
    new_data = new_data.reshape((4, 11, -1))
    real_data = digits.data[:44].reshape((4, 11, -1))
    fig, ax = plt.subplots(9, 11, subplot_kw=dict(xticks=[], yticks=[]))
    for j in range(11):
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