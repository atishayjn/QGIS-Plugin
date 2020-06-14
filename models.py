        def randomForest(self, fp1, fp2, fp3):
        #import statements
            try:
                from osgeo import gdal, gdal_array
            except ImportError:
                print("GDAL package not present\nInstalling...")
                import pip
                pip.main(["install", "--user", "GDAL"])
                from osgeo import gdal, gdal_array


            try:
                import pickle
            except ImportError:
                import pip
                pip.main(["install", "--user", "pickle"])
                import pickle


            try:
                from sklearn.model_selection import train_test_split
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import confusion_matrix,classification_report
            except ImportError:
                print("scikit-learn package not present\nInstalling...")
                import pip
                pip.main(["install", "--user", "scikit-learn"])

                from sklearn.model_selection import train_test_split
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import confusion_matrix,classification_report

            try:
                import numpy as np
            except ImportError:
                import pip
                pip.main(["install", "--user", "numpy"])
                import numpy as np

            try:
                import seaborn as sns
            except ImportError:
                import pip
                pip.main(["install", "--user", "seaborn"])
                import seaborn as sns

            try:
                import matplotlib.pyplot as plt
            except ImportError:
                import pip
                pip.main(["install", "--user", "matplotlib"])

                import matplotlib.pyplot as plt


            #Paths
            #DATA_PATH = "/content/drive/My Drive/PS-1/meghalaya_data/"
            IMAGE_ADD = fp1
            MODEL_ADD = fp2
            OUTPUT_ADD = fp3

            # #To open the image:
            img_ds = gdal.Open(IMAGE_ADD, gdal.GA_ReadOnly)

            img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount), gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))

for b in range(img.shape[2]):
    img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

            print(img.shape)

            rf = pickle.load(open(MODEL_ADD, 'rb'))

            img_as_array = img.reshape(-1, 8)
            print('Reshaped from {o} to {n}'.format(o=img.shape,
                                                    n=img_as_array.shape))

            # Now predict for each pixel
            class_prediction = rf.predict(img_as_array)

            # Reshape our classification map
            class_prediction = class_prediction.reshape(img[:, :, 0].shape)
            print(class_prediction.shape)


           # Visualize

            # First setup a 5-4-3 composite
            def color_stretch(image, index, minmax=(0, 10000)):
                colors = image[:, :, index].astype(np.float64)

                max_val = minmax[1]
                min_val = minmax[0]

                # Enforce maximum and minimum values
                colors[colors[:, :, :] > max_val] = max_val
                colors[colors[:, :, :] < min_val] = min_val

                for b in range(colors.shape[2]):
                    colors[:, :, b] = colors[:, :, b] * 1 / (max_val - min_val)
                    
                return colors
                
            img543 = color_stretch(img, [3, 2, 1], (0, 8000))

            # See https://github.com/matplotlib/matplotlib/issues/844/
            n = class_prediction.max()
            # Next setup a colormap for our map
            colors = dict((
                (0, (0, 0, 0, 255)),  # Nodata
                (2, (0, 150, 0, 255)),  # Forest
                (1, (0, 0, 255, 255)),  # Water
                (3, (150, 0, 0, 255))  # Landuse
            ))
            # Put 0 - 255 as float 0 - 1
            for k in colors:
                v = colors[k]
                _v = [_v / 255.0 for _v in v]
                colors[k] = _v
                
            index_colors = [colors[key] if key in colors else 
                            (255, 255, 255, 0) for key in range(1, n + 1)]
            cmap = plt.matplotlib.colors.ListedColormap(index_colors)#, 'Classification', n)

            # Now show the classmap next to the image
            plt.subplot(121)
            plt.imshow(img543)
            plt.title('Landsat Image')
            print(class_prediction.shape)
            plt.subplot(122)
            plt.imshow(class_prediction, cmap=cmap, interpolation='none')
            plt.title('Classified Image')

            plt.savefig("/content/out.pdf")
            plt.savefig("/content/out.png")
            plt.show()