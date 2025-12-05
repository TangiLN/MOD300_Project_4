import matplotlib.pyplot as plt
import numpy as np 
from mw_plot import MWFaceOn
from astropy import units as u
from mw_plot import MWSkyMap
from astropy.coordinates import SkyCoord
from sklearn.cluster import KMeans
from PIL import Image
plt.style.use("mw_plot.mplstyle")
plt.rcParams["text.usetex"] = False
class topic1_generation :
    """
    Class for the Topic 1 of the project 4 
    """
    def __init__(self):
        """
        Docstring for __init__
        
        :param self: Class topic1_generation
        """
        self.cluster_colors = np.array([
            [255, 0, 0],    # Cluster 0 : Rouge
            [0, 255, 0],    # Cluster 1 : Vert
            [0, 0, 255],    # Cluster 2 : Bleu
            [255, 255, 0]   # Cluster 3 : Jaune
        ], dtype=np.uint8)
    def generate_bird_eye_view(self):
        """Function to generate the plot of the birds Eye View from mw_plot"""
        mw1 = MWFaceOn(
            radius=20 * u.kpc,
            unit=u.kpc,
            coord="galactocentric",
            annotation=True,
            figsize=(10, 8),
        )
        mw1.title = "Birds Eyes View"
        mw1.scatter(8 * u.kpc, 0 * u.kpc, c="r", s=2)
    def generate_skymap(self,center=(0,0),radius=(8800,8800),option=None,name_png="galaxy.png",delete_axis=True):
        """
        Function to generate the different viw of the Milky Way
        
        :param center: Coordinate of the center tuple(x,y) 
        :param radius: Size of the radius, tuple(length,width)
        :param option: String to make a plot of a specific place
        :param name_png: Name of the image that will be stored
        :param axis : Bool that allow the plotting or not of the axis,useful for the image processing later
        """
        if option == "M31":
            m31_coords = SkyCoord.from_name("M31")
            center = (m31_coords.ra.deg * u.deg, m31_coords.dec.deg * u.deg)
        elif option == "M42" :
            m42_coords = SkyCoord.from_name("M42")
            center = (m42_coords.ra.deg * u.deg, m42_coords.dec.deg * u.deg)
        else :
            center=(center[0]*u.deg,center[1]*u.deg)
        mw1 = MWSkyMap(
            center=center,
            radius=radius * u.arcsec,
            background="Mellinger color optical survey",
        )
        fig, ax = plt.subplots(figsize=(5, 5))
        mw1.transform(ax)
        if delete_axis :
            ax.set_axis_off()
            fig.subplots_adjust(
                left=0, bottom=0,
                right=1, top=1, 
                wspace=0, hspace=0
            )
        mw1.savefig(name_png)
    def convert_png_to_rgb(self,img=None):
        """
        Function to Convert the image generated to an array
        Input -> img : name if the image to convert
        """
        #Security to be sure that we have an image passed to the function
        assert (img is not None), "Please insert an image as a parameter"
        image=Image.open(img)
        image_rgb = image.convert('RGB')
        array= np.array(image_rgb)
        print(array.shape)
        return array
    def encode_array(self,array,type='Grey'):
        """
        Function to set the encoding style
        :param array: Array to encode
        :param type: type of encoding to choose
        """
        x , y = [], []
        grey = np.sum(array[: , : , :] * np.array([0.299, 0.587, 0.114]), axis=2)
        if type=='Grey':
            min_grey = 200 
            max_grey = 230
            grey_mask = np.logical_and(grey > min_grey, grey <= max_grey)
            x,y=np.where(grey_mask)
        elif type== 'Red':
            red_treshold=80
            red_mask=((array[:,:,0 ] > red_treshold) & 
                    (array[:, :, 0] > array[:, :, 1] * 1.2) & 
                    (array[:, :, 0] > array[:, :, 2] * 1.2))
            x, y = np.where(red_mask)
        elif type=='Blue':
            blue_treshold=150
            blue_mask=((array[:,:,2 ] > blue_treshold) & 
                    (array[:, :, 2] > array[:, :, 0] * 1.2) & 
                    (array[:, :, 2] > array[:, :, 1] * 1.2))
            x, y = np.where(blue_mask)
        elif type =='Dark':
            min_dark=20
            max_dark=40
            dark_mask=np.logical_and(grey >=min_dark , grey < max_dark)
            x, y = np.where(dark_mask)
        else : 
            print("Wrong encoding selection : Blue - Red -Grey -Dark")
            return
        plt.scatter(x, y, s=0.1)
        #plt.gca().invert_yaxis()
        return x,y
    def k_mean_clustering(self,data,plot=True):
        """
        Function to make the K_Mean clustering 
        :param data: Data on wich apply the clustering
        :param plot: Bool to allow plotting or not
        """
        kmeans = KMeans(n_clusters=4) 
        kmeans.fit(data)
        labels = kmeans.labels_
        if plot: 
            plt.title("K_Mean clustering")
            plt.scatter(data[:, 0], data[:, 1], c=labels) 
            plt.show()
        return labels

    def over_impose(self, image_array, data_kmean, labels):
        """
        Function to over-impose the image and cluster.
        Inputs -> image_array : The image represented by an array generated before
                  data_kmean : Coordinate of the points found with kmean funciton
                  labels : labels of th
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(image_array)
        colors = [self.cluster_colors[label % len(self.cluster_colors)] / 255.0 for label in labels]
        plt.scatter(
            data_kmean[:, 1],
            data_kmean[:, 0],
            c=colors, 
            s=5,
            marker='o'
        )
        plt.title("Clustering Superposé sur l'Image Originale")
        # plt.gca().invert_yaxis() # Important si vous utilisez les indices d'array (0,0 en haut à gauche)
        plt.axis('off')
        plt.show()
    def run(self,array,type="Grey"):
        """
        Function to run a full sequence, useful for task 7 to avoid code duplication
        :param array: array of the image to work on 
        :param type: Encoding style 
        """
        data=self.encode_array(array=array,type=type)
        data_for_kmeans = np.column_stack(data)
        print(f"Number of pixels clustered: {len(data[0])}")
        label=self.k_mean_clustering(data=data_for_kmeans)
        self.over_impose(image_array=array,data_kmean=data_for_kmeans,labels=label)
