import numpy as np
import cv2
import matplotlib
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile
import gradio as gr

"""单张图像聚类分割"""
# 单张图像分割
def image_segmentation(image, n_clusters):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_reshaped = image.reshape((-1, 3))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(image_reshaped)
        segmented_img = kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape).astype(int)

        return segmented_img
    except Exception as e:
        print(f"Error during image segmentation: {e}")
        return None

"""MNIST数据集聚类"""
# 获取MNIST数据集
def fetch_mnist_data(subset_size=2000):
    try:
        mnist = fetch_openml('mnist_784', version=1)
        data = mnist.data.astype('float32') / 255.0

        if subset_size > len(data):
            subset_size = len(data)

        indices = np.random.choice(len(data), subset_size, replace=False)
        data = data.iloc[indices].to_numpy()

        return data
    except Exception as e:
        return f"An unexpected error occurred: {e}"
# MNIST数据集聚类
def cluster_mnist(data, n_clusters=10):
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data)
        labels = kmeans.labels_

        pca = PCA(2)
        data_2d = pca.fit_transform(data)

        plt.figure(figsize=(8, 8))

        for i in range(n_clusters):
            indices = labels == i
            plt.scatter(data_2d[indices, 0], data_2d[indices, 1], label=f'Cluster {i}', alpha=0.5)

        plt.legend(loc='lower right').title(f'K-means Clustering with {n_clusters} Clusters')
        
        plt.xlabel("PCA Component 1").ylabel("PCA Component 2").grid(True)

        # Save plot to a temporary PNG file
        temp_file = NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name)
        plt.close()

        return temp_file.name
    except Exception as e:
        print(f"Error during MNIST clustering: {e}")
        return None
    finally:
        from matplotlib import pyplot
        pyplot.switch_backend('Agg')  # 确保恢复后端到 Agg

"""gradio接口设计"""
def segment_image(filepath, n_clusters):
    image = cv2.imread(filepath)
    if image is None:
        return "Error: Unable to read the image. Please check the file path."
    segmented_img = image_segmentation(image, n_clusters)
    if segmented_img is not None:
        return segmented_img
    else:
        return "Image segmentation failed."
    
def mnist_clustering(n_clusters, subset_size):
    data = fetch_mnist_data(subset_size)
    if isinstance(data, str) and data.startswith("Error"):
        return data
    image_path = cluster_mnist(data, n_clusters)
    if image_path:
        return image_path
    else:
        return "Cluster plot generation failed."


if __name__ == "__main__":
    matplotlib.use('Agg')  # 确保启动时后端为Agg
    print("Script is starting...")

    # 单张图像聚类分割
    image_demo = gr.Interface(
        fn=segment_image,
        inputs=[gr.Image(type="filepath"),
                gr.Slider(minimum=2, maximum=10, step=1, value=3, label="Number of Clusters")],
        outputs="image",
        title="Image Segmentation with K-Means",
        description="Upload an image and select the number of clusters to segment the image using K-Means clustering."
    )

    # MNIST数据集聚类
    mnist_demo = gr.Interface(
        fn=mnist_clustering,
        inputs=[gr.Slider(minimum=2, maximum=20, step=1, value=10, label="Number of Clusters"),
                gr.Slider(minimum=1000, maximum=10000, step=1000, value=2000, label="Subset Size")],
        outputs="file",  # 修改输出为文件类型
        title="MNIST Clustering with K-Means",
        description="Perform K-Means clustering on a subset of the MNIST dataset and plot the result."
    )

    print("界面加载成功，与服务器建立连接...( つ•̀ω•́)つ🍺.")
    gr.TabbedInterface([image_demo, mnist_demo], ["Image Segmentation", "MNIST Clustering"]).launch()

