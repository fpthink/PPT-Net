import numpy as np
import os

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def count_files():
    root = '/test/dataset/benchmark_datasets/oxford'    # dataset path
    files = os.listdir(root)
    cnt = 0
    for i in range(len(files)):
        data_path = os.path.join(root, files[i], 'pointcloud_20m_10overlap')
        cnt += len(os.listdir(data_path))
    print('data files: {}'.format(cnt))
    return cnt

def plot_point_cloud(points, label=None, output_filename=''):
    """ points is a Nx3 numpy array """
    # import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    # ax = Axes3D(fig)
    ax = plt.subplot(111, projection='3d')
    if label is not None:
        point = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                            # cmap='RdYlBu',
                            c = label,
                            # linewidth=2,
                            alpha=0.5,
                            marker=".")
    else:
        point = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                            # cmap='RdYlBu',
                            c = points[:, 2],
                            # linewidth=2,
                            alpha=0.5,
                            marker=".")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # fig.colorbar(point)
    # plt.axis('scaled')
    # plt.axis('off')
    if output_filename!='':
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# points = np.random.randn(4096, 3)
# plot_point_cloud(points,output_filename='visual/test.png')