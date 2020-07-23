import cv2
from mesh_detector import MeshDetector
from matplotlib import pyplot
import json


if __name__ == "__main__":
    # Construct a face mesh detector.
    md = MeshDetector(
        "/home/robin/Desktop/head-pose-estimation/assets/face_landmark.tflite")

    # Read in a sample image for mesh detection.
    image_file = '/home/robin/Desktop/face.jpg'
    image = cv2.imread(image_file)

    # Note that the mesh detector is sensitive to the face box size.
    epn_width = 20
    image = cv2.copyMakeBorder(image,
                               epn_width, epn_width, epn_width, epn_width,
                               cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Get the mesh.
    mesh, _ = md.get_mesh(image)

    # # Save the mesh to file.
    # with open("mesh.json", "w") as fid:
    #     json.dump(mesh.flatten().tolist(), fid)

    # Draw the image and marks.
    fig_2d = pyplot.figure()
    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    md.draw_mesh(image, mesh)
    img_plot = fig_2d.add_subplot(1, 1, 1)
    img_plot.imshow(image)

    # Draw the face mesh points in 3D.
    fig_3d = pyplot.figure(figsize=(22, 16), tight_layout=True)
    mesh_plot = fig_3d.add_subplot(1, 1, 1, projection="3d")
    mesh_plot.view_init(270, 270)
    mesh_plot.set_axis_off()

    mesh *= 512
    x, y, z = mesh[:, 0], mesh[:, 1], mesh[:, 2]
    face, = mesh_plot.plot(x, y, z, color='#ae7181', marker='.',
                           linestyle='None', markersize=6, picker=True,
                           pickradius=5)

    # # Draw point index.
    # index = range(0, 467)
    # for x, y, z, s in zip(x, y, z, index):
    #     mesh_plot.text(x, y, z, str(s), fontsize=5)

    point_index_text = mesh_plot.text2D(
        0.05, 0.95, 'Point Index:', transform=mesh_plot.transAxes)

    def on_pick(event):
        if event.artist != face:
            return True

        N = len(event.ind)
        if not N:
            return True
        idx = event.ind
        point_index_text.set_text("Point Index:" + str(idx))
        pyplot.show()
        return True

    fig_3d.canvas.mpl_connect('pick_event', on_pick)

    # fig_3d.savefig('mesh.pdf')

    pyplot.show()
