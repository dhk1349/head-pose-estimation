import cv2
from mesh_detector import MeshDetector
from matplotlib import pyplot


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

    # Draw the results in one figure.
    fig = pyplot.figure()

    # Draw the image and marks.
    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    md.draw_mesh(image, mesh)
    img_plot = fig.add_subplot(1, 2, 1)
    img_plot.imshow(image)

    # Draw the face mesh points in 3D.
    mesh_plot = fig.add_subplot(1, 2, 2, projection="3d")
    mesh_plot.view_init(260, 270)
    x, y, z = mesh[:, 0], mesh[:, 1], mesh[:, 2]
    mesh *= 512
    face, = mesh_plot.plot(x, y, z, color='#ae7181', marker='.',
                           linestyle='None', markersize=4)

    pyplot.show()
