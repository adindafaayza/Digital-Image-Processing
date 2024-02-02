import numpy as np
from PIL import Image
import image_processing
import os
from flask import Flask, render_template, request, make_response
from datetime import datetime
from functools import wraps, update_wrapper
from shutil import copyfile
import random

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    return update_wrapper(no_cache, view)


@app.route("/index")
@app.route("/")
@nocache
def index():
    return render_template("home.html", file_path="img/image_here.jpg")


@app.route("/about")
@nocache
def about():
    return render_template('about.html')


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route("/upload", methods=["POST"])
@nocache
def upload():
    target = os.path.join(APP_ROOT, "static/img")
    if not os.path.isdir(target):
        if os.name == 'nt':
            os.makedirs(target)
        else:
            os.mkdir(target)
    for file in request.files.getlist("file"):
        file.save("static/img/img_now.jpg")
    copyfile("static/img/img_now.jpg", "static/img/img_normal.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/normal", methods=["POST"])
@nocache
def normal():
    copyfile("static/img/img_normal.jpg", "static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/grayscale", methods=["POST"])
@nocache
def grayscale():
    image_processing.grayscale()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/zoomin", methods=["POST"])
@nocache
def zoomin():
    image_processing.zoomin()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/zoomout", methods=["POST"])
@nocache
def zoomout():
    image_processing.zoomout()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/move_left", methods=["POST"])
@nocache
def move_left():
    image_processing.move_left()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/move_right", methods=["POST"])
@nocache
def move_right():
    image_processing.move_right()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/move_up", methods=["POST"])
@nocache
def move_up():
    image_processing.move_up()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/move_down", methods=["POST"])
@nocache
def move_down():
    image_processing.move_down()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/brightness_addition", methods=["POST"])
@nocache
def brightness_addition():
    image_processing.brightness_addition()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/brightness_substraction", methods=["POST"])
@nocache
def brightness_substraction():
    image_processing.brightness_substraction()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/brightness_multiplication", methods=["POST"])
@nocache
def brightness_multiplication():
    image_processing.brightness_multiplication()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/brightness_division", methods=["POST"])
@nocache
def brightness_division():
    image_processing.brightness_division()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/histogram_equalizer", methods=["POST"])
@nocache
def histogram_equalizer():
    image_processing.histogram_equalizer()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/edge_detection", methods=["POST"])
@nocache
def edge_detection():
    image_processing.edge_detection()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/blur", methods=["POST"])
@nocache
def blur():
    image_processing.blur()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/sharpening", methods=["POST"])
@nocache
def sharpening():
    image_processing.sharpening()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/zero_padding", methods=["POST"])
@nocache
def zero_padding():
    image_processing.zero_padding()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/customBlur", methods=["POST"])
@nocache
def customBlur():
    size = int(request.form['size'])
    image_processing.customBlur(size)
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/gaussianBlur", methods=["POST"])
@nocache
def gaussianBlur():
    size = int(request.form['size'])
    image_processing.gaussianBlur(size)
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/medianBlur", methods=["POST"])
@nocache
def medianBlur():
    size = int(request.form['size'])
    image_processing.medianBlur(size)
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/bilateralBlur", methods=["POST"])
@nocache
def bilateralBlur():
    image_processing.bilateralBlur()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/lowPassFilter", methods=["POST"])
@nocache
def lowPassFilter():
    image_processing.lowPassFilter()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/lowFilterPass", methods=["POST"])
@nocache
def lowFilterPass():
    size = int(request.form['size'])
    image_processing.lowFilterPass(size)
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/highPassFilter", methods=["POST"])
@nocache
def highPassFilter():
    image_processing.highPassFilter()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/highFilterPass", methods=["POST"])
@nocache
def highFilterPass():
    size = int(request.form['size'])
    image_processing.highFilterPass(size)
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/bandPassFilter", methods=["POST"])
@nocache
def bandPassFilter():
    image_processing.bandPassFilter()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/bandFilterPass", methods=["POST"])
@nocache
def bandFilterPass():
    size = int(request.form['size'])
    image_processing.bandFilterPass(size)
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/histogram_rgb", methods=["POST"])
@nocache
def histogram_rgb():
    image_processing.histogram_rgb()
    if image_processing.is_grey_scale("static/img/img_now.jpg"):
        return render_template("histogram.html", file_paths=["img/grey_histogram.jpg"])
    else:
        return render_template("histogram.html", file_paths=["img/red_histogram.jpg", "img/green_histogram.jpg", "img/blue_histogram.jpg"])


@app.route("/thresholding", methods=["POST"])
@nocache
def thresholding():
    lower_thres = int(request.form['lower_thres'])
    upper_thres = int(request.form['upper_thres'])
    image_processing.threshold(lower_thres, upper_thres)
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/puzzle", methods=["POST"])
@nocache
def puzzle():
    num_puzzles = int(request.form['size'])
    image_path = "static/img/img_now.jpg"  # Assuming this is the path of your uploaded image
    rows = num_puzzles
    cols = num_puzzles

    parts = image_processing.create_puzzle(rows)

    if parts is not None:
        return render_template("puzzle.html", image_paths=[f"static/img/puzzle_piece_{i}_{j}.jpg" for i in range(rows) for j in range(cols)], rows=rows, cols=cols)
    else:
        return "Terjadi kesalahan saat membagi gambar. Silakan coba lagi."

@app.route("/puzzle_random", methods=["POST"])
@nocache
def puzzle_random():
    num_puzzles = int(request.form['size'])
    image_path = "static/img/img_now.jpg"  # Assuming this is the path of your uploaded image
    rows = num_puzzles
    cols = num_puzzles

    parts = image_processing.create_puzzle(rows)

    if parts is not None:
        return render_template("puzzle_random.html", image_paths=[f"static/img/puzzle_piece_{i}_{j}.jpg" for i in range(rows) for j in range(cols)], rows=rows, cols=cols)
    else:
        return "Terjadi kesalahan saat membagi gambar. Silakan coba lagi."

@app.route("/memory_game", methods=["POST"])
def memory_game():
    # Define the list of filter functions to apply
    target = os.path.join(APP_ROOT, "static/img")
    if not os.path.isdir(target):
        if os.name == "nt":
            os.makedirs(target)
        else:
            os.mkdir(target)
    for file in request.files.getlist("file"):
        file.save("static/img/img_now.jpg")
    copyfile("static/img/img_now.jpg", "static/img/img_normal.jpg")
    filter_functions = [
        image_processing.inverseFilter,
        image_processing.grayscale,
        image_processing.highPassFilter,
        image_processing.lowPassFilter,
        image_processing.gaussianBlur25,
        image_processing.bandPassFilter,
        image_processing.brightness_addition,
        image_processing.brightness_multiplication,
        image_processing.brightness_division,
        image_processing.brightness_substraction,
        image_processing.medianBlur25,
        image_processing.histogram_equalizer,
        image_processing.bilateralBlur,
        image_processing.edge_detection
    ]

    target = os.path.join(APP_ROOT, "static/img")
    if not os.path.isdir(target):
        if os.name == "nt":
            os.makedirs(target)
        else:
            os.mkdir(target)

    # Copy "img_normal.jpg" as the source image
    copyfile("static/img/img_normal.jpg", "static/img/img_now.jpg")

    filtered_file_paths = []  # Create an empty list to store the generated file paths
    num_duplicates = 2  # Define how many times each image should be duplicated

    for filter_func in filter_functions:
        for _ in range(num_duplicates):  # Loop to duplicate the image
            # Copy "img_normal.jpg" as the source image for each filter iteration
            copyfile("static/img/img_normal.jpg", "static/img/img_now.jpg")

            filter_func()  # Apply the filter to "img_now.jpg"

            # Save the filtered image with a new name (e.g., add filter name as a suffix)
            filter_name = filter_func.__name__
            new_image_path = f"static/img/img_now_{filter_name}.jpg"
            copyfile("static/img/img_now.jpg", new_image_path)
            filtered_file_paths.append(new_image_path)  # Append the generated file path


    random.shuffle(filtered_file_paths)  # Shuffle the list of filtered file paths

    return render_template("memory_game.html", filtered_file_paths=filtered_file_paths)


@app.route('/show_image_values', methods=['POST'])
@nocache
def show_image_values():
    # Mendapatkan nilai dari gambar
    pixel_values, width, height = image_processing.get_image_values('static/img/img_now.jpg')
    
    # Mengirimkan nilai-nilai tersebut ke template HTML
    return render_template("pixel_values.html", pixel_values=pixel_values, width=width, height=height)


@app.route('/read_digits_from_images', methods=['POST'])
def read_digits_from_image():
    target = os.path.join(APP_ROOT, "static/img")
    if not os.path.isdir(target):
        if os.name == 'nt':
            os.makedirs(target)
        else:
            os.mkdir(target)
    for file in request.files.getlist("file"):
        file.save("static/img/img_now.jpg")
    copyfile("static/img/img_now.jpg", "static/img/img_normal.jpg")
    results = image_processing.recognize_digits_in_images()
    return render_template("recognized_digits.html", results=results)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")