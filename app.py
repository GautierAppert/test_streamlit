import streamlit as st
from PIL import Image
import numpy as np
import os.path


@st.cache(allow_output_mutation=True)
def load_subwindows() -> list:

    global x

    if os.path.isfile("x.npy"):

        x = np.load("x.npy")

        return list(x)

    else:

        x = []

        return x


class SampleGeneration:
    """
    Class to generate a sample of patches/subwindows
    """

    def __init__(
        self,
        size: int,
        steps: int,
        start: list,
        frame: list,
        sample_size: int,
        original_image: np.array,
    ):
        self.start = start
        self.frame = frame
        self.size = size
        self.steps = steps
        self.sample_size = sample_size
        self.original_image = original_image

    @staticmethod
    def get_multiples(value: int, max_val: int):
        return list(range(value, max_val + 1, value))

    def get_positions(self, idx_frame, idx_start, size, steps):
        upper_range = idx_frame - idx_start - size

        return SampleGeneration.get_multiples(steps, upper_range + 1)

    def extract_particular_subwindow(self, x_position, y_position):
        x0 = self.start[0] + self.steps * x_position
        x1 = x0 + self.size + 1
        y0 = self.start[1] + self.steps * y_position
        y1 = y0 + self.size + 1

        return self.original_image[x0:x1, y0:y1]

    def extract_randomized_subwindows(self):

        x_indexes = self.get_positions(
            self.frame[0], self.start[0], self.size, self.steps
        )
        y_indexes = self.get_positions(
            self.frame[1], self.start[1], self.size, self.steps
        )
        x_indexes = np.random.choice(x_indexes, size=self.sample_size, replace=True)
        y_indexes = np.random.choice(y_indexes, size=self.sample_size, replace=True)

        ## concatenate x_indexes and y_indexes into a numpy array dim = 2
        random_positions = np.array(list(zip(x_indexes, y_indexes)))

        subwindows_list = list(
            np.apply_along_axis(
                lambda x: self.extract_particular_subwindow(x[0], x[1]),
                1,
                random_positions,
            )
        )

        return subwindows_list


def patch_process():
    """
    Main function to run the app:
     if "Fragmentation" is selected, we'll call first the function display_yellow_frame(),
     create_sample(), show_syntax_label() and
     then the function associated with the fragmentation process.
    """

    st.sidebar.title("Fragmentation and Syntax Analysis App")
    options = st.sidebar.selectbox("Select Page", ["Fragmentation", "Syntax Analysis"])

    if options == "Fragmentation":

        st.header("**Fragmentation**")
        display_yellow_frame()
        create_sample()
        show_syntax_label()

    elif options == "Syntax Analysis":

        st.write(
            "**Work in progress**",
        )


def display_yellow_frame():
    """
    Function to display the yellow frame on the big image.
    """

    st.sidebar.title("Prepare sample")
    uploaded_file = st.sidebar.file_uploader("Load new image", type=["jpg", "jpeg"])
    frame_slider_features = {"max_x": 5000, "max_y": 5000}

    if uploaded_file is not None:

        global im_array

        im_array = np.asarray(
            Image.open(uploaded_file)
        )  ## Import an image and turn it into a numpy array.
        im_array = im_array[:, :, 1]  ## keeping only the green channel
        frame_slider_features["max_x"] = im_array.shape[0]
        frame_slider_features["max_y"] = im_array.shape[1]

        original_image_display_location = st.empty()
        original_image_display_location.image(im_array, width=700)

    with st.sidebar.form(key="form1"):

        st.markdown("### Upper left frame corner :")
        layout = st.columns([1, 1])
        with layout[0]:
            ULFrameX = st.slider(
                "X",
                min_value=1,
                max_value=frame_slider_features["max_x"],
                step=1,
                value=min(500, frame_slider_features["max_x"]),
            )
        with layout[1]:
            ULFrameY = st.slider(
                "Y",
                min_value=1,
                max_value=frame_slider_features["max_y"],
                step=1,
                value=min(500, frame_slider_features["max_y"]),
            )

        st.markdown("### Lower right frame corner :")
        layout = st.columns([1, 1])
        with layout[0]:
            LRFrameX = st.slider(
                " X",
                min_value=1,
                max_value=frame_slider_features["max_x"],
                step=1,
                value=min(2500, frame_slider_features["max_x"]),
            )
        with layout[1]:
            LRFrameY = st.slider(
                " Y",
                min_value=1,
                max_value=frame_slider_features["max_y"],
                step=1,
                value=min(2500, frame_slider_features["max_y"]),
            )

        ## Instantiation of the dict patch_sizing
        global patch_sizing
        patch_sizing = {}
        patch_sizing["start"] = [ULFrameX, ULFrameY]
        patch_sizing["frame"] = [LRFrameX, LRFrameY]

        ## submit button associated with 'form1'
        submit_button = st.form_submit_button(label="Show Frame")

        if submit_button:

            try:
                ## creating the image with the yellow frame
                im_frame = im_array.copy()
                im_frame = np.repeat(im_frame[:, :, np.newaxis], 3, axis=2)
                im_frame[
                    patch_sizing["start"][0] : patch_sizing["frame"][0],
                    patch_sizing["start"][1] : patch_sizing["frame"][1],
                    2,
                ] = 0
                original_image_display_location.image(im_frame, width=600)

            except Exception as e:
                st.write("Error: ", e)


def create_sample():

    with st.sidebar.form("form2"):

        st.markdown("### Image size:")
        size_slider = st.slider("", min_value=1, max_value=400, step=1, value=300)

        st.markdown("### Step size:")
        step_slider = st.slider("", min_value=1, max_value=400, step=1, value=1)

        st.markdown("### Sample size:")
        sample_size = st.slider(
            "Sample size: ", min_value=1, max_value=2000, step=1, value=100
        )

        st.markdown("### Epsilon value:")
        epsilon_slider = st.slider(
            "", min_value=-4.0, max_value=2.0, step=0.1, value=-2.0
        )
        epsilon = 10 ** (epsilon_slider)

        ## submit button associated with 'form2'
        submit_button = st.form_submit_button(label="Add to training set")

        if submit_button:

            my_sample = SampleGeneration(
                size_slider,
                step_slider,
                patch_sizing["start"],
                patch_sizing["frame"],
                sample_size,
                im_array.copy(),
            )

            if "x" not in globals():
                x = load_subwindows()

            x.extend(my_sample.extract_randomized_subwindows())

            lx = [np.log(subwindows + epsilon) for subwindows in x]

            np.save("x.npy", x)


def show_syntax_label():
    """function to show patches in the sample"""

    x = load_subwindows()
    st.sidebar.markdown("### Show patches in image number")
    image_nb = st.sidebar.slider("", min_value=1, max_value=len(x), step=1, value=1)

    if len(x) != 0:

        st.image(x[image_nb - 1], width=500)


if __name__ == "__main__":

    patch_process()
