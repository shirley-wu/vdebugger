from typing import List, Union

from vision_functions import find_in_image, simple_qa, verify_property, best_text_match, compute_depth


def bool_to_yesno(bool_answer: bool) -> str:
    return "yes" if bool_answer else "no"


class ImagePatch:
    """A Python class containing a crop of an image centered around a particular object, as well as relevant information.
    Attributes
    ----------
    cropped_image : array_like
        An array-like of the cropped image taken from the original image.
    left : int
        An int describing the position of the left border of the crop's bounding box in the original image.
    lower : int
        An int describing the position of the bottom border of the crop's bounding box in the original image.
    right : int
        An int describing the position of the right border of the crop's bounding box in the original image.
    upper : int
        An int describing the position of the top border of the crop's bounding box in the original image.

    Methods
    -------
    find(object_name: str) -> List[ImagePatch]
        Returns a list of new ImagePatch objects containing crops of the image centered around any objects found in the image matching the object_name.
    simple_query(question: str=None) -> str
        Returns the answer to a basic question asked about the image. If no question is provided, returns the answer to "What is this?".
    exists(object_name: str) -> bool
        Returns True if the object specified by object_name is found in the image, and False otherwise.
    verify_property(property: str) -> bool
        Returns True if the property is met, and False otherwise.
    compute_depth()->float
        Returns the median depth of the image crop.
    best_text_match(string1: str, string2: str) -> str
        Returns the string that best matches the image.
    crop(left: int, lower: int, right: int, upper: int) -> ImagePatch
        Returns a new ImagePatch object containing a crop of the image at the given coordinates.
    """

    def __init__(self, image, left: int = None, lower: int = None, right: int = None, upper: int = None):
        """Initializes an ImagePatch object by cropping the image at the given coordinates and stores the coordinates as attributes.
        If no coordinates are provided, the image is left unmodified, and the coordinates are set to the dimensions of the image.
        Parameters
        -------
        image : array_like
            An array-like of the original image.
        left : int
            An int describing the position of the left border of the crop's bounding box in the original image.
        lower : int
            An int describing the position of the bottom border of the crop's bounding box in the original image.
        right : int
            An int describing the position of the right border of the crop's bounding box in the original image.
        upper : int
            An int describing the position of the top border of the crop's bounding box in the original image.
        """
        if left is None and right is None and upper is None and lower is None:
            self.cropped_image = image
            self.left = 0
            self.lower = 0
            self.right = image.shape[2]  # width
            self.upper = image.shape[1]  # height
        else:
            self.cropped_image = image[:, lower:upper, left:right]
            self.left = left
            self.upper = upper
            self.right = right
            self.lower = lower

        self.width = self.cropped_image.shape[2]
        self.height = self.cropped_image.shape[1]

        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.lower + self.upper) / 2

    def find(self, object_name: str) -> List["ImagePatch"]:
        """Returns a new ImagePatch object containing the crop of the image centered around the object specified by object_name.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.

        Examples
        --------
        >>> # Given an image: Find the foo.
        >>> def execute_command(image) -> List[ImagePatch]:
        >>>     image_patch = ImagePatch(image)
        >>>     foo_patches = image_patch.find("foo")
        >>>     return foo_patches
        """
        return find_in_image(self.cropped_image, object_name)

    def simple_query(self, question: str = None) -> str:
        """Returns the answer to a basic question asked about the image. If no question is provided, returns the answer to "What is this?".
        Parameters
        -------
        question : str
            A string describing the question to be asked.

        Examples
        -------
        >>> # Given an image: Which kind of animal is not eating?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     animal_patches = image_patch.find("animal")
        >>>     for animal_patch in animal_patches:
        >>>         if not animal_patch.verify_property("animal", "eating"):
        >>>             return animal_patch.simple_query("What kind of animal is eating?") # crop would include eating so keep it in the query
        >>>     # If no animal is not eating, query the image directly
        >>>     return image_patch.simple_query("Which kind of animal is not eating?")

        >>> # Given an image: What is in front of the horse?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     # contains a relation (around, next to, on, near, on top of, in front of, behind, etc), so ask directly
        >>>     return image_patch.simple_query("What is in front of the horse?")
        """
        return simple_qa(self.cropped_image, question)

    def exists(self, object_name: str) -> bool:
        """Returns True if the object specified by object_name is found in the image, and False otherwise.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.

        Examples
        -------
        >>> # Given an image: Are there both cakes and gummy bears in the photo?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     is_cake = image_patch.exists("cake")
        >>>     is_gummy_bear = image_patch.exists("gummy bear")
        >>>     return bool_to_yesno(is_cake and is_gummy_bear)
        """
        return len(self.find(object_name)) > 0

    def verify_property(self, object_name: str, property: str) -> bool:
        """Returns True if the object possesses the property, and False otherwise.
        Differs from 'exists' in that it presupposes the existence of the object specified by object_name, instead checking whether the object possesses the property.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        property : str
            A string describing the property to be checked.

        Examples
        -------
        >>> # Given an image: Do the letters have blue color?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     letters_patches = image_patch.find("letters")
        >>>     # Question assumes only one letter patch
        >>>     if len(letters_patches) == 0:
        >>>         # If no letters are found, query the image directly
        >>>         return image_patch.simple_query("Do the letters have blue color?")
        >>>     return bool_to_yesno(letters_patches[0].verify_property("letters", "blue"))
        """
        return verify_property(self.cropped_image, object_name, property)

    def compute_depth(self):
        """Returns the median depth of the image crop
        Parameters
        ----------
        Returns
        -------
        float
            the median depth of the image crop

        Examples
        --------
        >>> # Given an image: Find the bar furthest away.
        >>> def execute_command(image)->ImagePatch:
        >>>     image_patch = ImagePatch(image)
        >>>     bar_patches = image_patch.find("bar")
        >>>     bar_patches.sort(key=lambda bar: bar.compute_depth())
        >>>     return bar_patches[-1]
        """
        depth_map = compute_depth(self.cropped_image)
        return depth_map.median()

    def best_text_match(self, option_list: List[str]) -> str:
        """Returns the string that best matches the image.
        Parameters
        -------
        option_list : str
            A list with the names of the different options
        prefix : str
            A string with the prefixes to append to the options

        Examples
        -------
        >>> # Given an image: Is the cap gold or white?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     cap_patches = image_patch.find("cap")
        >>>     # Question assumes one cap patch
        >>>     if len(cap_patches) == 0:
        >>>         # If no cap is found, query the image directly
        >>>         return image_patch.simple_query("Is the cap gold or white?")
        >>>     return cap_patches[0].best_text_match(["gold", "white"])
        """
        return best_text_match(self.cropped_image, option_list)

    def crop(self, left: int, lower: int, right: int, upper: int) -> "ImagePatch":
        """Returns a new ImagePatch cropped from the current ImagePatch.
        Parameters
        -------
        left : int
            The leftmost pixel of the cropped image.
        lower : int
            The lowest pixel of the cropped image.
        right : int
            The rightmost pixel of the cropped image.
        upper : int
            The uppermost pixel of the cropped image.
        -------
        """
        return ImagePatch(self.cropped_image, left, lower, right, upper)


def best_image_match(list_patches: List[ImagePatch], content: List[str], return_index=False) -> Union[ImagePatch, int]:
    """Returns the patch most likely to contain the content.
    Parameters
    ----------
    list_patches : List[ImagePatch]
    content : List[str]
        the object of interest
    return_index : bool
        if True, returns the index of the patch most likely to contain the object

    Returns
    -------
    int
        Patch most likely to contain the object
    """
    return best_image_match(list_patches, content, return_index)


def distance(patch_a: ImagePatch, patch_b: ImagePatch) -> float:
    """
    Returns the distance between the edges of two ImagePatches. If the patches overlap, it returns a negative distance
    corresponding to the negative intersection over union.

    Parameters
    ----------
    patch_a : ImagePatch
    patch_b : ImagePatch

    Examples
    --------
    # Return the qux that is closest to the foo
    >>> def execute_command(image):
    >>>     image_patch = ImagePatch(image)
    >>>     qux_patches = image_patch.find('qux')
    >>>     foo_patches = image_patch.find('foo')
    >>>     foo_patch = foo_patches[0]
    >>>     qux_patches.sort(key=lambda x: distance(x, foo_patch))
    >>>     return qux_patches[0]
    """
    return distance(patch_a, patch_b)


# Examples of using ImagePatch


# Given two images, one on the left and one on the right: Is the statement true? A person is modeling the mittens in the image on the right.
def execute_command(image_dict) -> str:
    image_patch = ImagePatch(image_dict['right'])
    return image_patch.simple_query("Is there a person modeling the mittens?")


# Given two images, one on the left and one on the right: Is the statement true? One image contains exactly three devices, and the other image features one central device with its screen open to nearly 90-degrees.
def execute_command(image_dict) -> str:
    for image_first, image_second in [[image_dict['left'], image_dict['right']],
                                      [image_dict['right'], image_dict['left']]]:
        image_first = ImagePatch(image_first)
        image_second = ImagePatch(image_second)
        first_device_patches = image_first.find('device')
        second_device_patches = image_second.find('device')
        if len(first_device_patches) == 3 and len(second_device_patches) == 1:
            answer = image_second.simple_query("Is the device's screen open to nearly 90-degrees?")
            if answer == "yes":
                return "yes"
    return "no"


# Given two images, one on the left and one on the right: Is the statement true? Each image includes at least one soda bottle shaped gummy candy, with a brown bottom half and clear top half, and no gummy soda bottles are in wrappers.
def execute_command(image_dict) -> str:
    for image in image_dict.values():
        image = ImagePatch(image)
        gummy_candy_patches = image.find('gummy candy')
        count = 0
        for gummy_candy_patch in gummy_candy_patches:
            if gummy_candy_patch.simple_query("Does the shape of gummy candy look like a soda bottle?") == "yes":
                if gummy_candy_patch.simple_query("Is the gummy candy in wrappers?") == "yes":
                    return "no"
                if gummy_candy_patch.simple_query("Is the top half clear?") == "yes":
                    if gummy_candy_patch.simple_query("Is the bottom half brown?") == "yes":
                        count += 1
        if count == 0:
            return "no"
    return "yes"


# Given two images, one on the left and one on the right: Is the statement true? The left image shows a group of no more than five people, including at least three women, sitting on something while looking at their phones.
def execute_command(image_dict) -> str:
    image_patch = ImagePatch(image_dict['left'])
    people_patches = image_patch.find('people')
    if len(people_patches) <= 5:
        count = 0
        for person_patch in people_patches:
            if person_patch.simple_query("Is this a woman?") == "yes":
                if person_patch.simple_query("Is the person sitting?") == "yes":
                    if person_patch.simple_query("Is the person looking at the phone?") == "yes":
                        count += 1
        if count >= 3:
            return 'yes'
    return 'no'


# Given two images, one on the left and one on the right: Is the statement true? There is exactly one lid.
def execute_command(image_dict) -> str:
    lid_patches = []
    for image_patch in image_dict.values():
        image_patch = ImagePatch(image_patch)
        lid_patches += image_patch.find('lid')
    return bool_to_yesno(len(lid_patches) == 1)


# Given two images, one on the left and one on the right: Is the statement true? A person is holding a syringe.
def execute_command(image_dict) -> str:
    for image_patch in image_dict.values():
        person_patches = image_patch.find('person')
        for person_patch in person_patches:
            if person_patch.simple_query("Is the person holding a syringe?") == "yes":
                return "yes"
    return "no"


# Given two images, one on the left and one on the right: Is the statement true? Only two zebras have their heads up.
def execute_command(image_dict) -> str:
    count = 0
    for image_patch in image_dict.values():
        image_patch = ImagePatch(image_patch)
        zebra_patches = image_patch.find('zebra')
        for zebra_patch in zebra_patches:
            if zebra_patch.simple_query("Is the zebra's head up?") == "yes":
                count += 1
    return bool_to_yesno(count == 2)


# INSERT_QUERY_HERE
