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


# Given an image: What toy is wearing a shirt?
def execute_command(image) -> str:
    # not a relational verb so go step by step
    image_patch = ImagePatch(image)
    toy_patches = image_patch.find("toy")
    # Question assumes only one toy patch
    if len(toy_patches) == 0:
        # If no toy is found, query the image directly
        return image_patch.simple_query("What toy is wearing a shirt?")
    for toy_patch in toy_patches:
        is_wearing_shirt = (toy_patch.simple_query("Is the toy wearing a shirt?") == "yes")
        if is_wearing_shirt:
            return toy_patch.simple_query(
                "What toy is wearing a shirt?")  # crop would include the shirt so keep it in the query
    # If no toy is wearing a shirt, pick the first toy
    return toy_patches[0].simple_query("What toy is wearing a shirt?")


# Given an image: Who is the man staring at?
def execute_command(image) -> str:
    # asks for the predicate of a relational verb (staring at), so ask directly
    image_patch = ImagePatch(image)
    return image_patch.simple_query("Who is the man staring at?")


# Given an image: Find more visible chair.
def execute_command(image) -> ImagePatch:
    # Return the chair
    image_patch = ImagePatch(image)
    # Remember: return the chair
    return image_patch.find("chair")[0]


# Given an image: Find lamp on the bottom.
def execute_command(image) -> ImagePatch:
    # Return the lamp
    image_patch = ImagePatch(image)
    lamp_patches = image_patch.find("lamp")
    lamp_patches.sort(key=lambda lamp: lamp.vertical_center)
    # Remember: return the lamp
    return lamp_patches[0]  # Return the bottommost lamp


# Given a list of images: Does the pole that is near a building that is near a green sign and the pole that is near bushes that are near a green sign have the same material?
def execute_command(image_list) -> str:
    material_1 = None
    material_2 = None
    for image in image_list:
        image = ImagePatch(image)
        # find the building
        building_patches = image.find("building")
        for building_patch in building_patches:
            poles = building_patch.find("pole")
            signs = building_patch.find("sign")
            greensigns = [sign for sign in signs if sign.verify_property('sign', 'green')]
            if len(poles) > 0 and len(greensigns) > 0:
                material_1 = poles[0].simple_query("What is the material of the pole?")
        # find the bush
        bushes_patches = image.find("bushes")
        for bushes_patch in bushes_patches:
            poles = bushes_patch.find("pole")
            signs = bushes_patch.find("sign")
            greensigns = [sign for sign in signs if sign.verify_property('sign', 'green')]
            if len(poles) > 0 and len(greensigns) > 0:
                material_2 = poles[0].simple_query("What is the material of the pole?")
    return bool_to_yesno(material_1 == material_2)


# Given an image: Find middle kid.
def execute_command(image) -> ImagePatch:
    # Return the kid
    image_patch = ImagePatch(image)
    kid_patches = image_patch.find("kid")
    if len(kid_patches) == 0:
        kid_patches = [image_patch]
    kid_patches.sort(key=lambda kid: kid.horizontal_center)
    # Remember: return the kid
    return kid_patches[len(kid_patches) // 2]  # Return the middle kid


# Given an image: Is that blanket to the right of a pillow?
def execute_command(image) -> str:
    image_patch = ImagePatch(image)
    blanket_patches = image_patch.find("blanket")
    # Question assumes only one blanket patch
    if len(blanket_patches) == 0:
        # If no blanket is found, query the image directly
        return image_patch.simple_query("Is that blanket to the right of a pillow?")
    for blanket_patch in blanket_patches:
        pillow_patches = image_patch.find("pillow")
        for pillow_patch in pillow_patches:
            if pillow_patch.horizontal_center > blanket_patch.horizontal_center:
                return "yes"
    return "no"


# Given an image: How many people are there?
def execute_command(image) -> str:
    image_patch = ImagePatch(image)
    person_patches = image_patch.find("person")
    return str(len(person_patches))


# Given a list of images: Is the man that is wearing dark pants driving?.
def execute_command(image_list) -> str:
    for image in image_list:
        image = ImagePatch(image)
        man_patches = image.find("man")
        for man_patch in man_patches:
            pants = man_patch.find("pants")
            if len(pants) == 0:
                continue
            if pants[0].verify_property("pants", "dark"):
                return man_patch.simple_query("Is this man driving?")
    return  ImagePatch(image_list[0]).simple_query("Is the man that is wearing dark pants driving?")


# Given an image: Is there a backpack to the right of the man?
def execute_command(image) -> str:
    image_patch = ImagePatch(image)
    man_patches = image_patch.find("man")
    # Question assumes one man patch
    if len(man_patches) == 0:
        # If no man is found, query the image directly
        return image_patch.simple_query("Is there a backpack to the right of the man?")
    man_patch = man_patches[0]
    backpack_patches = image_patch.find("backpack")
    # Question assumes one backpack patch
    if len(backpack_patches) == 0:
        return "no"
    for backpack_patch in backpack_patches:
        if backpack_patch.horizontal_center > man_patch.horizontal_center:
            return "yes"
    return "no"


# Given a list of images: What is the pizza with red tomato on it on?
def execute_command(image_list) -> str:
    for image in image_list:
        image = ImagePatch(image)
        pizza_patches = image.find("pizza")
        for pizza_patch in pizza_patches:
            tomato_patches = pizza_patch.find("tomato")
            has_red_tomato = False
            for tomato_patch in tomato_patches:
                if tomato_patch.verify_property("tomato", "red"):
                    has_red_tomato = True
            if has_red_tomato:
                return pizza_patch.simple_query("What is the pizza on?")
    return ImagePatch(image_list[0]).simple_query("What is the pizza with red tomato on it on?")


# Given an image: Find chair to the right near the couch.
def execute_command(image) -> ImagePatch:
    # Return the chair
    image_patch = ImagePatch(image)
    chair_patches = image_patch.find("chair")
    if len(chair_patches) == 0:
        chair_patches = [image_patch]
    elif len(chair_patches) == 1:
        return chair_patches[0]
    chair_patches_right = [c for c in chair_patches if c.horizontal_center > image_patch.horizontal_center]
    couch_patches = image_patch.find("couch")
    if len(couch_patches) == 0:
        couch_patches = [image_patch]
    couch_patch = couch_patches[0]
    chair_patches_right.sort(key=lambda c: distance(c, couch_patch))
    chair_patch = chair_patches_right[0]
    # Remember: return the chair
    return chair_patch


# Given an image: Are there bagels or lemons?
def execute_command(image) -> str:
    image_patch = ImagePatch(image)
    is_bagel = image_patch.exists("bagel")
    is_lemon = image_patch.exists("lemon")
    return bool_to_yesno(is_bagel or is_lemon)


# Given an image: In which part is the bread, the bottom or the top?
def execute_command(image) -> str:
    image_patch = ImagePatch(image)
    bread_patches = image_patch.find("bread")
    # Question assumes only one bread patch
    if len(bread_patches) == 0:
        # If no bread is found, query the image directly
        return image_patch.simple_query("In which part is the bread, the bottom or the top?")
    if bread_patches[0].vertical_center < image_patch.vertical_center:
        return "bottom"
    else:
        return "top"


# Given an image: Find foo to bottom left.
def execute_command(image) -> ImagePatch:
    # Return the foo
    image_patch = ImagePatch(image)
    foo_patches = image_patch.find("foo")
    lowermost_coordinate = min([patch.vertical_center for patch in foo_patches])
    foo_patches_bottom = [patch for patch in foo_patches if patch.vertical_center - lowermost_coordinate < 100]
    if len(foo_patches_bottom) == 0:
        foo_patches_bottom = foo_patches
    elif len(foo_patches_bottom) == 1:
        return foo_patches_bottom[0]
    foo_patches_bottom.sort(key=lambda foo: foo.horizontal_center)
    foo_patch = foo_patches_bottom[0]
    # Remember: return the foo
    return foo_patch


# Given an image: Find number 17.
def execute_command(image) -> ImagePatch:
    # Return the person
    image_patch = ImagePatch(image)
    person_patches = image_patch.find("person")
    for patch in person_patches:
        if patch.exists("17"):
            return patch
    # Remember: return the person
    return person_patches[0]


# Given a list of images: Is the statement true? There is at least 1 image with a brown dog that is near a bicycle and is wearing a collar.
def execute_command(image_list) -> str:
    for image in image_list:
        image = ImagePatch(image)
        dog_patches = image.find("dog")
        for dog in dog_patches:
            near_bicycle = dog.simple_query("Is the dog near a bicycle?")
            wearing_collar = dog.simple_query("Is the dog wearing a collar?")
            if near_bicycle == "yes" and wearing_collar == "yes":
                return 'yes'
    return 'no'


# Given an image: Find dog to the left of the post who is closest to girl wearing a shirt with text that says "I love you".
def execute_command(image) -> ImagePatch:
    # Return the dog
    image_patch = ImagePatch(image)
    shirt_patches = image_patch.find("shirt")
    if len(shirt_patches) == 0:
        shirt_patches = [image_patch]
    shirt_patch = best_image_match(list_patches=shirt_patches, content=["I love you shirt"])
    post_patches = image_patch.find("post")
    post_patches.sort(key=lambda post: distance(post, shirt_patch))
    post_patch = post_patches[0]
    dog_patches = image_patch.find("dog")
    dogs_left_patch = [dog for dog in dog_patches if dog.left < post_patch.left]
    if len(dogs_left_patch) == 0:
        dogs_left_patch = dog_patches
    dogs_left_patch.sort(key=lambda dog: distance(dog, post_patch))
    dog_patch = dogs_left_patch[0]
    # Remember: return the dog
    return dog_patch


# Given an image: Find balloon on the right and second from the bottom.
def execute_command(image) -> ImagePatch:
    # Return the balloon
    image_patch = ImagePatch(image)
    balloon_patches = image_patch.find("balloon")
    if len(balloon_patches) == 0:
        balloon_patches = [image_patch]
    elif len(balloon_patches) == 1:
        return balloon_patches[0]
    leftmost_coordinate = min([patch.horizontal_center for patch in balloon_patches])
    balloon_patches_right = [patch for patch in balloon_patches if patch.horizontal_center - leftmost_coordinate < 100]
    if len(balloon_patches_right) == 0:
        balloon_patches_right = balloon_patches
    balloon_patches_right.sort(key=lambda p: p.vertical_center)
    balloon_patch = balloon_patches_right[1]
    # Remember: return the balloon
    return balloon_patch


# Given an image: Find girl in white next to man in left.
def execute_command(image) -> ImagePatch:
    # Return the girl
    image_patch = ImagePatch(image)
    girl_patches = image_patch.find("girl")
    girl_in_white_patches = [g for g in girl_patches if g.verify_property("girl", "white clothing")]
    if len(girl_in_white_patches) == 0:
        girl_in_white_patches = girl_patches
    man_patches = image_patch.find("man")
    man_patches.sort(key=lambda man: man.horizontal_center)
    leftmost_man = man_patches[0]  # First from the left
    girl_in_white_patches.sort(key=lambda girl: distance(girl, leftmost_man))
    girl_patch = girl_in_white_patches[0]
    # Remember: return the girl
    return girl_patch


# Given a list of images: Is the statement true? There is 1 table that is in front of woman that is wearing jacket.
def execute_command(image_list) -> str:
    for image in image_list:
        image = ImagePatch(image)
        woman_patches = image.find("woman")
        for woman in woman_patches:
            if woman.simple_query("Is the woman wearing jacket?") == "yes":
                tables = woman.find("table")
                return bool_to_yesno(len(tables) == 1)
    return 'no'


# Given an image: Find top left.
def execute_command(image) -> ImagePatch:
    # Return the person
    image_patch = ImagePatch(image)
    # Figure out what thing the caption is referring to. We need a subject for every caption
    persons = image_patch.find("person")
    top_all_objects = max([obj.vertical_center for obj in persons])
    # Select objects that are close to the top
    # We do this because the caption is asking first about vertical and then about horizontal
    persons_top = [p for p in persons if top_all_objects - p.vertical_center < 100]
    if len(persons_top) == 0:
        persons_top = persons
    # And after that, obtain the leftmost object among them
    persons_top.sort(key=lambda obj: obj.horizontal_center)
    person_leftmost = persons_top[0]
    # Remember: return the person
    return person_leftmost


# Given an image: What type of weather do you see in the photograph?
def execute_command(image) -> str:
    image_patch = ImagePatch(image)
    return image_patch.simple_query("What type of weather do you see in the photograph?")


# Given an image: How many orange life vests can be seen?
def execute_command(image) -> str:
    image_patch = ImagePatch(image)
    life_vest_patches = image_patch.find("life vest")
    orange_life_vest_patches = []
    for life_vest_patch in life_vest_patches:
        if life_vest_patch.verify_property('life vest', 'orange'):
            orange_life_vest_patches.append(life_vest_patch)
    return str(len(orange_life_vest_patches))


# Given an image: What is behind the pole?
def execute_command(image) -> str:
    image_patch = ImagePatch(image)
    # contains a relation (around, next to, on, near, on top of, in front of, behind, etc), so ask directly
    return image_patch.simple_query("What is behind the pole?")


# Given an image: Find second to top flower.
def execute_command(image) -> ImagePatch:
    # Return the flower
    image_patch = ImagePatch(image)
    flower_patches = image_patch.find("flower")
    flower_patches.sort(key=lambda flower: flower.vertical_center)
    flower_patch = flower_patches[-2]
    # Remember: return the flower
    return flower_patch


# Given an image: Find back.
def execute_command(image) -> ImagePatch:
    # Return the person
    image_patch = ImagePatch(image)
    person_patches = image_patch.find("person")
    person_patches.sort(key=lambda person: person.compute_depth())
    person_patch = person_patches[-1]
    # Remember: return the person
    return person_patch


# Given an image: Find chair at the front.
def execute_command(image) -> ImagePatch:
    # Return the chair
    image_patch = ImagePatch(image)
    chair_patches = image_patch.find("chair")
    chair_patches.sort(key=lambda chair: chair.compute_depth())
    chair_patch = chair_patches[0]
    # Remember: return the chair
    return chair_patch


# Given an image: Find white and yellow pants.
def execute_command(image) -> ImagePatch:
    # Return the person
    image_patch = ImagePatch(image)
    # Clothing always requires returning the person
    person_patches = image_patch.find("person")
    person_patch = best_image_match(person_patches, ["white pants", "yellow pants"])
    # Remember: return the person
    return person_patch


# Given an image: Find cow facing the camera.
def execute_command(image) -> ImagePatch:
    # Return the cow
    image_patch = ImagePatch(image)
    cow_patches = image_patch.find("cow")
    if len(cow_patches) == 0:
        cow_patches = [image_patch]
    cow_patch = best_image_match(list_patches=cow_patches, content=["cow facing the camera"])
    # Remember: return the cow
    return cow_patch


# Given a list of images: Is the statement true? There is 1 image that contains exactly 3 blue papers.
def execute_command(image_list) -> str:
    image_cnt = 0
    for image in image_list:
        image = ImagePatch(image)
        paper_patches = image.find("paper")
        blue_paper_patches = []
        for paper in paper_patches:
            if paper.verify_property("paper", "blue"):
                blue_paper_patches.append(paper)
        if len(blue_paper_patches) == 3:
            image_cnt += 1
    return bool_to_yesno(image_cnt == 1)


# Given an image: Find black car just under stop sign.
def execute_command(image) -> ImagePatch:
    # Return the car
    image_patch = ImagePatch(image)
    stop_sign_patches = image_patch.find("stop sign")
    if len(stop_sign_patches) == 0:
        stop_sign_patches = [image_patch]
    stop_sign_patch = stop_sign_patches[0]
    car_patches = image_patch.find("black car")
    car_under_stop = []
    for car in car_patches:
        if car.upper < stop_sign_patch.upper:
            car_under_stop.append(car)
    # Find car that is closest to the stop sign
    car_under_stop.sort(key=lambda car: car.vertical_center - stop_sign_patch.vertical_center)
    # Remember: return the car
    return car_under_stop[0]


# Given a list of images: Is there either a standing man that is holding a cell phone or a sitting man that is holding a cell phone?
def execute_command(image_list) -> str:
    for image in image_list:
        image = ImagePatch(image)
        man_patches = image.find("man")
        for man in man_patches:
            holding_cell_phone = man.simple_query("Is this man holding a cell phone?")
            if holding_cell_phone == "yes":
                if man.simple_query("Is this man sitting?") == "yes":
                    return 'yes'
                if man.simple_query("Is this man standing?") == "yes":
                    return 'yes'
    return 'no'


# Given a list of images: How many people are running while looking at their cell phone?
def execute_command(image) -> str:
    image_patch = ImagePatch(image)
    people_patches = image_patch.find("person")
    # Question assumes only one person patch
    if len(people_patches) == 0:
        # If no people are found, query the image directly
        return image_patch.simple_query("How many people are running while looking at their cell phone?")
    people_count = 0
    for person_patch in people_patches:
        # Verify two conditions: (1) running (2) looking at cell phone
        if person_patch.simple_query("Is the person running?") == "yes":
            if person_patch.simple_query("Is the person looking at cell phone?") == "yes":
                people_count += 1
    return str(people_count)


# Given a list of images: Does the car that is on a highway and the car that is on a street have the same color?
def execute_command(image_list) -> str:
    color_1 = None
    color_2 = None
    for image in image_list:
        image = ImagePatch(image)
        car_patches = image.find("car")
        for car_patch in car_patches:
            if car_patch.simple_query("Is the car on the highway?") == "yes":
                color_1 = car_patch.simple_query("What is the color of the car?")
            elif car_patch.simple_query("Is the car on a street?") == "yes":
                color_2 = car_patch.simple_query("What is the color of the car?")
    return bool_to_yesno(color_1 == color_2)


# Given a list of images: Is the statement true? There are 3 magazine that are on table.
def execute_command(image_list) -> str:
    count = 0
    for image in image_list:
        image = ImagePatch(image)
        magazine_patches = image.find("magazine")
        for magazine_patch in magazine_patches:
            on_table = magazine_patch.simple_query("Is the magazine on a table?")
            if on_table == "yes":
                count += 1
    return bool_to_yesno(count == 3)


# INSERT_QUERY_HERE