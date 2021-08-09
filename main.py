# LexiSign - Sanjay Ravishankar
# STEMWarriorHacks 2021

# Import libraries
import os
import math
import cv2
import mediapipe

# Initialize MediaPipe frameworks
drawing_library = mediapipe.solutions.drawing_utils
hand_library = mediapipe.solutions.hands
hands = hand_library.Hands(max_num_hands=1)
points = 21
# List of connections between important points
connections = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20], [5, 9], [9, 13], [13, 17], [5, 8], [9, 12], [13, 16], [17, 20], [4, 8], [4, 12], [4, 16], [4, 20]]
connections_to_draw = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20], [5, 9], [9, 13], [13, 17]]

# Bounding box function
def within_box(user, box_1, box_2):
    if box_1 <= user <= box_2:
        return True
    return False

# Magnitude function (distance)
def magnitude(v):
    return math.sqrt(pow(v[0], 2) + pow(v[1], 2))

# Direction function (slope)
def direction(v):
    if v[0]: # Avoid division by 0
        return v[1] / v[0]
    return float("inf")

# Function for MediaPipe hand tracking
def get_landmarks(img, draw=False, recurse=0):
    img_copy = img
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        if draw:
            drawing_library.draw_landmarks(img, hand, hand_library.HAND_CONNECTIONS)
        landmark = hand.landmark
        for lm in landmark:
            # Scale xy coordinate values
            lm.x = int(lm.x * img.shape[1])
            lm.y = int(lm.y * img.shape[0])
        return landmark
    if recurse:
        recurse -= 1
        return get_landmarks(img_copy, draw, recurse)

# Read and store data from image files
alphabet_list = []
alphabet_vectors = []
for file in os.listdir("alphabet/reference/"):
    img_r = cv2.imread("alphabet/reference/" + file)
    img_l = cv2.flip(img_r, 1)
    images = [img_r, img_l]
    for img in images:
        landmarks = get_landmarks(img, recurse=10)
        if landmarks:
            vectors = []
            for c in connections:
                vectors.append((landmarks[c[1]].x - landmarks[c[0]].x, landmarks[c[1]].y - landmarks[c[0]].y))
        alphabet_vectors.append(vectors)
        alphabet_list.append(file[0].lower())

# Transcribing text variables
transcribing = False
speed = 30
current_letter = None
letter_occurrences = 0
help_text = "Welcome to LexiSign! Transcribe the ASL alphabet to text. Click 'Start' to begin transcription, and click 'Pause' or 'Stop' to end. Clicking 'Stop' will also clear the transcribed text. Click 'Help' to see this message again."
text = help_text
dot_text = "."
dot_text_occurrences = 0

# Image analysis variables
magnitude_limit = 30
direction_limit = 10
minimum_score = len(connections) * 0.75

# Read camera feed and display window
logo = cv2.imread("images/logo.png")
buttons = cv2.imread("images/buttons.png")
cv2.namedWindow("LexiSign")

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise Exception("Error: couldn't open camera")

while cv2.getWindowProperty("LexiSign", 0) >= 0:
    letter = None
    # Get resized frame image from camera
    success, frame = cam.read()
    h, w, c = frame.shape
    if w <= 400:
        frame = cv2.resize(frame, (400, int(h * (400 / w))))
    if h <= 400:
        frame = cv2.resize(frame, (int(w * (400 / h)), 400))
    frame = frame[0:400, 0:400]
    # Get landmarks of image
    landmarks = get_landmarks(frame)
    if landmarks:
        # Calculate vectors and draw lines
        vectors = []
        for c in connections:
            if c in connections_to_draw:
                cv2.line(frame, (int(landmarks[c[0]].x), int(landmarks[c[0]].y)),
                     (int(landmarks[c[1]].x), int(landmarks[c[1]].y)), (255, 191, 229), 2)
            vectors.append((landmarks[c[1]].x - landmarks[c[0]].x, landmarks[c[1]].y - landmarks[c[0]].y))

        # Classify image
        letter_magnitude_scores = []
        letter_direction_scores = []
        # Validate differences between vectors
        for letter_vectors in alphabet_vectors:
            distance_score = 0
            slope_score = 0
            for j, letter_vector in enumerate(letter_vectors):
                scale_factor_x = letter_vectors[0][0] / (vectors[0][0] if vectors[0][0] != 0 else 0.001)
                scale_factor_y = letter_vectors[0][1] / (vectors[0][1] if vectors[0][1] != 0 else 0.001)
                frame_vector = (vectors[j][0] * scale_factor_x, vectors[j][1] * scale_factor_y)
                difference_vector = (frame_vector[0] - letter_vector[0], frame_vector[1] - letter_vector[1])
                if magnitude(difference_vector) <= magnitude_limit:
                    distance_score += 1
                if abs(direction(difference_vector)) <= direction_limit:
                    slope_score += 1
            letter_magnitude_scores.append(distance_score)
            letter_direction_scores.append(slope_score)

        # Find correct letter using recursive function
        def find_letter():
            if not len(letter_magnitude_scores):
                return
            highest_distance_score_index = letter_magnitude_scores.index(max(letter_magnitude_scores))
            corresponding_slope_score = letter_direction_scores[highest_distance_score_index]
            if corresponding_slope_score >= minimum_score:
                return alphabet_list[highest_distance_score_index]
            letter_magnitude_scores.pop(highest_distance_score_index)
            letter_direction_scores.pop(highest_distance_score_index)
            return find_letter()
        letter = find_letter()

        # Update text
        if letter:
            cv2.putText(frame, letter, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 5)
            if transcribing:
                if current_letter == letter:
                    letter_occurrences += 1
                    if letter_occurrences >= speed:
                        text += letter
                        letter_occurrences = 0
                else:
                    current_letter = letter
                    letter_occurrences = 1

    else:
        if transcribing:
            # Add continual spaces to text if no landmarks are found
            if not current_letter:
                letter_occurrences += 1
                if letter_occurrences == speed:
                    text += " "
                    letter_occurrences = 0
            else:
                current_letter = None
                letter_occurrences = 1

            # Update dot text
            dot_text_occurrences += 1
            if dot_text_occurrences == int(speed / 3):
                dot_text += "."
                dot_text_occurrences = 0
                if dot_text == "....":
                    dot_text = "."
        else:
            if dot_text != ".":
                dot_text = "."

    # Render window
    blank = cv2.imread("images/blank.png")
    blank[0:400, 0:400] = frame
    blank[400:500, 0:300] = logo
    if letter:
        blank[400:500, 300:400] = cv2.resize(cv2.imread("alphabet/profile/" + letter + ".png"), (100, 100))

    # On-screen buttons
    blank[400:500, 400:700] = buttons
    def push_button(event, x, y, flags, param):
        if event == 4:
            user = (x, y)
            global transcribing
            global help_text
            global text
            stop_text = "Transcribing stopped. Click 'Start' to transcribe again."
            if within_box(user, (400, 400), (475, 500)):
                # Start
                transcribing = True
                if text == stop_text or text == help_text:
                    text = ""
            if within_box(user, (475, 400), (550, 500)):
                # Pause
                transcribing = False
            if within_box(user, (550, 400), (625, 500)):
                # Stop
                transcribing = False
                text = "Transcribing stopped. Click 'Start' to transcribe again."
            if within_box(user, (625, 400), (700, 500)):
                # Help
                transcribing = False
                text = help_text
    cv2.setMouseCallback("LexiSign", push_button)

    # Format and display transcribed text
    cv2.rectangle(blank, (400, 0), (700, 400), (60, 60, 60), -1)
    q = 23
    word_list = [i + " " for i in text.split()]
    temp_string = ""
    lines = []
    for w in word_list:
        if len(temp_string + w) <= q:
            temp_string += w
        else:
            lines.append(temp_string)
            temp_string = w
    lines.append(temp_string)
    for i, l in enumerate(lines):
        display_text = l
        if l == lines[-1] and transcribing:
            display_text += dot_text
        cv2.putText(blank, display_text, (430, 50 + (i * 23)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Show window
    cv2.imshow("LexiSign", blank)
    # 10 ms delay
    cv2.waitKey(10)