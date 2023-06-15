from tkinter import messagebox

from dollarpy import Template, Point, Recognizer
import os

templates_folder = 'templates'

templates = []
results = []
foot_count = 70
trace_array = []
for i in range(0, 5):
    points = []
    with open(os.path.join(templates_folder, f'template{i}.txt'), 'r') as file:
        for line in file:
            x, y = line.strip('()\n').split(',')
            points.append(Point(float(x), float(y)))
    template = Template(f'template{i}', points)
    templates.append(template)

points = []

with open('trajectory.txt', 'r') as file:
    for line in file:
        x, y = line.strip('()\n').split(',')
        points.append(Point(float(x), float(y)))

recognizer = Recognizer(templates)

length = len(points)

for x in range(length):
    trace_array.append(points[x])
    if(len(trace_array) % foot_count == 0):
        results.append(recognizer.recognize(trace_array))
        trace_array.clear()



threshold = 0.6

count = 0
for result in results:
    if result is not None and isinstance(result, tuple) and result[1] >= threshold:
        count = count + 1
        print("Result", f"The template is a match with a score of {result}")
    #else:
        #messagebox.showinfo("Result", "The template is not a match")

print(f"Total templates matched: {count}")
