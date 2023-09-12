import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import time
import numpy as np


def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]


def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]


def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1


def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    lesser = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]
    return quicksort(lesser) + equal + quicksort(greater)


def timsort(arr):
    arr.sort()


def plot_scatter_graph():
    # Initialize lists to store values
    n_values = []
    time_taken = []

    # Perform sorting algorithm for different n values
    for n in range(0, 1001):
        # Generate a random list of n elements
        arr = [random.randint(1, 1000) for _ in range(n)]

        # Measure the time taken for sorting
        start_time = time.time()
        merge_sort(arr)  # Use the sorting algorithm you want to analyze here
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds

        # Append n and time taken to the lists
        n_values.append(n)
        time_taken.append(elapsed_time)

    # Calculate average time
    average_time = np.mean(time_taken)

    # Create scatter plot
    fig, ax = plt.subplots()
    ax.scatter(n_values, time_taken, marker="x", color="red", alpha=0.5, s=5)
    ax.set_title("Sorting Algorithm Analysis")
    ax.set_xlabel("n")
    ax.set_ylabel("Time Taken (milliseconds)")

    # Add regression curve
    z = np.polyfit(n_values, time_taken, 2)
    p = np.poly1d(z)
    regression_curve = p(n_values)
    ax.plot(n_values, regression_curve, color="blue", label="Regression Curve")

    # Display equation of the regression curve
    equation_text = f"y = {p}"
    equation_text = equation_text.replace("\n", " ").replace("  ", " ")
    text_x = max(n_values) * 0.8  # Adjust the x position of the equation text
    text_y = min(time_taken) * 0.9  # Adjust the y position of the equation text
    ax.text(text_x, text_y, equation_text, fontsize=10, ha="right", va="bottom")
    ax.text(
        text_x,
        text_y - (max(time_taken) * 0.05),
        "Regression Curve",
        fontsize=10,
        ha="right",
        va="bottom",
    )

    ax.legend()
    plt.show()

    # ...


if __name__ == "__main__":
    plot_scatter_graph()
