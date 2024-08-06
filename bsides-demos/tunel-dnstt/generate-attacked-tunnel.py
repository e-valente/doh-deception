import os
import random
import socket
import string
import time
import numpy as np

mean = 2.0  # mean sleep time in seconds
stddev = 0.01  # standard deviation in seconds

# Number of iterations (requests)
num_requests = 1000


def generate_random_data(size=12000):
    """Generate random data."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=size))


def random_normal(mean, stddev):
    """Generate a random number with normal distribution."""
    return max(0, np.random.normal(mean, stddev))


def send_data(data, host='127.0.0.1', port=7000):
    """Send data using socket."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(data.encode())
        #print(s.recv(1024).decode())


for i in range(num_requests):
    # Generate a normally distributed random number
    sleep_time = random_normal(mean, stddev)

    # Generate random data
    data = generate_random_data()

    # Send the data using socket
    send_data(data)

    print(f"Sleeping for {sleep_time:.2f} seconds...")
    time.sleep(sleep_time)

    # Simulate a request (e.g., call a function or execute a command)
    print(f"Handling request {i}")
