import socket
import random
import string
import time


def generate_random_data(length=32):
    """Generate a string of random alphanumeric characters."""
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


def send_random_data(host, port):
    """Send random data to a specified host and port in an infinite loop."""
    while True:
        data = generate_random_data()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((host, port))
                s.sendall(data.encode('utf-8'))
                #print(f'Sent: {data}')
            except ConnectionError as e:
                print(f'Connection error: {e}')

        # Optional: Add a sleep interval to control the sending rate
        time.sleep(1)


if __name__ == "__main__":
    host = '127.0.0.1'
    port = 7000
    send_random_data(host, port)
